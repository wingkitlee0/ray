import logging
import threading
import time
from collections import defaultdict
from typing import Dict, Hashable, Iterable, List, Tuple, Union

import numpy as np

import ray
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.sort import SortKey
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata
from ray.util.queue import Queue

logger = logging.getLogger(__name__)


def get_group_boundaries(
    keys: Union[np.ndarray, dict[str, np.ndarray]],
) -> Tuple[Union[np.ndarray, dict[str, np.ndarray]], np.ndarray]:
    """Find the boundaries of the groups.

    Args:
        keys: an array of keys or a dict of key columns

    Returns:
        grouped_keys: keys found
        indices: starting indices of each

    Note:
        This is a fast linear-time implementation that uses
        vectorization. This implementation does not relied
        the keys being sorted and unique.
        For example, [2, 2, 2, 1, 1, 3, 2] results in
        grouped_keys = [2, 1, 3, 2], indices = [0, 3, 5, 6]
    """

    if not isinstance(keys, dict):
        # Include the first index explicitly
        indices = np.hstack([[0], np.where(np.diff(keys) != 0)[0] + 1, [len(keys)]])
        return keys[indices[:-1]], indices

    s = {0}
    for arr in keys.values():
        s |= set(np.where(np.diff(arr) != 0)[0] + 1)
    s = sorted(s)
    s.append(len(arr))
    indices = np.array(s)

    grouped_keys = {
        key: value[indices[:-1]] for key, value in keys.items() if len(value) > 0
    }
    return grouped_keys, indices


# def _split_single_block_by_key(
#     idx: int,
#     block: Block,
#     key: SortKey,
# ) -> Tuple[List[Block], BlockMetadata]:
#     """Split the block based on the key

#     Args:
#         idx: block id (within one RefBundle)
#         block: data
#         output_num_blocks: not used as we will determine this from the key
#         key: the column name for repartition
#     """
#     import numpy as np

#     stats = BlockExecStats.builder()

#     columns = key.get_columns()
#     accessor = BlockAccessor.for_block(block)
#     keys = accessor.to_numpy(columns)

#     block_stats = get_block_stats(keys, idx)

#     indices = np.hstack([block_stats["indices"], [len(keys)]])

#     out_blocks = []
#     for start, end in zip(indices[:-1], indices[1:]):
#         out_blocks.append(accessor.slice(start, end))

#     # metadata is coming the pre-split block
#     meta = BlockAccessor.for_block(block).get_metadata(
#         input_files=None, exec_stats=stats.build()
#     )
#     return out_blocks, meta


def get_batch_stats(
    keys: Union[np.ndarray, dict[str, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given an array of keys of a given batch (arbitrary size), return the
    necessary statistics for merging

    Args:
        keys: The keys to compute stats on

    Returns:
        group_keys: The unique keys in the batch
        counts: The number of rows for each key
        indices: The split indices between groups
    """
    group_keys, indices = get_group_boundaries(keys)

    counts = np.diff(indices, append=len(keys))

    return group_keys, counts, indices


# TODO: handle block metadata
def _split_block(
    block: Block, keys: Union[str, List[str]]
) -> Tuple[List[Block], List[BlockMetadata]]:
    """Splits a block by the given keys.

    Args:
        block: The block to split.
        keys: The keys to split on.

    Returns:
        A tuple of the split blocks and their metadata.
    """
    stats = BlockExecStats.builder()

    _block = BlockAccessor.batch_to_block(block)
    block_accessor = BlockAccessor.for_block(_block)
    arr = block_accessor.to_numpy(keys)

    if len(arr) == 0:
        return None

    _, _, split_indices = get_batch_stats(arr)

    if len(split_indices) == 1:
        return [block_accessor.to_block()]

    out_blocks = []
    out_metadata = []
    for start, end in zip(split_indices[:-1], split_indices[1:]):
        out_block = block_accessor.slice(start, end)
        accessor = BlockAccessor.for_block(out_block)
        meta = accessor.get_metadata(input_files=None, exec_stats=stats.build())
        out_blocks.append(out_block)
        out_metadata.append(meta)

    return out_blocks, out_metadata


@ray.remote
class Reducer:
    def __init__(self, idx: int, keys: Union[str, List[str]]):
        """A reducer actor for a given partition."""
        self.idx = idx
        self.keys = keys
        self.mapping: Dict[Hashable, Block] = {}
        self._num_collect = 0
        self._num_blocks = 0
        self.lock = threading.Lock()

    def collect(self, blocks: list[Block]):
        """Collect a batch of single-key sub-blocks and put them into
        the mapping. This is where the actual merging happens.
        """
        with self.lock:
            for batch in blocks:
                if len(batch) == 0:
                    continue
                block = BlockAccessor.batch_to_block(batch)
                block_accessor = BlockAccessor.for_block(block)

                # get the hashable key
                _keys = block_accessor.to_numpy(self.keys)
                if isinstance(_keys, dict):
                    key = tuple(v[0] for v in _keys.values())
                else:
                    key = _keys[0]

                if key in self.mapping:
                    self.mapping[key].append(block)
                else:
                    self.mapping[key] = [block]

            self._num_collect += 1
            self._num_blocks += len(blocks)

    def get_mapping(self):
        with self.lock:
            return self.mapping

    def get_num_collect_and_blocks(self):
        with self.lock:
            return (self._num_collect, self._num_blocks)

    # TODO: how to enable/disable sort option
    # TODO: should this function be placed outside Reducer
    def put_mapping_into_queue(self, idx: int, out_queue: Queue):
        for key, values in self.mapping.items():
            accessor = BlockAccessor.for_block(values[0])
            block, block_metadata = accessor.merge_sorted_blocks(
                values, sort_key=SortKey(self.keys)
            )
            out_queue.put((idx, key, ray.put(block), block_metadata))
        return "success"


# TODO: need to figure out how to putting item with different idx into the in_queue
@ray.remote
class Coordinator:
    """Actor that coordinates between data producers and reducers.

    Attributes:
        keys (List[str]): The keys to reduce on.
        in_queue (Queue): Input queue from data producers.
        out_queue (Queue): Output queue to write merged blocks.
        reducers (Dict[int, ActorHandle]): Mapping of reducer index to actor handle.
        thread (threading.Thread): The coordinator thread.
    """

    def __init__(
        self,
        keys: Union[str, List[str]],
        in_queue: Queue,
        out_queue: Queue,
        meta_queue: Queue,
    ):
        self.keys = keys
        self.in_queue = in_queue
        self.reducers = {}
        self.thread = threading.Thread(
            target=self.run, args=(in_queue, out_queue, meta_queue)
        )
        self.thread.start()
        self.message_counts = defaultdict(int)
        self.message_counts_after_end = {}

    def run(self, in_queue, out_queue, meta_queue):
        while True:
            if in_queue.qsize() > 0:
                item: tuple = in_queue.get()
                if item[0] == "end":
                    idx_ = item[1]
                    self.message_counts_after_end[idx_] = 0
                    # self.reducers[idx_].close.remote()
                    # del self.reducers[idx_]
                    # print("current number of reducers: ", len(self.reducers))
                    # out_queue.put((idx_, self.reducers[idx_].get_mapping.remote()))
                    ray.get(
                        self.reducers[idx_].put_mapping_into_queue.remote(
                            idx_, out_queue
                        )
                    )
                    continue
                idx, data_ref, meta_ref = item
                assert isinstance(data_ref, ray.ObjectRef)

                self.message_counts[idx] += 1
                if idx in self.message_counts_after_end:
                    self.message_counts_after_end[idx] += 1

                if idx not in self.reducers:
                    self.reducers[idx] = Reducer.remote(idx, self.keys)
                    # print("current number of reducers: ", len(self.reducers))

                ray.get(self.reducers[idx].collect.remote(data_ref))
                meta_queue.put(meta_ref)
            else:
                time.sleep(0.1)

    def close(self):
        ray.actor.exit_actor()

    def get_reducers(self):
        return self.reducers

    def get_message_counts(self):
        return self.message_counts

    def get_message_counts_after_end(self):
        return self.message_counts_after_end


def process_fragment(
    i,
    blocks_per_fragments: Iterable[Block],
    queue: Queue,
    keys: Union[str, List[str]],
) -> str:
    """Submits a split-block task for each block in the fragment."""
    ctx = ray.get_runtime_context()
    task_id = ctx.get_task_id()

    split_block = cached_remote_fn(_split_block, num_returns=2)

    for block in blocks_per_fragments:
        ref, meta_ref = split_block.remote(block, keys)
        queue.put((i, ref, meta_ref))
    queue.put(("end", i, task_id))
    return "success"
