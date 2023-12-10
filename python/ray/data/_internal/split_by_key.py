import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

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
        indices = np.hstack([[0], np.where(np.diff(keys) != 0)[0] + 1])
        return keys[indices], indices

    s = {0}
    for arr in keys.values():
        s |= set(np.where(np.diff(arr) != 0)[0] + 1)
    indices = np.array(sorted(s))

    grouped_keys = {
        key: value[indices] for key, value in keys.items() if len(value) > 0
    }
    return grouped_keys, indices


@dataclass
class BlockStat:
    keys: np.ndarray
    counts: np.ndarray
    indices: np.ndarray
    batch_id: int

    def __post_init__(self):
        if not (len(self.keys) == len(self.counts) == len(self.indices)):
            raise ValueError(
                f"length not match: ({len(self.keys)},{len(self.counts)},"
                f"{len(self.indices)}"
            )

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self) -> str:
        return f"BlockStat({self.keys}, {self.counts}, {self.indices}, {self.batch_id})"


@dataclass
class SubBlockStats:
    key: Union[int, str]
    count: int
    batch_id: list[int]  # batch_id corresponding to to_batch()
    min_batch_id: int

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self) -> str:
        return (
            f"SingleBlockStat(key={self.key}, cnt={self.count},"
            f"min_batch_id={self.min_batch_id})"
        )

    def __iadd__(self, obj: "SubBlockStats") -> "SubBlockStats":
        if self.key != obj.key:
            raise ValueError("keys must be equal")

        batch_id = sorted(self.batch_id + obj.batch_id)
        min_batch_id = batch_id[0]

        # use the obj's start_idx if it's an "earlier" block
        out = SubBlockStats(
            key=self.key,
            count=self.count + obj.count,
            batch_id=batch_id,
            min_batch_id=min_batch_id,
        )
        return out


def get_block_stats(
    keys: Union[np.ndarray, dict[str, np.ndarray]], batch_id: int
) -> BlockStat:
    """Given an array of keys of a given batch (arbitrary size), return the
    necessary statistics for merging
    """
    group_keys, indices = get_group_boundaries(keys)

    counts = np.diff(indices, append=len(keys))

    return BlockStat(group_keys, counts, indices, batch_id)


def _split_single_block_by_key(
    idx: int,
    block: Block,
    key: SortKey,
) -> Tuple[List[Block], BlockMetadata]:
    """Split the block based on the key

    Args:
        idx: block id (within one RefBundle)
        block: data
        output_num_blocks: not used as we will determine this from the key
        key: the column name for repartition
    """
    import numpy as np

    stats = BlockExecStats.builder()

    columns = key.get_columns()
    accessor = BlockAccessor.for_block(block)
    keys = accessor.to_numpy(columns)

    block_stats = get_block_stats(keys, idx)

    indices = np.hstack([block_stats["indices"], [len(keys)]])

    out_blocks = []
    for start, end in zip(indices[:-1], indices[1:]):
        out_blocks.append(accessor.slice(start, end))

    # metadata is coming the pre-split block
    meta = BlockAccessor.for_block(block).get_metadata(
        input_files=None, exec_stats=stats.build()
    )
    return out_blocks, meta


# def _split_blocks_by_key(
#     blocks: List[Block],
#     key: SortKey,
# ) -> Tuple[List[List[Block]], List[BlockMetadata]]:
#     """To be wrapped as a remote function"""

#     subblocks_list, metadata_list = [], []
#     for idx, block in enumerate(blocks):
#         subblocks, metadata = _split_single_block_by_key(idx, block, key)
#         subblocks_list.append(subblocks)
#         metadata_list.append(metadata)

#     return subblocks_list, metadata_list


def get_batch_stats(keys: Union[np.ndarray, dict[str, np.ndarray]]):
    """Given an array of keys of a given batch (arbitrary size), return the
    necessary statistics for merging
    """
    group_keys, indices = get_group_boundaries(keys)

    counts = np.diff(indices, append=len(keys))

    return group_keys, counts, indices


def _split_block(block: Block, keys: Union[str, List[str]]) -> list[Block]:
    block_accessor = BlockAccessor.for_block(block)
    arr = block_accessor.to_numpy(keys)

    if len(arr) == 0:
        return None

    _, _, split_indices = get_batch_stats(arr)

    if len(split_indices) == 1:
        return block_accessor.to_block()

    out_blocks = []
    for start, end in zip(split_indices[:-1], split_indices[1:]):
        out_blocks.append(block_accessor.slice(start, end))
    return out_blocks


@ray.remote
class Reducer:
    def __init__(self, idx: int):
        """A reducer actor for a given partition."""
        print(f"Creating a reducer for partition {idx}")
        self.idx = idx
        self.mapping = {}
        self._num_collect = 0
        self._num_blocks = 0
        self.lock = threading.Lock()

    def collect(self, blocks: list[Block], keys: Union[str, List[str]]):
        """Collect a batch of single-key sub-blocks and put them into
        the mapping. This is where the actual merging happens.
        """
        with self.lock:
            for batch in blocks:
                if len(batch) == 0:
                    continue
                block = BlockAccessor.batch_to_block(batch)
                block_accessor = BlockAccessor.for_block(block)
                key_array = block_accessor.to_numpy(keys)[0, :]
                key = tuple(key_array)
                if key in self.mapping:
                    self.mapping[key].append(block)
                else:
                    self.mapping[key] = [block]

            self._num_collect += 1
            self._num_blocks += len(blocks)

    def close(self):
        with self.lock:
            print(
                f"Closing Reducer {self.idx}. It collected {self._num_collect} batches and {self._num_blocks} blocks"
            )
        # ray.actor.exit_actor()

    def get_mapping(self):
        with self.lock:
            return self.mapping

    def get_num_collect_and_blocks(self):
        with self.lock:
            return (self._num_collect, self._num_blocks)

    def put_mapping_into_queue(self, idx: int, out_queue: Queue):
        out_queue.put((idx, self.mapping))
        return "success"


# TODO: need to figure out how to putting item with different idx into the in_queue
@ray.remote
class Coordinator:
    def __init__(self, keys: Union[str, List[str]], in_queue, out_queue: Queue):
        self.keys = keys
        self.in_queue = in_queue
        self.reducers = {}
        self.thread = threading.Thread(target=self.run, args=(in_queue, out_queue))
        self.thread.start()
        self.message_counts = defaultdict(int)
        self.message_counts_after_end = {}

    def run(self, in_queue, out_queue):
        while True:
            if in_queue.qsize() > 0:
                item: tuple = in_queue.get()
                if item[0] == "end":
                    print(item)
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
                idx, data_ref = item
                assert isinstance(data_ref, ray.ObjectRef)

                self.message_counts[idx] += 1
                if idx in self.message_counts_after_end:
                    self.message_counts_after_end[idx] += 1

                if idx not in self.reducers:
                    self.reducers[idx] = Reducer.remote(idx)
                    print("current number of reducers: ", len(self.reducers))

                ray.get(self.reducers[idx].collect.remote(data_ref, self.keys))
                # self.reducers[idx].collect.remote(data_ref)
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
    blocks_per_fragments: Iterable[pa.Table],
    queue: Queue,
    keys: Union[str, List[str]],
) -> str:
    """Submits a split-block task for each block in the fragment."""
    ctx = ray.get_runtime_context()
    task_id = ctx.get_task_id()

    split_block = cached_remote_fn(_split_block)

    for block in blocks_per_fragments:
        ref = split_block.remote(block, keys)
        queue.put((i, ref))
    queue.put(("end", i, task_id))
    return "success"
