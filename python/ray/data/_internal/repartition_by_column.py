import asyncio
import itertools
import time
from collections import deque
from math import ceil
from typing import Any, Deque, Iterator, List, Tuple, Union

import numpy as np
import pyarrow as pa

import ray
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data.block import BlockAccessor, BlockExecStats, BlockMetadata
from ray.util.queue import Queue

logger = DatasetLogger(__name__)

# TODO: use TableBlockBuilder etc to handle merging of pyarrow table, as
#       as it can handle extension types


def batched(blocks: List[Any], batch_size: int) -> Iterator[List[Any]]:
    """Iterates over the blocks and yields batches of objects.

    Note:
        This function can be replaced by itertools.batched when Python 3.12.
    """
    for i in range(0, len(blocks), batch_size):
        yield blocks[i : i + batch_size]


@ray.remote
def get_blocks_ref(blocks: List[List[int]]) -> List[ray.ObjectRef]:
    for block in blocks:
        yield block


@ray.remote
class Splitter:
    def __init__(self, idx):
        self.idx = idx
        self.name = ray.get_runtime_context().get_actor_name()
        self.item_count = 0

    def split_list(
        self, item: List[int]
    ) -> Iterator[Union[List[int], List[List[int]]]]:
        group_names = []
        for group_name, group in itertools.groupby(item, lambda x: x):
            group_names.append(group_name)
            g = list(group)
            yield g
        logger.get_logger().info(f"{self.name}: {group_names=}")
        yield group_names

    def split_pyarrow_table(
        self, item: pa.Table, key_column_name: str
    ) -> Iterator[Union[pa.Table, List[int]]]:
        """Split a single table into multiple parts. Each part has the
        same group key.
        """
        arr = item.column(key_column_name).to_numpy()

        # Find the indices where the key changes.
        indices = np.hstack([[0], np.where(np.diff(arr) != 0)[0] + 1, [len(arr)]])

        self.item_count += 1

        # If there's only one group, return it as the left part.
        if len(indices) == 1:
            yield item
            yield [arr[0]]

        group_names = []
        for start, end in zip(indices[:-1], indices[1:]):
            group_names.append(arr[start])
            yield item.slice(start, end)
        yield group_names


@ray.remote
class Merger:
    def __init__(self, idx):
        self.idx = idx
        self.name = f"Merger-({self.idx})"

    def merge_list(self, group_keys: List[int], blks):
        blks = ray.get(list(blks))
        logger.get_logger().info(f"{self.name}: {group_keys=}, {blks=}")

        for key, block_iterator in itertools.groupby(
            zip(group_keys, blks), lambda x: x[0]
        ):
            block = [b for _, b in block_iterator]
            logger.get_logger().info(key, block)
            yield list(np.concatenate(block))

    @ray.method(num_returns=3)
    def merge_tables(self, keys_and_blocks: List[Tuple]):
        """Merge pyarrow tables

        Neighboring tables with the same group key are concatenated. Similar to
        `itertools.groupby`, this operation is local and does not give the same
        result as a `groupby` which collects same key globally.

        Yields:
            This function yields a list of merged tables. The last element of
            the output is a list of metadata for each block.
        """
        # materialize the blocks because of the concatenation
        keys = [k for k, _ in keys_and_blocks]
        blks = [b for _, b in keys_and_blocks]
        blks = ray.get(blks)

        blks_len = [len(b) for b in blks]
        logger.get_logger().info(
            f"{self.name}: {len(keys)=}, {min(keys)=}, {max(keys)=}, {min(blks_len)=}, {max(blks_len)=}"
        )

        all_blocks, all_metadata, all_keys = [], [], []
        for key, block_iterator in itertools.groupby(zip(keys, blks), lambda x: x[0]):
            stats = BlockExecStats.builder()
            blocks = [b for _, b in block_iterator]
            all_keys.append(key)

            block = pa.concat_tables(blocks)

            meta = BlockAccessor.for_block(block).get_metadata(
                input_files=None,
                exec_stats=stats.build(),
            )
            all_blocks.append(block)
            all_metadata.append(meta)
            all_keys.append(key)

        return all_blocks, all_metadata, all_keys


@ray.remote(num_cpus=0)
class Actor:
    def __init__(self, idx, keys: str, world_size: int):
        self.idx = idx
        self.keys = keys
        self.world_size = world_size
        self.name = f"Actor-({self.idx})"

        self.splitted_blocks: Deque[Tuple[int, ray.ObjectRef]] = deque()

        # For exchange boundary
        self.is_left_most = self.idx == 0
        self.is_right_most = self.idx == self.world_size - 1
        self.left = None if self.is_left_most else asyncio.Queue(1)
        self.right = None if self.is_right_most else asyncio.Queue(1)  # local only
        self.next_left = None if self.is_right_most else asyncio.Queue(1)

        # Indicate it's ready to consume
        self.consume_ready = asyncio.Event()

        # For output
        # self.output_queue = asyncio.Queue()
        self.output_queue = Queue()

        self.splitter = Splitter.remote(self.idx)
        self.merger = Merger.remote(self.idx)

    async def split(self, block_refs: List[ray.ObjectRef]) -> List[ray.ObjectRef]:
        """Split a list of blocks based on the group key.

        This is the map task that generates multiple sub-blocks for each input block,
        depending on the group key.
        """

        for ref in block_refs:
            splitted = []
            async for item in self.splitter.split_pyarrow_table.remote(ref, self.keys):
                splitted.append(item)

            # materialize the keys but keep the block refs
            splitted_groups = await splitted.pop()

            self.splitted_blocks.extend(zip(splitted_groups, splitted))

        if not self.is_left_most:
            key, blk = self.splitted_blocks.popleft()
            self.left.put_nowait((key, blk))

        if not self.is_right_most:
            key, blk = self.splitted_blocks.pop()
            self.right.put_nowait((key, blk))

        # TODO: this flag may be removed when it is converted into
        # a fully streaming operation
        self.consume_ready.set()
        logger.get_logger().info(f"{self.name}-split: done.")

    async def put_next_left(self, item):
        """Put the next left item to the next_left queue.

        This function is to be called by the sender.
        """

        logger.get_logger().info(
            f"{self.name}-put-next-left: received an item with key {item[0]}"
        )
        await self.next_left.put(item)

    async def send_to_left(self, left_actor):
        """Send the left item to the left actor."""
        logger.get_logger().info(f"{self.name}-send-to-left: waiting left to be ready")
        item = await self.left.get()
        logger.get_logger().info(
            f"{self.name}-send-to-left: sent an item with key {item[0]}"
        )
        await left_actor.put_next_left.remote(item)

    async def handle_right(self):
        """Handle the right edge by putting the last two items back to
        the merging queue.

        This function is to be called by each actor except the right most one.
        """

        logger.get_logger().info(f"{self.name}-consume: waiting self.right to be ready")
        right_item = await self.right.get()
        self.right.task_done()
        self.splitted_blocks.append(right_item)

        logger.get_logger().info(f"{self.name}-consume: waiting for next left")
        next_left = await self.next_left.get()
        self.next_left.task_done()
        logger.get_logger().info(f"{self.name}-consume: next_left's key={next_left[0]}")

        self.splitted_blocks.append(next_left)

    async def merge(self):
        if not self.is_right_most:
            await self.handle_right()
        else:
            logger.get_logger().info(
                f"{self.name}-consume: right most. skipping boundary handling"
            )

        logger.get_logger().info(
            f"{self.name}-consume: waiting for the signal to consume"
        )
        await self.consume_ready.wait()
        logger.get_logger().info(f"{self.name}-consume: {len(self.splitted_blocks)=}")

        all_blocks, all_metadata, all_keys = self.merger.merge_tables.remote(
            self.splitted_blocks
        )

        time_start = time.perf_counter()
        all_blocks = await all_blocks
        all_metadata = await all_metadata
        all_keys = await all_keys
        time_end = time.perf_counter()
        logger.get_logger().info(
            f"{self.name}-consume: {(time_end - time_start)=:.3f}s"
        )

        for block, meta, key in zip(all_blocks, all_metadata, all_keys):
            await self.output_queue.put_async((block, meta, key))
        await self.output_queue.put_async("done")

    def get_queue(self):
        return self.output_queue

    # # async def consume(self):
    # #     output = []
    # #     while True:
    # #         # item = await self.output_queue.get()
    # #         item = await self.output_queue.get_async()
    # #         # self.output_queue.task_done()
    # #         if item == "done":
    # #             logger.get_logger().info(
    # #                 f"{self.name}-consume: finished with {len(output)} items."
    # #             )
    # #             # return output
    # #             keys = output.pop()
    # #             metadata = output.pop()
    # #             return list(zip(output, metadata, keys))

    # #         output.append(item)

    # def consume_iterator(self):
    #     while True:
    #         item = self.output_queue.get()
    #         # self.output_queue.task_done()
    #         if item == "done":
    #             logger.get_logger().info(
    #                 f"{self.name}-consume: finished with {len(output)} items."
    #             )
    #             # return output
    #             keys = output.pop()
    #             metadata = output.pop()
    #             return list(zip(output, metadata, keys))


async def repartition_by_column(
    idx,
    blocks: List[ray.ObjectRef],
    keys,
    num_actors: int,
    use_batching: bool,
    actors,
) -> List[List[ray.ObjectRef]]:
    if len(blocks) <= num_actors:
        num_actors = 1

    num_blocks_per_actor = ceil(len(blocks) / num_actors)
    logger.get_logger().info(
        f"{idx}: {len(blocks)=}, {num_actors=}, {num_blocks_per_actor=}"
    )

    # actors = [
    #     Actor.options(name=f"Actor-({idx, i})").remote(i, keys, num_actors)
    #     for i in range(num_actors)
    # ]

    split_tasks = [
        actors[i].split.remote(batch_per_actor)
        for i, batch_per_actor in enumerate(batched(blocks, num_blocks_per_actor))
    ]
    boundary_tasks = [
        actor.send_to_left.remote(left_actor)
        for actor, left_actor in zip(actors[1:], actors[:-1])
    ]
    merge_tasks = [actor.merge.remote() for actor in actors]
    consume_tasks = [actor.get_queue.remote() for actor in actors]

    await asyncio.gather(*split_tasks)
    await asyncio.gather(*boundary_tasks)
    await asyncio.gather(*merge_tasks)

    # returns `num_actors` of batches (list of tuples)
    return consume_tasks


@ray.remote(num_returns="streaming")
def repartition_runner(
    ref_id,
    blocks,
    map_args,
    actors,
):
    queue_refs = asyncio.run(
        repartition_by_column(
            ref_id,
            blocks,
            *map_args,
            actors,
        )
    )

    all_metadata, all_keys = [], []
    while queue_refs:
        [ready], queue_refs = ray.wait(queue_refs)
        queue: Queue = ray.get(ready)

        logger.get_logger().info(f"queue has {queue.size()} elements")

        metadata, keys = [], []
        # while not queue.empty():
        #     item = queue.get()
        #     if item == "done":
        #         break
        #     else:
        #         block, meta, key = item
        #         metadata.append(meta)
        #         keys.append(key)
        #         yield block

        items = queue.get_nowait_batch(queue.size())
        items.pop()

        for i, (block, meta, key) in enumerate(items):
            if i == 0:
                print(f"{type(block)=}, {type(meta)=}, {type(key)=}")
            yield block
            metadata.append(meta)
            keys.append(key)

        all_metadata.append(metadata)
        all_keys.append(keys)

    logger.get_logger().info(f"{len(all_metadata)} metadata")
    logger.get_logger().info(f"{len(all_keys)} keys")

    for metadata in all_metadata:
        yield metadata
    for keys in all_keys:
        yield keys


def repartition_by_column_stage_impl(blocks, keys, ctx):
    raise NotImplementedError
