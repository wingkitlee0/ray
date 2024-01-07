import asyncio
import functools
import itertools
from collections import deque
from math import ceil
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Tuple, TypeVar

import numpy as np
import pyarrow as pa

import ray
from ray.data._internal.execution.interfaces import RefBundle, TaskContext
from ray.data._internal.planner.exchange.interfaces import ExchangeTaskScheduler
from ray.data._internal.planner.exchange.repartition_task_spec import (
    RepartitionTaskSpec,
)
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata
from ray.types import ObjectRef

KeyType = TypeVar("KeyType")


# TODO: support string key type
# TODO: support multiple keys
def split_pyarrow_table(item: pa.Table, key="g") -> Deque[Tuple[KeyType, pa.Table]]:
    arr = item.column(key).to_numpy()
    indices = np.hstack([[0], np.where(np.diff(arr) != 0)[0] + 1, [len(arr)]])
    lengths = np.diff(indices)

    if len(indices) == 1:
        return deque([(arr[0], item)])

    return deque(
        [
            (arr[start], item.slice(start, length))
            for start, length in zip(indices[:-1], lengths)
        ]
    )


def split_blocks(
    blocks: List[List[int]], split_fn: Callable[[List[int]], Iterator[List[int]]]
) -> Iterator[List[int]]:
    """Split each block using the split_fn"""
    for block in blocks:
        yield from split_fn(ray.get(block))


def merge_lists(items: Iterator[List[int]]) -> List[int]:
    return functools.reduce(lambda a, b: a + b, items)


def merge_tables(items: pa.Table) -> pa.Table:
    return pa.concat_tables([item[1] for item in items])


@ray.remote
class Coordinator:
    def __init__(self, idx, queue: asyncio.Queue, split_fn, merge_fn, key_fn):
        self.idx = idx
        self.queue = queue
        self.split_fn = split_fn
        self.merge_fn = merge_fn
        self.key_fn = key_fn
        self.boundary_event = asyncio.Event()
        self.output = []

    async def split_task(self, blocks, prev_coordinator: "Coordinator" = None):
        block_meta = []
        stats = BlockExecStats.builder()
        for block in blocks:
            for subblock in self.split_fn(ray.get(block)):
                meta = BlockAccessor.for_block(subblock).get_metadata(
                    input_files=None,
                    exec_stats=stats.build(),
                )
                await self.queue.put(block)
                block_meta.append(meta)
                stats = BlockExecStats.builder()
        else:
            stats.build()

        self.boundary_event.set()
        print("done")

        if prev_coordinator is not None:
            await prev_coordinator.wait_for_boundary_event.remote()
            print("do something")
            item = await self.queue.get()
            self.queue.task_done()
            await prev_coordinator.put.remote(item)
            prev_id = await prev_coordinator.queue_id.remote()
            print(
                f"item sent to prev coordinator-{prev_id}: {item[0]}",
            )
            await prev_coordinator.put.remote("done")

        return block_meta

    async def wait_for_boundary_event(self):
        await self.boundary_event.wait()
        print("yay!")

    async def put(self, item):
        await self.queue.put(item)

    def qsize(self):
        return self.queue.qsize()

    async def queue_id(self):
        return self.idx

    async def consume(self):
        print("consuming queue", self.idx)
        while True:
            try:
                item = await self.queue.get()
                self.queue.task_done()
                if item == "done":  # Sentinel value to indicate done
                    return
                self.output.append(item)
            except asyncio.CancelledError:
                return

    @ray.method(num_returns="streaming")
    def merge_iterator(self):
        """Reduce the neigbouring blocks with same key into a single block.

        It uses `itertools.groupby` which only groups consecutive elements with same key.
        This is different from a standard group-by in pandas or SQL.

        Returns:
            Blocks grouped by key. The last item is a list of keys.
        """
        print(f"merging for queue-{self.idx}")
        keys = []
        metadata = []
        for key, g in itertools.groupby(self.output, key=self.key_fn):
            stats = BlockExecStats.builder()
            block = self.merge_fn(list(g))
            keys.append(key)
            meta = BlockAccessor.for_block(block).get_metadata(
                input_files=None,
                exec_stats=stats.build(),
            )
            metadata.append(meta)
            yield block
        yield keys
        yield metadata


async def wrapper(queue_actor):
    return queue_actor.consume.remote()


@ray.remote
class Merger:
    def __init__(self, num_queues: int):
        self.num_queues = num_queues

    @ray.method(num_returns=2)
    async def execute(self, blocks, split_fn, merge_fn, key_fn):
        batch_size = ceil(len(blocks) / self.num_queues)
        print(f"{batch_size=}")

        splitted_queues = [asyncio.Queue() for _ in range(self.num_queues)]
        coordinators = [
            Coordinator.remote(i, q, split_fn, merge_fn, key_fn)
            for i, q in enumerate(splitted_queues)
        ]
        consumer_refs = [wrapper(actor) for actor in coordinators]
        # consumer_refs = [coordinator.consume.remote() for coordinator in coordinators]

        refs = []
        for i, offset in enumerate(range(0, len(blocks), batch_size)):
            current_slice = blocks[offset : offset + batch_size]
            prev_coordinator = coordinators[i - 1] if i > 0 else None

            ref = coordinators[i].split_task.remote(current_slice, prev_coordinator)
            refs.append(ref)

        ray.get(refs)

        for coordinator in coordinators:
            print(ray.get(coordinator.qsize.remote()))
            ray.get(coordinator.wait_for_boundary_event.remote())

        # Wait for all consumers to finish
        await asyncio.gather(*consumer_refs)

        for q in splitted_queues:
            await q.join()

        output, keys, metadata = [], [], []
        for coordinator in coordinators:
            results = list(coordinator.merge_iterator.remote())
            metadata.append(results.pop())
            keys.append(results.pop())
            output.extend(results)
        return output, keys, metadata


class RepartitionByColumnTaskScheduler(ExchangeTaskScheduler):
    """Split-by-column experiment"""

    def execute(
        self,
        refs: List[RefBundle],
        output_num_blocks: int,
        ctx: TaskContext,
        map_ray_remote_args: Optional[Dict[str, Any]] = None,
        reduce_ray_remote_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[RefBundle], StatsDict]:
        """
        Args:
            output_num_blocks: not used as it's determined by the actual split
        """

        input_owned_by_consumer = all(rb.owns_blocks for rb in refs)

        if map_ray_remote_args is None:
            map_ray_remote_args = {}
        if reduce_ray_remote_args is None:
            reduce_ray_remote_args = {}
        if "scheduling_strategy" not in reduce_ray_remote_args:
            reduce_ray_remote_args = reduce_ray_remote_args.copy()
            reduce_ray_remote_args["scheduling_strategy"] = "SPREAD"

        output_refs, key_refs, metadata_refs = [], [], []
        for i, ref_bundle in enumerate(refs):
            blocks = [b for b, _ in ref_bundle.blocks]

            merger = Merger.remote(3)
            output_blocks, keys, metadata = merger.execute.remote(
                blocks,
                split_fn=split_pyarrow_table,
                merge_fn=merge_tables,
                key_fn=lambda x: x[0],
            )
            output_refs.append(output_blocks)
            key_refs.extend(keys)
            metadata_refs.extend(metadata)

        # split_task = cached_remote_fn(self._exchange_spec.map)
        # reduce_task = cached_remote_fn(self._exchange_spec.reduce)

        # list of (ref_id, block_id, block_ref, meta)
        # blocks_with_metadata: List[Tuple[ObjectRef[Block], BlockMetadata]] = []
        # for i, ref_bundle in enumerate(refs):
        #     for j, (b, m) in enumerate(ref_bundle.blocks):
        #         blocks_with_metadata.append((i, j, b, m))

        # sub_progress_bar_dict = ctx.sub_progress_bar_dict
        # bar_name = SplitTaskSpec.SPLIT_SUB_PROGRESS_BAR_NAME
        # assert bar_name in sub_progress_bar_dict, sub_progress_bar_dict
        # map_bar = sub_progress_bar_dict[bar_name]

        # split_map_out: List[List] = [
        #     list(
        #         split_task.options(
        #             **map_ray_remote_args,
        #             num_returns="streaming",
        #         ).remote(i, block, -1, *self._exchange_spec._map_args)
        #     )
        #     for i, (j, k, block, _) in enumerate(blocks_with_metadata)
        # ]

        # # split_metadata = list of metadata
        # split_block_refs: List[List[ObjectRef]] = []  # list of list
        # split_metadata = []  # list of metadata
        # split_keys = []  # list of refs (of list)
        # output_num_blocks = 0
        # for i, refs in enumerate(split_map_out):
        #     output_num_blocks += len(refs) - 2
        #     split_block_refs.append(refs[:-2])
        #     split_metadata.append(refs[-2])
        #     split_keys.append(refs[-1])

        # split_metadata = map_bar.fetch_until_complete(split_metadata)
        # split_keys: List[List[Tuple]] = map_bar.fetch_until_complete(split_keys)
        # # print(split_keys)
        # print("output_num_blocks = ", output_num_blocks)
        # print("len = ", sum([len(s) for s in split_keys]))

        sub_progress_bar_dict = ctx.sub_progress_bar_dict
        bar_name = SplitTaskSpec.MERGE_SUB_PROGRESS_BAR_NAME
        assert bar_name in sub_progress_bar_dict, sub_progress_bar_dict
        reduce_bar = sub_progress_bar_dict[bar_name]

        output_metadata = reduce_bar.fetch_until_complete(list(metadata_refs))

        # # split_block_refs is a list of list
        # reduce_block_refs = []  # flattened list
        # reduce_metadata = []
        # for blocks_to_reduce in split_block_refs:
        #     for block_to_reduce in blocks_to_reduce:
        #         refs = list(
        #             reduce_task
        #             .options(**reduce_ray_remote_args, num_returns="streaming")
        #             .remote(*self._exchange_spec._reduce_args, block_to_reduce, partial_reduce=False)
        #         )
        #         reduce_metadata.append(refs[-1])
        #         reduce_block_refs.extend(refs[:-1])

        # reduce_metadata = reduce_bar.fetch_until_complete(list(reduce_metadata))

        # mapping = {}
        # for merge_blocks, merge_keys in zip(split_block_refs, split_keys):
        #     for block, group in zip(merge_blocks, merge_keys):
        #         if group in mapping:
        #             mapping[group].append(block)
        #         else:
        #             mapping[group] = [block]

        # reduce_block_refs = []  # flattened list
        # reduce_metadata = []
        # for blocks in mapping.values():
        #     ref, meta_ref = reduce_task.options(
        #         **reduce_ray_remote_args, num_returns=2
        #     ).remote(*self._exchange_spec._reduce_args, *blocks)
        #     reduce_block_refs.append(ref)
        #     reduce_metadata.append(meta_ref)

        # reduce_metadata = reduce_bar.fetch_until_complete(list(reduce_metadata))

        # # Handle empty blocks.
        # if len(reduce_block_refs) < output_num_blocks:
        #     import pyarrow as pa

        #     from ray.data._internal.arrow_block import ArrowBlockBuilder
        #     from ray.data._internal.pandas_block import (
        #         PandasBlockBuilder,
        #         PandasBlockSchema,
        #     )

        #     num_empty_blocks = output_num_blocks - len(reduce_block_refs)
        #     first_block_schema = reduce_metadata[0].schema
        #     if first_block_schema is None:
        #         raise ValueError(
        #             "Cannot split partition on blocks with unknown block format."
        #         )
        #     elif isinstance(first_block_schema, pa.Schema):
        #         builder = ArrowBlockBuilder()
        #     elif isinstance(first_block_schema, PandasBlockSchema):
        #         builder = PandasBlockBuilder()
        #     empty_block = builder.build()
        #     empty_meta = BlockAccessor.for_block(empty_block).get_metadata(
        #         input_files=None, exec_stats=None
        #     )  # No stats for empty block.
        #     empty_block_refs, empty_metadata = zip(
        #         *[(ray.put(empty_block), empty_meta) for _ in range(num_empty_blocks)]
        #     )
        #     reduce_block_refs.extend(empty_block_refs)
        #     reduce_metadata.extend(empty_metadata)

        output_blocks = []
        for block, meta in zip(output_refs, output_metadata):
            output_blocks.append(
                RefBundle([(block, meta)], owns_blocks=input_owned_by_consumer)
            )
        stats = {
            "split": output_metadata,
        }

        return (output_blocks, stats)
