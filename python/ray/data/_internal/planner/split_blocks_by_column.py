from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import ray
from ray.data._internal.boundaries import get_group_boundaries
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import (
    AllToAllTransformFn,
    RefBundle,
    TaskContext,
)
from ray.data._internal.planner.exchange.split_task_scheduler import SplitTaskScheduler
from ray.data._internal.planner.exchange.split_task_spec import SplitTaskSpec
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata

logger = DatasetLogger(__name__)


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
    block_accessor = BlockAccessor.for_block(ray.get(_block))
    arr = block_accessor.to_numpy(keys)

    if len(arr) == 0:
        return None

    _, split_indices = get_group_boundaries(arr)

    if len(split_indices) == 1:
        return [block_accessor.to_block()]

    def get_out_block(accessor, start, end):
        return accessor.slice(start, end)

    slice_task = cached_remote_fn(get_out_block)

    out_blocks = []
    out_metadata = []
    for start, end in zip(split_indices[:-1], split_indices[1:]):
        out_block = slice_task.remote(block_accessor, start, end)
        # out_block = block_accessor.slice(start, end)
        # accessor = BlockAccessor.for_block(out_block)
        meta = block_accessor.get_metadata(input_files=None, exec_stats=stats.build())
        out_blocks.append(out_block)
        out_metadata.append(meta)

    return out_blocks, out_metadata


def generate_split_blocks_by_column_fn(
    keys: Union[str, List[str]]
) -> AllToAllTransformFn:
    """Generate function to randomize order of blocks."""

    def fn(
        refs: List[RefBundle], context: TaskContext, keys: Union[str, List[str]]
    ) -> Tuple[List[RefBundle], StatsDict]:
        blocks_with_metadata = []
        for ref_bundle in refs:
            for block, meta in ref_bundle.blocks:
                blocks_with_metadata.append((block, meta))

        if len(blocks_with_metadata) == 0:
            return refs, {}

        input_owned = all(b.owns_blocks for b in refs)

        output = []
        meta_list = []
        for block, meta in blocks_with_metadata:
            subblocks, subblocks_meta = _split_block(block, keys)

            for b, m in zip(subblocks, subblocks_meta):
                meta_list.append(m)
                output.append(RefBundle([(b, meta)], owns_blocks=input_owned))

            return output, {"split": meta_list}

    return partial(fn, keys=keys)


def generate_split_blocks_by_column_fn2(
    keys: Union[str, List[str]], ray_remote_args: Optional[Dict[str, Any]]
) -> AllToAllTransformFn:
    """Generate function to split blocks by the specified key column"""

    def fn(
        refs: List[RefBundle],
        ctx: TaskContext,
        keys: Union[str, List[str]],
        ray_remote_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[RefBundle], StatsDict]:
        blocks = []
        metadata = []
        for ref_bundle in refs:
            for block, block_metadata in ref_bundle.blocks:
                blocks.append(block)
                metadata.append(block_metadata)
        if len(blocks) == 0:
            return (blocks, {})

        split_spec = SplitTaskSpec(keys=keys)
        scheduler = SplitTaskScheduler(split_spec)

        return scheduler.execute(
            refs=refs,
            output_num_blocks=-1,
            ctx=ctx,
            map_ray_remote_args=ray_remote_args,
            reduce_ray_remote_args=ray_remote_args,
        )

    # NOTE: use partial function to pass parameters to avoid error like
    # "UnboundLocalError: local variable ... referenced before assignment",
    # because `key` and `descending` variables are reassigned in `fn()`.
    return partial(fn, keys=keys, ray_remote_args=ray_remote_args)
