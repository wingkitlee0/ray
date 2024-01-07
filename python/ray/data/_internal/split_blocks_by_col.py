from typing import List, Optional, Tuple, Union

from ray.data._internal.block_list import BlockList
from ray.data._internal.boundaries import get_group_boundaries
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import TaskContext
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
    block_accessor = BlockAccessor.for_block(_block)
    arr = block_accessor.to_numpy(keys)

    if len(arr) == 0:
        return None

    _, split_indices = get_group_boundaries(arr)

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


def repartition_by_column(
    blocks: BlockList,
    keys: Union[str, List[str]],
    ctx: Optional[TaskContext] = None,
) -> Tuple[BlockList, dict]:
    from ray.data._internal.block_batching.iter_batches import iter_batches
    from ray.data._internal.execution.legacy_compat import _block_list_to_bundles

    ref_bundles = _block_list_to_bundles(blocks, blocks._owned_by_consumer)
    owned_by_consumer = blocks._owned_by_consumer

    new_blocks = []
    new_metadata = []
    for ref_bundle in ref_bundles:
        for data_batch in iter_batches(
            block_refs=ref_bundle.blocks,
            ensure_copy=True,
        ):
            block_list, block_metadata_list = _split_block(data_batch)
            new_blocks.extend(block_list)
            new_metadata.extend(block_metadata_list)

    return BlockList(new_blocks, new_metadata, owned_by_consumer=owned_by_consumer), {}
