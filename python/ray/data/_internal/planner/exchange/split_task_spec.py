from typing import Any, List, Tuple, TypeVar, Union

from ray.data._internal.boundaries import get_group_boundaries
from ray.data._internal.planner.exchange.interfaces import ExchangeTaskSpec
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata

T = TypeVar("T")


class SplitTaskSpec(ExchangeTaskSpec):
    """Example ExchangeTaskSpec"""

    SPLIT_SUB_PROGRESS_BAR_NAME = "Split blocks by column"
    MERGE_SUB_PROGRESS_BAR_NAME = "Merge blocks by column"

    def __init__(
        self,
        keys: Union[str, List[str]],
    ):
        super().__init__(
            map_args=[keys],
            reduce_args=[keys],
        )

    @staticmethod
    def map(
        idx: int,
        block: Block,
        output_num_blocks: int,
        keys: Union[str, List[str]],
    ) -> List[Union[BlockMetadata, Block]]:
        """This is a single ray task

        Args:
            output_num_blocks: not used as it's determined by the actual split
        """
        stats = BlockExecStats.builder()
        accessor = BlockAccessor.for_block(block)

        arr = accessor.to_numpy(keys)

        _, split_indices = get_group_boundaries(arr)

        if len(split_indices) == 2:
            out = [block]
            out_keys = [get_key_from_block(block, keys)]
        else:
            out: List[Block] = []
            out_keys = []
            for start, end in zip(split_indices[:-1], split_indices[1:]):
                _block = accessor.slice(start, end)
                _keys = get_key_from_block(_block, keys)
                out.append(_block)
                out_keys.append(_keys)

        meta = accessor.get_metadata(input_files=None, exec_stats=stats.build())
        yield from out
        yield meta
        yield out_keys

    @staticmethod
    def reduce_idenity(
        keys: Union[str, List[str]],
        *mapper_outputs: List[Block],
        partial_reduce: bool = False,
    ) -> Tuple[Block, BlockMetadata]:
        """Do nothing for now"""
        stats = BlockExecStats.builder()
        accessor = BlockAccessor.for_block(mapper_outputs[0])

        meta = accessor.get_metadata(input_files=None, exec_stats=stats.build())
        yield from mapper_outputs
        yield meta

    @staticmethod
    def reduce(
        keys: Union[str, List[str]],
        *mapper_outputs: List[Block],
        partial_reduce: bool = False,
    ) -> Tuple[Block, BlockMetadata]:
        import pyarrow as pa

        stats = BlockExecStats.builder()
        blocks = [BlockAccessor.for_block(block).to_arrow() for block in mapper_outputs]
        out_block = pa.concat_tables(blocks)
        meta = BlockAccessor.for_block(out_block).get_metadata(
            input_files=None,
            exec_stats=stats.build(),
        )
        return out_block, meta


def get_key_from_block(block: Block, keys: Union[str, List[str]]) -> Tuple[Any]:
    accessor = BlockAccessor.for_block(block)
    arr = accessor.to_numpy(keys)

    if isinstance(arr, dict):
        return tuple([v[0] for v in arr.values()])
    else:
        return (arr[0],)
