from typing import List, Tuple, TypeVar, Union

import numpy as np

from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.planner.exchange.interfaces import ExchangeTaskSpec
from ray.data._internal.progress_bar import ProgressBar
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.sort import SortKey
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata
from ray.types import ObjectRef

from ray.data._internal.boundaries import get_group_boundaries

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
        else:
            out: List[Block] = [
                accessor.slice(start, end)
                for start, end in zip(split_indices[:-1], split_indices[1:])
            ]

        meta = accessor.get_metadata(
            input_files=None, exec_stats=stats.build()
        )
        yield from out
        yield meta

    @staticmethod
    def reduce(
        keys: Union[str, List[str]],
        *mapper_outputs: List[Block],
        partial_reduce: bool = False,
    ) -> Tuple[Block, BlockMetadata]:
        """Do nothing for now"""
        stats = BlockExecStats.builder()
        accessor = BlockAccessor.for_block(mapper_outputs[0])

        meta = accessor.get_metadata(
            input_files=None, exec_stats=stats.build()
        )
        yield from mapper_outputs
        yield meta


