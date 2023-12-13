from typing import Any, Dict, List, Optional, Tuple


import ray
from ray.data._internal.execution.interfaces import RefBundle, TaskContext
from ray.data._internal.planner.exchange.interfaces import ExchangeTaskScheduler

from ray.data._internal.remote_fn import cached_remote_fn

from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockAccessor, BlockMetadata
from ray.types import ObjectRef

from ray.data._internal.planner.exchange.split_task_spec import SplitTaskSpec


class SplitTaskScheduler(ExchangeTaskScheduler):
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

        input_num_rows = 0
        input_owned_by_consumer = True
        for ref_bundle in refs:
            block_num_rows = ref_bundle.num_rows()
            if block_num_rows is None:
                raise ValueError(
                    "Cannot split partition on blocks with unknown number of rows."
                )
            input_num_rows += block_num_rows
            if not ref_bundle.owns_blocks:
                input_owned_by_consumer = False

        if map_ray_remote_args is None:
            map_ray_remote_args = {}
        if reduce_ray_remote_args is None:
            reduce_ray_remote_args = {}
        if "scheduling_strategy" not in reduce_ray_remote_args:
            reduce_ray_remote_args = reduce_ray_remote_args.copy()
            reduce_ray_remote_args["scheduling_strategy"] = "SPREAD"

        split_task = cached_remote_fn(self._exchange_spec.map)
        reduce_task = cached_remote_fn(self._exchange_spec.reduce)

        blocks_with_metadata: List[Tuple[ObjectRef[Block], BlockMetadata]] = []
        for ref_bundle in refs:
            blocks_with_metadata.extend(ref_bundle.blocks)


        sub_progress_bar_dict = ctx.sub_progress_bar_dict
        bar_name = SplitTaskSpec.SPLIT_SUB_PROGRESS_BAR_NAME
        assert bar_name in sub_progress_bar_dict, sub_progress_bar_dict
        map_bar = sub_progress_bar_dict[bar_name]

        split_map_out: List[List] = [
            list(split_task.options(
                **map_ray_remote_args,
                num_returns="streaming",
            ).remote(i, block, -1, *self._exchange_spec._map_args))
            for i, (block, _) in enumerate(blocks_with_metadata)
        ]

        # split_metadata = list of metadata
        split_block_refs = []  # list of list
        split_metadata = []   # list of metadata
        output_num_blocks = 0
        for i, refs in enumerate(split_map_out):
            output_num_blocks += len(refs) - 1
            split_metadata.append(refs[-1])
            split_block_refs.append(refs[:-1])


        split_metadata = map_bar.fetch_until_complete(split_metadata)

        sub_progress_bar_dict = ctx.sub_progress_bar_dict
        bar_name = SplitTaskSpec.MERGE_SUB_PROGRESS_BAR_NAME
        assert bar_name in sub_progress_bar_dict, sub_progress_bar_dict
        reduce_bar = sub_progress_bar_dict[bar_name]

        # split_block_refs is a list of list
        reduce_block_refs = []  # flattened list
        reduce_metadata = []
        for blocks_to_reduce in split_block_refs:
            for block_to_reduce in blocks_to_reduce:
                refs = list(
                    reduce_task
                    .options(**reduce_ray_remote_args, num_returns="streaming")
                    .remote(*self._exchange_spec._reduce_args, block_to_reduce, partial_reduce=False)
                )
                reduce_metadata.append(refs[-1])
                reduce_block_refs.extend(refs[:-1])

        reduce_metadata = reduce_bar.fetch_until_complete(list(reduce_metadata))


        # Handle empty blocks.
        if len(reduce_block_refs) < output_num_blocks:
            import pyarrow as pa

            from ray.data._internal.arrow_block import ArrowBlockBuilder
            from ray.data._internal.pandas_block import (
                PandasBlockBuilder,
                PandasBlockSchema,
            )

            num_empty_blocks = output_num_blocks - len(reduce_block_refs)
            first_block_schema = reduce_metadata[0].schema
            if first_block_schema is None:
                raise ValueError(
                    "Cannot split partition on blocks with unknown block format."
                )
            elif isinstance(first_block_schema, pa.Schema):
                builder = ArrowBlockBuilder()
            elif isinstance(first_block_schema, PandasBlockSchema):
                builder = PandasBlockBuilder()
            empty_block = builder.build()
            empty_meta = BlockAccessor.for_block(empty_block).get_metadata(
                input_files=None, exec_stats=None
            )  # No stats for empty block.
            empty_block_refs, empty_metadata = zip(
                *[(ray.put(empty_block), empty_meta) for _ in range(num_empty_blocks)]
            )
            reduce_block_refs.extend(empty_block_refs)
            reduce_metadata.extend(empty_metadata)

        output = []
        for block, meta in zip(reduce_block_refs, reduce_metadata):
            output.append(
                RefBundle([(block, meta)], owns_blocks=input_owned_by_consumer)
            )
        stats = {
            "split": split_metadata,
            "reduce": reduce_metadata,
        }

        return (output, stats)
