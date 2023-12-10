import time
from typing import Any, Dict, List, Optional, Tuple

import ray
from ray.data._internal.execution.interfaces import RefBundle, TaskContext
from ray.data._internal.planner.exchange.interfaces import ExchangeTaskScheduler

# from ray.data._internal.planner.exchange.repartition_by_col_task_spec import (
#     RepartitionByColTaskSpec,
# )
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.split_by_key import Coordinator, process_fragment
from ray.data._internal.stats import StatsDict
from ray.util.queue import Queue


# The code structure follows SplitRepartitionTaskScheduler
# which uses a custom map function instead of the ExchangeTaskSpec
class SplitRepartitionByColTaskScheduler(ExchangeTaskScheduler):
    """
    The split (non-shuffle) repartition scheduler based on a column.

    Map-stage: split the blocks

    Reduce-stage: merge the blocks with the same key

    See `RepartitionByColTaskSpec` for the details of implementation.

    """

    def execute(
        self,
        refs: List[RefBundle],
        output_num_blocks: int,
        ctx: TaskContext,
        map_ray_remote_args: Optional[Dict[str, Any]] = None,
        reduce_ray_remote_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[RefBundle], StatsDict]:
        """
        Note:
            Unlike the standard non-shuffle repartition, we do not need
            the number of rows
        """
        input_owned_by_consumer = all(b.owns_blocks for b in refs)

        if map_ray_remote_args is None:
            map_ray_remote_args = {}
        if reduce_ray_remote_args is None:
            reduce_ray_remote_args = {}
        if "scheduling_strategy" not in reduce_ray_remote_args:
            reduce_ray_remote_args = reduce_ray_remote_args.copy()
            reduce_ray_remote_args["scheduling_strategy"] = "SPREAD"

        process_fragment_task = cached_remote_fn(process_fragment)
        reduce_task = cached_remote_fn(self._exchange_spec.reduce)

        split_map_metadata = None

        # # Each ref_bundle is coming from one ReadTask
        # # split_block_refs[i] = a list of subblocks split
        # # split_metadata = a flattened list of metadata
        # split_block_refs: List[List[ObjectRef[Block]]] = []
        # split_metadata: List[BlockMetadata] = []
        # for ref_bundle in refs:
        #     blocks_ref = [block for block, _ in ref_bundle.blocks]

        #     sub_block_refs, metadata_refs = split_blocks_task.options(
        #         **map_ray_remote_args,
        #         num_options=2,
        #     ).remote(blocks_ref, *self._exchange_spec._map_args)
        #     split_block_refs.extend(sub_block_refs)
        #     split_metadata.extend(metadata_refs)

        num_bundles = len(refs)

        queue = Queue()
        out_queue = Queue()  # for results and status
        coordinator = Coordinator.options(**reduce_ray_remote_args).remote(
            self._exchange_spec._reduce_args, queue, out_queue
        )

        t1 = time.perf_counter()
        # Perform map-task to all the fragments and submit the output to a queue.
        # Coordinator then reads the queue
        task_refs = []
        for bundle_id, ref_bundle in enumerate(refs):
            blocks_ref = [block for block, _ in ref_bundle.blocks]
            ref = process_fragment_task.options(
                **map_ray_remote_args,
            ).remote(bundle_id, blocks_ref, queue, *self._exchange_spec._map_args)
            task_refs.append(ref)

        print(f"Finished submitting map-tasks for {bundle_id+1} fragments")
        print(ray.get(task_refs))

        # Ensure all the fragments are taken care of
        while out_queue.qsize() < num_bundles:
            print("waiting the queue to finish...", out_queue.qsize())
            time.sleep(0.2)

        t2 = time.perf_counter()
        reduce_stage_time = t2 - t1
        print("number of remaining tasks: ", queue.qsize())

        message_counts = ray.get(coordinator.get_message_counts.remote())
        print("number of messages per fragment = ", message_counts)
        message_counts_after_end = sum(
            ray.get(coordinator.get_message_counts_after_end.remote()).values()
        )
        print("number of messages after end = ", message_counts_after_end)

        # coordinator.close.remote()

        reducers = ray.get(coordinator.get_reducers.remote())
        for i, reducer_ref in reducers:
            mapping = ray.get(reducer_ref.get_mapping().remote())
            print(i, mapping)

        reduce_block_refs = []
        reduce_metadata = []

        # # Setup progress bar for the map tasks
        # # and fetch the metadata
        # sub_progress_bar_dict = ctx.sub_progress_bar_dict
        # bar_name = (
        #     RepartitionByColTaskSpec.REPARTITION_BY_COLUMN_SPLIT_SUB_PROGRESS_BAR_NAME
        # )
        # assert bar_name in sub_progress_bar_dict, sub_progress_bar_dict
        # map_bar = sub_progress_bar_dict[bar_name]
        # split_map_metadata = map_bar.fetch_until_complete(map_bar)

        # # Submit and schedule reduce tasks
        # reduce_return = [
        #     reduce_task.options(**reduce_ray_remote_args, num_returns=2).remote(
        #         *self._exchange_spec._reduce_args,
        #         *split_block_refs[j],
        #     )
        #     for j in range(output_num_blocks)
        #     # Only process splits which contain blocks.
        #     if len(split_block_refs[j]) > 0
        # ]

        # # Setup progress bar for the reduce tasks
        # # and fetch the metadata
        # sub_progress_bar_dict = ctx.sub_progress_bar_dict
        # bar_name = (
        #     RepartitionByColTaskSpec.REPARTITION_BY_COLUMN_MERGE_SUB_PROGRESS_BAR_NAME
        # )
        # assert bar_name in sub_progress_bar_dict, sub_progress_bar_dict
        # reduce_bar = sub_progress_bar_dict[bar_name]

        # reduce_block_refs, reduce_metadata = zip(*reduce_return)
        # reduce_metadata = reduce_bar.fetch_until_complete(list(reduce_metadata))
        # reduce_block_refs, reduce_metadata = list(reduce_block_refs), list(
        #     reduce_metadata
        # )

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

        output = []
        for block, meta in zip(reduce_block_refs, reduce_metadata):
            output.append(
                RefBundle([(block, meta)], owns_blocks=input_owned_by_consumer)
            )
        stats = {
            "split": split_map_metadata,
            "reduce": reduce_metadata,
        }

        return (output, stats)
