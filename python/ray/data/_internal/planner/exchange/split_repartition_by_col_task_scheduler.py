import time
from typing import Any, Dict, List, Optional, Tuple

import ray
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import RefBundle, TaskContext
from ray.data._internal.planner.exchange.interfaces import ExchangeTaskScheduler
from ray.data._internal.planner.exchange.repartition_by_col_task_spec import (
    RepartitionByColTaskSpec,
)
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.split_by_key import Coordinator, process_fragment
from ray.data._internal.stats import StatsDict
from ray.data.block import BlockMetadata
from ray.util.queue import Queue

logger = DatasetLogger(__name__)


# The code structure follows SplitRepartitionTaskScheduler
# which uses a custom map function instead of the ExchangeTaskSpec
class SplitRepartitionByColTaskScheduler(ExchangeTaskScheduler):
    """
    Task scheduler for (non-shuffle) repartition by column.

    Implementation:

    Repartition-by-column is used to ensure the group boundaries
    are aligned with the block boundaries. This means that each
    group is fully contained in N blocks. We assume the keys are
    continuous within each block before repartition. This is not
    intended to be a general groupby ops.

    1. Map-stage: each block is split into blocks with a single
    group. For example, a dataset with N input blocks and K groups,
    the output of the map stage will be between max(N, K) and N * K,
    depending on how many splits per block.

    2. Merge-stage: Given a list of mapped-blocks, we merge them
    if they belong to the same group. The output of this stage is
    a list of blocks with a single group.

    Note:
        By default, We do not explicitly sort the blocks.
    """

    # TODO: Preserve the ordering of the input blocks. A naive implementation
    # is to add a single-value column to label the ordering
    # TODO: Allow control of target_max_block_size
    # For target_max_block_size, it could be implemented in both map
    # and reduce stages.
    # TODO: Allow setting output_num_blocks. When it is smaller than the
    # number of groups, each output block may contain more than one group
    # On the other hand, if output_num_blocks is large, one group may be
    # split into multiple blocks. User should have control to this behavior
    # TODO: Allow sorting within the block (based on a column other than the
    # partition key). See Spark RDD's repartitionAndSortWithinPartitions

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

        num_bundles = len(refs)

        queue = Queue()
        out_queue = Queue()  # for passing map results into coordinator
        meta_queue = Queue()  # output as a flattened list of metadata
        coordinator = Coordinator.options(**reduce_ray_remote_args).remote(
            *self._exchange_spec._reduce_args, queue, out_queue, meta_queue
        )

        t1 = time.perf_counter()
        # Perform map-task to all the fragments and submit the output to a queue.
        # Coordinator then reads the queue

        # Each ref_bundle is coming from one ReadTask
        # split_block_refs[i] = a list of subblocks split

        task_refs = []
        for bundle_id, ref_bundle in enumerate(refs):
            blocks_ref = [block for block, _ in ref_bundle.blocks]
            ref = process_fragment_task.options(
                **map_ray_remote_args,
            ).remote(bundle_id, blocks_ref, queue, *self._exchange_spec._map_args)
            task_refs.append(ref)

        # Ensure all the fragments are taken care of
        while out_queue.qsize() < num_bundles:
            # print("waiting the queue to finish...", out_queue.qsize())
            time.sleep(0.2)

        _logger = logger.get_logger()
        _logger.debug(f"Finished submitting map-tasks for {bundle_id+1} fragments")

        # not sure if needed
        ray.get(task_refs)

        t2 = time.perf_counter()
        map_stage_time = t2 - t1
        _logger.debug(f"map-stage taken {map_stage_time:0.3f}s")

        # reduce_block_refs = []
        # for i, reducer_ref in reducers.items():
        #     mapping = ray.get(reducer_ref.get_mapping.remote())
        #     for key, value in mapping.items():
        #         print(key, type(value))
        #         reduce_block_refs.append(value)

        # Setup progress bar for the map tasks
        # and fetch the metadata
        sub_progress_bar_dict = ctx.sub_progress_bar_dict
        bar_name = (
            RepartitionByColTaskSpec.REPARTITION_BY_COLUMN_SPLIT_SUB_PROGRESS_BAR_NAME
        )
        assert bar_name in sub_progress_bar_dict, sub_progress_bar_dict
        map_bar = sub_progress_bar_dict[bar_name]

        split_map_metadata: List[BlockMetadata] = []
        n = meta_queue.qsize()
        while meta_queue.qsize() > 0:
            meta = meta_queue.get()
            split_map_metadata.extend(meta)
            map_bar.update(len(split_map_metadata), total=n)

        _logger.debug("number of remaining tasks: %d", queue.qsize())
        message_counts = ray.get(coordinator.get_message_counts.remote())
        _logger.debug("number of messages per fragment = %d", message_counts)
        message_counts_after_end = sum(
            ray.get(coordinator.get_message_counts_after_end.remote()).values()
        )
        _logger.debug("number of messages after end = %d", message_counts_after_end)

        # # Setup progress bar for the reduce tasks
        # # and fetch the metadata
        sub_progress_bar_dict = ctx.sub_progress_bar_dict
        bar_name = (
            RepartitionByColTaskSpec.REPARTITION_BY_COLUMN_MERGE_SUB_PROGRESS_BAR_NAME
        )
        assert bar_name in sub_progress_bar_dict, sub_progress_bar_dict
        reduce_bar = sub_progress_bar_dict[bar_name]

        # reduce_bar.fetch_until_complete

        reduce_block_refs = []
        reduce_metadata = []
        n = out_queue.qsize()
        while out_queue.qsize() > 0:
            idx, key, block, meta = out_queue.get()
            # print(idx, key)
            reduce_block_refs.append(block)
            reduce_metadata.append(meta)
            reduce_bar.update(len(reduce_metadata), total=n)

        # reduce_block_refs, reduce_metadata = zip(*reduce_return)
        # reduce_metadata = reduce_bar.fetch_until_complete(list(reduce_metadata))
        # reduce_block_refs, reduce_metadata = list(reduce_block_refs), list(
        #     reduce_metadata
        # )

        # TODO: figure out if we need to fill in empty blocks
        # since output_num_blocks is None..

        # Fill empty blocks.
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
        stats: StatsDict = {
            # "split": split_map_metadata,
            "reduce": reduce_metadata,
        }

        logger.get_logger().info("number of items in queue: %d", queue.qsize())

        return (output, stats)
