import time
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import ray
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import RefBundle, TaskContext
from ray.data._internal.planner.exchange.interfaces import ExchangeTaskScheduler
from ray.data._internal.planner.exchange.repartition_task_spec import (
    RepartitionByColumnTaskSpec,
)
from ray.data._internal.repartition_by_column import Actor, repartition_runner
from ray.data._internal.stats import StatsDict

logger = DatasetLogger(__name__)


KeyType = TypeVar("KeyType")


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
        time_start = time.perf_counter()

        input_owned_by_consumer = all(rb.owns_blocks for rb in refs)

        logger.get_logger().info(f"number of RefBundles = {len(refs)}")

        if map_ray_remote_args is None:
            map_ray_remote_args = {}
        if reduce_ray_remote_args is None:
            reduce_ray_remote_args = {}
        if "scheduling_strategy" not in reduce_ray_remote_args:
            reduce_ray_remote_args = reduce_ray_remote_args.copy()
            reduce_ray_remote_args["scheduling_strategy"] = "SPREAD"

        keys, num_actors, use_batching = self._exchange_spec._map_args

        ref_id = 0
        # drop the metadata
        all_blocks = [b for ref_bundle in refs for b, _ in ref_bundle.blocks]

        logger.get_logger().info(f"ref_id: {ref_id}, {len(all_blocks)=}")

        actors = [
            Actor.options(name=f"Actor-({ref_id, i})").remote(i, keys, num_actors)
            for i in range(num_actors)
        ]

        result_refs = list(
            runner.remote(
                ref_id,
                all_blocks,
                self._exchange_spec._map_args,
                actors,
            )
        )

        logger.get_logger().info(f"Finished repartitioning")
        logger.get_logger().info(f"result_refs: {len(result_refs)}")

        all_keys = []
        for _ in range(num_actors):
            all_keys.append(result_refs.pop())

        all_metadata = []
        for _ in range(num_actors):
            all_metadata.append(result_refs.pop())

        sub_progress_bar_dict = ctx.sub_progress_bar_dict
        bar_name = RepartitionByColumnTaskSpec.SPLIT_SUB_PROGRESS_BAR_NAME
        assert bar_name in sub_progress_bar_dict, sub_progress_bar_dict
        map_bar = sub_progress_bar_dict[bar_name]

        all_metadata = map_bar.fetch_until_complete(all_metadata)
        all_metadata = [m for metadata in all_metadata for m in metadata]

        all_keys = map_bar.fetch_until_complete(all_keys)
        all_keys = [m for keys in all_keys for m in keys]

        logger.get_logger().info(f"all_keys: {len(all_keys)}")
        logger.get_logger().info(f"all_metadata: {len(all_metadata)}")
        logger.get_logger().info(f"all_blocks: {len(result_refs)}")

        all_blocks = [
            RefBundle([(block, meta)], input_owned_by_consumer)
            for block, meta in zip(result_refs, all_metadata)
        ]

        assert (
            len(all_blocks) == len(all_metadata) == len(all_keys)
        ), f"{len(all_blocks)=}, {len(all_metadata)=}, {len(all_keys)=}"

        logger.get_logger().info(f"number of output blocks = {len(all_blocks)}")
        logger.get_logger().info(f"number of keys = {len(all_keys)}")

        # TODO: add progress bar
        # TODO: use reduce_bar.fetch_until_complete etc
        # TODO: handle block metadata better

        stats = {"repartition": all_metadata}

        time_end = time.perf_counter()
        logger.get_logger().info(f"repartition time = {(time_end - time_start):.4}s")

        return (all_blocks, stats)
