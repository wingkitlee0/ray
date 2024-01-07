import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import ray
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import RefBundle, TaskContext
from ray.data._internal.planner.exchange.interfaces import ExchangeTaskScheduler
from ray.data._internal.planner.exchange.repartition_task_spec import (
    RepartitionByColumnTaskSpec,
)
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.repartition_by_column import Actor, repartition_by_column
from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata
from ray.types import ObjectRef

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

        if use_batching:
            repartitioned_refs = []
            all_actors = []
            for ref_id, ref_bundle in enumerate(refs):
                actors = [
                    Actor.options(name=f"Actor-({ref_id, i})").remote(
                        i, keys, num_actors
                    )
                    for i in range(num_actors)
                ]
                ref = asyncio.run(
                    repartition_by_column(
                        ref_id,
                        [b for b, _ in ref_bundle.blocks],
                        *self._exchange_spec._map_args,
                        actors,
                    )
                )
                all_actors.extend(actors)
            repartitioned_refs.extend(ref)
        else:
            ref_id = 0
            # drop the metadata
            all_blocks = [b for ref_bundle in refs for b, _ in ref_bundle.blocks]

            logger.get_logger().info(f"ref_id: {ref_id}, {len(all_blocks)=}")

            actors = [
                Actor.options(name=f"Actor-({ref_id, i})").remote(i, keys, num_actors)
                for i in range(num_actors)
            ]

            repartitioned_refs = asyncio.run(
                repartition_by_column(
                    ref_id,
                    all_blocks,
                    *self._exchange_spec._map_args,
                    actors,
                )
            )

        logger.get_logger().info(f"Finished repartitioning")

        # Looping over num_actors
        all_blocks, all_metadata, all_keys = [], [], []
        for i, blocks_and_metadata in enumerate(repartitioned_refs):
            _blocks, _metdata, _keys = [], [], []
            for block, meta, key in blocks_and_metadata:
                _blocks.append(block)
                _metdata.append(meta)
                _keys.append(key)

            logger.get_logger().info(
                f"repartition-{i}: {len(_keys)=}, {min(_keys)=}, {max(_keys)=}"
            )

            all_blocks.extend(
                [
                    RefBundle([(block, meta)], input_owned_by_consumer)
                    for block, meta in zip(_blocks, _metdata)
                ]
            )
            all_metadata.extend(_metdata)
            all_keys.extend(_keys)

        assert len(all_blocks) == len(all_metadata) == len(all_keys)

        logger.get_logger().info(f"number of output blocks = {len(all_blocks)}")
        logger.get_logger().info(f"number of keys = {len(all_keys)}")

        # TODO: add progress bar
        # TODO: use reduce_bar.fetch_until_complete etc
        # TODO: handle block metadata better

        stats = {"repartition": all_metadata}

        time_end = time.perf_counter()
        logger.get_logger().info(f"repartition time = {(time_end - time_start):.4}s")

        return (all_blocks, stats)
