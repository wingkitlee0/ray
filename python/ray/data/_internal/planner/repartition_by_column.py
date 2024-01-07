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
from ray.data._internal.planner.exchange.repartition_task_scheduler import (
    RepartitionByColumnTaskScheduler,
)
from ray.data._internal.planner.exchange.repartition_task_spec import (
    RepartitionByColumnTaskSpec,
)
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata

logger = DatasetLogger(__name__)


def generate_repartition_by_column_fn(
    keys: Union[str, List[str]],
    num_actors_per_stream: int,
    use_batching: bool,
    ray_remote_args: Optional[Dict[str, Any]],
) -> AllToAllTransformFn:
    """Generate function to split blocks by the specified key column"""

    def fn(
        refs: List[RefBundle],
        ctx: TaskContext,
        keys: Union[str, List[str]],
        num_actors_per_stream: int,
        use_batching: bool,
        ray_remote_args: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[RefBundle], StatsDict]:
        repartition_task_spec = RepartitionByColumnTaskSpec(
            keys=keys,
            num_actors_per_stream=num_actors_per_stream,
            use_batching=use_batching,
        )
        scheduler = RepartitionByColumnTaskScheduler(repartition_task_spec)

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
    return partial(
        fn,
        keys=keys,
        num_actors_per_stream=num_actors_per_stream,
        use_batching=use_batching,
        ray_remote_args=ray_remote_args,
    )
