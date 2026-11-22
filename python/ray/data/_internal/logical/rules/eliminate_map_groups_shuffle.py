from typing import Callable, Iterable, List, Tuple, Type

import pyarrow as pa

from ray.data._internal.compute import TaskPoolStrategy
from ray.data._internal.execution.interfaces import PhysicalOperator, TaskContext
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.map_transformer import (
    BlockMapTransformFn,
    MapTransformer,
)
from ray.data._internal.execution.operators.shuffle_operators.shuffle_map_operator import (  # noqa: E501
    ShuffleMapOp,
)
from ray.data._internal.execution.operators.shuffle_operators.shuffle_reduce_operator import (  # noqa: E501
    ShuffleReduceOp,
)
from ray.data._internal.logical.interfaces import (
    ClusteredDistribution,
    KeyGroupedPartitioning,
    PhysicalPlan,
    Plan,
    Rule,
)
from ray.data._internal.logical.operators import MapBatches, Repartition
from ray.data._internal.logical.rules.eliminate_aggregate_shuffle import (
    _shuffle_ops_before,
)
from ray.data._internal.logical.rules.inherit_target_max_block_size import (
    InheritTargetMaxBlockSizeRule,
)
from ray.data._internal.logical.rules.operator_fusion import FuseOperators
from ray.data._internal.output_buffer import OutputBlockSizeOption
from ray.data.block import Block, BlockAccessor

__all__ = ["EliminateMapGroupsShuffle"]


def _make_sort_partition_fn(
    key_columns: Tuple[str, ...]
) -> Callable[[Iterable[Block], TaskContext], Iterable[Block]]:
    """Concatenate a whole partition's blocks into one block sorted by the keys.

    The UDF's group slicing assumes each block is sorted by the keys, which the
    hash repartition's sorting reduce provides.
    """

    def _sort(blocks: Iterable[Block], ctx: TaskContext) -> Iterable[Block]:
        tables: List[pa.Table] = [
            BlockAccessor.for_block(block).to_arrow() for block in blocks
        ]
        tables = [table for table in tables if table.num_rows > 0]
        if not tables:
            return
        combined = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        yield combined.sort_by([(key, "ascending") for key in key_columns])

    return _sort


class EliminateMapGroupsShuffle(Rule):
    """Skips a map_groups' hash repartition when its input is already
    hash-partitioned on a subset of the group keys, e.g. by a join on those keys::

        JoinShuffleReduce -> HashShuffleMap -> SortingReduce -> MapBatches  # before
        JoinShuffleReduce -> SortPartition -> MapBatches                    # after

    The replacement sorts each whole partition in one task, as the reduce did.
    Its input is the join's reduce, which is told to emit whole partitions, so
    the task-pool map gets each partition as one bundle.
    """

    @classmethod
    def dependencies(cls) -> List[Type[Rule]]:
        # So the sort copies the reduce's inherited block size.
        return [InheritTargetMaxBlockSizeRule]

    @classmethod
    def dependents(cls) -> List[Type[Rule]]:
        return [FuseOperators]

    def apply(self, plan: Plan) -> Plan:
        assert isinstance(plan, PhysicalPlan)
        if not plan.context.enable_map_groups_shuffle_elimination:
            return plan
        self._op_map = plan.op_map.copy()
        dag = self._eliminate_in_dag(plan.dag)
        return PhysicalPlan(dag, self._op_map, plan.context)

    def _eliminate_in_dag(self, op: PhysicalOperator) -> PhysicalOperator:
        op._input_dependencies = [
            self._eliminate_in_dag(input_op) for input_op in op.input_dependencies
        ]
        self._try_replace_repartition(op)
        return op

    def _try_replace_repartition(self, map_op: PhysicalOperator) -> None:
        """Rewire ``map_op`` to read a sort of its input instead of the
        repartition's shuffle, if the input already has the group keys' layout."""
        if not isinstance(map_op, MapOperator) or len(map_op.input_dependencies) != 1:
            return
        (reduce_op,) = map_op.input_dependencies
        if not isinstance(reduce_op, ShuffleReduceOp):
            return
        # Fused maps would be dropped with the reduce. Only a shuffle restored by
        # an earlier pass has one, and this rule doesn't run after that.
        if reduce_op._fused_output_map_transformer is not None:
            return
        repartition = self._op_map.get(reduce_op)
        if not (
            isinstance(repartition, Repartition)
            and repartition.keys
            and repartition.sort
            and isinstance(self._op_map.get(map_op), MapBatches)
        ):
            return

        shuffle_ops = _shuffle_ops_before(reduce_op)
        if shuffle_ops is None or not isinstance(shuffle_ops[-1], ShuffleMapOp):
            return
        shuffle_map = shuffle_ops[-1]
        (source,) = shuffle_map.input_dependencies
        # The source must still feed only this shuffle, so its emitted bundles are
        # whole partitions only for this consumer.
        if len(source.output_dependencies) != 1:
            return

        key_columns = tuple(repartition.keys)
        partitioning = source.output_partitioning
        if not ClusteredDistribution(key_columns).is_satisfied_by(partitioning):
            return
        key_grouped = isinstance(partitioning, KeyGroupedPartitioning)
        # Keep the shuffle if the group-by asks for another partition count. A
        # key-grouped read has no count (its groups come from the data), and the
        # count only sets parallelism, so the groups are not affected.
        if (
            not key_grouped
            and repartition.num_outputs is not None
            and getattr(partitioning, "num_partitions", None) != repartition.num_outputs
        ):
            return
        if not isinstance(source, ShuffleReduceOp):
            # A key-grouped read: each group is already one sorted block, so
            # map_batches reads the read directly. Whole partitions can't be
            # asked of a read, so no SortPartition op is added.
            if not key_grouped:
                return
            self._bypass_shuffle(map_op, shuffle_map, reduce_op, source)
            return

        source.set_emit_whole_partitions()
        transformer = MapTransformer(
            [
                BlockMapTransformFn(
                    _make_sort_partition_fn(key_columns),
                    output_block_size_option=OutputBlockSizeOption.of(
                        target_max_block_size=reduce_op.data_context.target_max_block_size
                    ),
                )
            ]
        )
        sort_op = MapOperator.create(
            transformer,
            source,
            reduce_op.data_context,
            target_max_block_size_override=reduce_op.target_max_block_size_override,
            name=f"SortPartition(key_columns={key_columns})",
            compute_strategy=TaskPoolStrategy(),
            # Fused into the join's reduce tasks, it would add to their memory.
            supports_fusion=False,
        )
        sort_op.set_logical_operators(repartition)

        # create() already linked source to sort_op; drop the shuffle's ops.
        source._output_dependencies.remove(shuffle_map)
        map_op._input_dependencies = [sort_op]
        sort_op._output_dependencies = [map_op]
        for op in (shuffle_map, reduce_op):
            self._op_map.pop(op)
        self._op_map[sort_op] = repartition

    def _bypass_shuffle(
        self,
        map_op: MapOperator,
        shuffle_map: PhysicalOperator,
        reduce_op: PhysicalOperator,
        source: PhysicalOperator,
    ) -> None:
        """Connect ``source`` straight to ``map_op`` and drop the shuffle's ops."""
        source._output_dependencies = [map_op]
        map_op._input_dependencies = [source]
        for op in (shuffle_map, reduce_op):
            self._op_map.pop(op)
