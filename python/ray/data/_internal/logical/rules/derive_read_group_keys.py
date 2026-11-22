"""Group a hive parquet read by the keys of a downstream ``groupby`` at plan time.

A hive read picks its file partitioner when it is built, before any downstream
operator exists. This rule sees the whole plan, so it can swap the partitioner
of the ``ListFiles`` feeding a ``Repartition(keys=...)`` (what ``groupby(keys)``
plans as) for one that emits one read unit per key value, instead of asking the
user to name the key up front.
"""

import copy
from dataclasses import replace
from typing import Dict, Optional, Tuple

from ray.data._internal.datasource_v2.common.hive_partition_grouper import (
    HivePartitionGrouper,
)
from ray.data._internal.logical.interfaces import LogicalOperator, LogicalPlan, Rule
from ray.data._internal.logical.operators import (
    Aggregate,
    Filter,
    Limit,
    ListFiles,
    ReadFiles,
    Repartition,
)
from ray.data.datasource.partitioning import Partitioning, PartitionStyle

__all__ = [
    "DeriveReadGroupKeys",
]

# Operators between the read and the group-by that keep every row's key values
# unchanged, so a key that is confined to one read unit stays confined. Any other
# operator may rewrite, drop or reorder key columns or rows and is a barrier.
# ``Filter`` drops rows only. ``Limit`` keeps a prefix of the rows, so each key's
# rows still come from one read unit.
_KEY_PRESERVING_OPS = (Filter, Limit)


class DeriveReadGroupKeys(Rule):
    """Replace a hive read's partitioner with one that groups by the group-by keys.

    A ``Repartition(keys=...)`` is what ``groupby(keys)`` plans as. When the chain
    from it down to ``ReadFiles`` only passes through key-preserving operators,
    and the keys are hive fields of the read, the ``ListFiles`` under that
    ``ReadFiles`` gets a :class:`HivePartitionGrouper` that emits one read unit per
    key value. The previous partitioner becomes the grouper's fallback for paths
    without hive keys.

    The rule changes only the listing. It does not declare the read's output
    partitioning or remove the repartition; those are separate steps.

    Gated by ``DataContext.enable_hive_key_read_grouping``, off by default.
    """

    def apply(self, plan: LogicalPlan) -> LogicalPlan:  # pyrefly: ignore[bad-override]
        if not plan.context.enable_hive_key_read_grouping:
            return plan

        # Pass 1: find the listings to change. Keyed by identity so pass 2 replaces
        # exactly those nodes.
        targets: Dict[int, HivePartitionGrouper] = {}
        keys_for_listing: Dict[int, Tuple[str, ...]] = {}
        for node in plan.dag.post_order_iter():
            keys = _group_keys_of(node)
            if keys is None:
                continue
            read = _read_below(node)
            if read is None:
                continue
            (listing,) = read.input_dependencies
            if not isinstance(listing, ListFiles):
                continue
            grouper = _grouper_for(listing, read, keys)
            if grouper is not None:
                targets[id(listing)] = grouper
                keys_for_listing[id(listing)] = keys
        if not targets:
            return plan

        # Pass 2, bottom-up. Swap each listing's partitioner, then give its
        # ``ReadFiles`` the same keys on its scanner, so the physical read declares
        # ``KeyGroupedPartitioning`` and emits one block per key group. The rewritten
        # listings are kept alive so their ids stay unique while the pass runs.
        rewritten: Dict[int, Tuple[ListFiles, Tuple[str, ...]]] = {}

        def transform(node: LogicalOperator) -> LogicalOperator:
            if isinstance(node, ListFiles) and id(node) in targets:
                new_listing = replace(node, file_partitioner=targets[id(node)])
                rewritten[id(new_listing)] = (new_listing, keys_for_listing[id(node)])
                return new_listing
            if isinstance(node, ReadFiles):
                (listing,) = node.input_dependencies
                if id(listing) in rewritten:
                    _, keys = rewritten[id(listing)]
                    # Copy rather than ``replace``: the copy keeps the input link.
                    target = copy.copy(node)
                    object.__setattr__(
                        target, "scanner", replace(node.scanner, group_keys=keys)
                    )
                    return target
            return node

        dag = plan.dag._apply_transform(transform)
        return LogicalPlan(dag=dag, context=plan.context)


def _group_keys_of(node: LogicalOperator) -> Optional[Tuple[str, ...]]:
    """The group keys of a ``groupby`` node: a ``Repartition`` (map_groups) or an
    ``Aggregate``. None for any other node, or a global aggregation."""
    if isinstance(node, Repartition):
        keys = node.keys
    elif isinstance(node, Aggregate):
        keys = node.key
    else:
        return None
    if not keys:
        return None
    return (keys,) if isinstance(keys, str) else tuple(keys)


def _read_below(group_by: LogicalOperator) -> Optional[ReadFiles]:
    """The ``ReadFiles`` under ``group_by`` if only key-preserving ops sit between."""
    (op,) = group_by.input_dependencies
    while isinstance(op, _KEY_PRESERVING_OPS):
        (op,) = op.input_dependencies
    return op if isinstance(op, ReadFiles) else None


def _grouper_for(
    listing: ListFiles, read: ReadFiles, keys: Tuple[str, ...]
) -> Optional[HivePartitionGrouper]:
    """A grouper by ``keys`` for ``listing``, or None to keep its partitioner."""
    partitioner = listing.file_partitioner
    # No partitioner means no fallback for paths that carry no hive keys.
    if partitioner is None:
        return None
    partitioning: Optional[Partitioning] = getattr(read.scanner, "partitioning", None)
    if partitioning is None or partitioning.style != PartitionStyle.HIVE:
        return None
    hive_fields = set(partitioning.field_names or [])
    if not hive_fields or not set(keys) <= hive_fields:
        return None
    if isinstance(partitioner, HivePartitionGrouper):
        # Already grouped at read time, by every hive key. That is correct for this
        # group-by only when the keys are exactly those.
        return None
    return HivePartitionGrouper(partitioning, fallback=partitioner, group_keys=keys)
