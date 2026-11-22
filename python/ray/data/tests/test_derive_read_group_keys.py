import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import ray
from ray.data._internal.datasource_v2.common.hive_partition_grouper import (
    HivePartitionGrouper,
)
from ray.data._internal.logical.operators import ListFiles
from ray.data._internal.logical.optimizers import LogicalOptimizer, get_execution_plan
from ray.data.context import DataContext
from ray.data.tests.conftest import *  # noqa: F401, F403
from ray.tests.conftest import *  # noqa: F401, F403


def _write_hive(root, keys=("a", "b", "c")):
    for key in keys:
        for i in range(2):
            part_dir = os.path.join(root, f"k={key}")
            os.makedirs(part_dir, exist_ok=True)
            pq.write_table(
                pa.table({"v": [i, i + 1]}),
                os.path.join(part_dir, f"{i}.parquet"),
            )


def _optimized_listing(ds):
    plan = LogicalOptimizer().optimize(ds._logical_plan)
    listings = [op for op in plan.dag.post_order_iter() if isinstance(op, ListFiles)]
    assert len(listings) == 1
    return listings[0]


def _physical_ops(ds):
    physical_plan, _ = get_execution_plan(ds._logical_plan)
    return list(physical_plan.dag.post_order_iter())


@pytest.fixture
def grouping_enabled(restore_data_context):
    DataContext.get_current().enable_hive_key_read_grouping = True


def test_groupby_key_reaches_hive_listing(
    ray_start_regular_shared, tmp_path, grouping_enabled
):
    _write_hive(str(tmp_path))
    ds = ray.data.read_parquet(str(tmp_path)).groupby("k").map_groups(lambda b: b)

    listing = _optimized_listing(ds)

    assert isinstance(listing.file_partitioner, HivePartitionGrouper)


def test_map_between_read_and_groupby_blocks_grouping(
    ray_start_regular_shared, tmp_path, grouping_enabled
):
    _write_hive(str(tmp_path))
    ds = (
        ray.data.read_parquet(str(tmp_path))
        .map_batches(lambda b: b.assign(k=b["k"].astype(str) + "x"))
        .groupby("k")
        .map_groups(lambda b: b)
    )

    listing = _optimized_listing(ds)

    assert not isinstance(listing.file_partitioner, HivePartitionGrouper)


def test_non_hive_key_gets_no_grouping(
    ray_start_regular_shared, tmp_path, grouping_enabled
):
    _write_hive(str(tmp_path))
    ds = ray.data.read_parquet(str(tmp_path)).groupby("v").map_groups(lambda b: b)

    listing = _optimized_listing(ds)

    assert not isinstance(listing.file_partitioner, HivePartitionGrouper)


def test_flag_off_leaves_listing_unchanged(
    ray_start_regular_shared, tmp_path, restore_data_context
):
    _write_hive(str(tmp_path))
    ds = ray.data.read_parquet(str(tmp_path)).groupby("k").map_groups(lambda b: b)

    listing = _optimized_listing(ds)

    assert not isinstance(listing.file_partitioner, HivePartitionGrouper)


def test_public_map_groups_read_has_no_hash_shuffle(tmp_path, grouping_enabled):
    _write_hive(str(tmp_path), keys=("a", "b", "c"))
    ds = ray.data.read_parquet(str(tmp_path)).groupby("k").map_groups(lambda b: b)
    names = [op.name for op in _physical_ops(ds)]
    assert not any("HashShuffle" in name for name in names), names


def test_public_map_groups_read_matches_default_output(tmp_path, restore_data_context):
    _write_hive(str(tmp_path), keys=("a", "b", "c"))

    def per_group_sum(batch):
        return {"k": [batch["k"][0]], "total": [int(sum(batch["v"]))]}

    def run(flag):
        DataContext.get_current().enable_hive_key_read_grouping = flag
        ds = ray.data.read_parquet(str(tmp_path)).groupby("k").map_groups(per_group_sum)
        return sorted((r["k"], r["total"]) for r in ds.take_all())

    assert run(True) == run(False)


def test_orc_read_with_key_grouping_flag_on(tmp_path, grouping_enabled):
    import pyarrow.orc as orc

    orc.write_table(pa.table({"v": [1, 2, 3]}), str(tmp_path / "a.orc"))
    rows = ray.data.read_orc(str(tmp_path)).take_all()
    assert sorted(r["v"] for r in rows) == [1, 2, 3]


def test_public_aggregate_read_has_no_hash_shuffle(tmp_path, grouping_enabled):
    from ray.data.aggregate import Sum

    _write_hive(str(tmp_path), keys=("a", "b", "c"))
    ds = ray.data.read_parquet(str(tmp_path)).groupby("k").aggregate(Sum("v"))
    names = [op.name for op in _physical_ops(ds)]
    assert not any("HashAggregate" in name and "Reduce" in name for name in names)
    assert not any("HashShuffle" in name for name in names), names


def test_public_aggregate_read_matches_default_output(tmp_path, restore_data_context):
    from ray.data.aggregate import Sum

    _write_hive(str(tmp_path), keys=("a", "b", "c"))

    def run(flag):
        DataContext.get_current().enable_hive_key_read_grouping = flag
        ds = ray.data.read_parquet(str(tmp_path)).groupby("k").aggregate(Sum("v"))
        return sorted((r["k"], r["sum(v)"]) for r in ds.take_all())

    default, grouped = run(False), run(True)
    assert [k for k, _ in default] == [k for k, _ in grouped]
    assert all(pytest.approx(a) == b for (_, a), (_, b) in zip(default, grouped)), (
        default,
        grouped,
    )
