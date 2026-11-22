from typing import List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import ray
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.operators.shuffle_operators.shuffle_map_operator import (  # noqa: E501
    ShuffleMapOp,
)
from ray.data._internal.datasource_v2.formats.parquet.parquet_datasource_v2 import (
    ParquetDatasourceV2,
)
from ray.data._internal.logical.optimizers import get_execution_plan
from ray.data.context import ShuffleStrategy
from ray.data.dataset import Dataset
from ray.data.tests.conftest import *  # noqa: F401, F403
from ray.tests.conftest import *  # noqa: F401, F403

NUM_PARTITIONS = 4
NUM_ROWS = 120


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _from_columns(**columns: np.ndarray) -> Dataset:
    table = pa.table(columns)
    chunks = np.array_split(np.arange(table.num_rows), 6)
    return ray.data.from_arrow([table.take(idx) for idx in chunks])


def _join() -> Dataset:
    """Joins on ``id``, so its output is hash-partitioned on ``id`` only."""
    ids = np.arange(NUM_ROWS)
    left = _from_columns(id=ids, grp=ids % 4, lval=ids * 2)
    right_ids = np.arange(2 * NUM_ROWS)
    right = _from_columns(rid=right_ids % NUM_ROWS, rval=right_ids * 3)
    return left.join(
        right,
        on=("id",),
        right_on=("rid",),
        join_type="inner",
        num_partitions=NUM_PARTITIONS,
    )


def _map_groups(
    ds: Dataset, keys: List[str], num_partitions: Optional[int] = None
) -> Dataset:
    def summarize(df: pd.DataFrame) -> pd.DataFrame:
        # One output row per group: its keys, row count and value sum. Defined
        # here so Ray serializes it by value; workers can't import this module.
        out = df[keys].iloc[:1].reset_index(drop=True)
        out["n"] = len(df)
        out["s"] = int(df["lval"].sum() + df["rval"].sum())
        return out

    return ds.groupby(keys, num_partitions=num_partitions).map_groups(
        summarize, batch_format="pandas"
    )


def _physical_ops(ds: Dataset) -> List[PhysicalOperator]:
    physical_plan, _ = get_execution_plan(ds._logical_plan)
    return list(physical_plan.dag.post_order_iter())


def _has_repartition_shuffle(ds: Dataset) -> bool:
    # The join's own shuffle maps are named JoinShuffleMap*; only the group-by's
    # hash repartition is named HashShuffleMap.
    return any(
        isinstance(op, ShuffleMapOp) and op.name.startswith("HashShuffleMap")
        for op in _physical_ops(ds)
    )


def _has_sort_partition(ds: Dataset) -> bool:
    return any("SortPartition" in op.name for op in _physical_ops(ds))


def _assert_same_rows(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    columns = sorted(expected.columns)
    actual = actual[columns].sort_values(columns).reset_index(drop=True)
    expected = expected[columns].sort_values(columns).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


@pytest.fixture
def shuffle_v2(restore_data_context):
    restore_data_context.shuffle_strategy = ShuffleStrategy.SHUFFLE_V2
    restore_data_context.use_disk_based_hash_shuffle = False
    return restore_data_context


# ---------------------------------------------------------------------------
# Plan shape
# ---------------------------------------------------------------------------


def test_map_groups_skips_shuffle_on_join_keys(
    ray_start_regular_shared_2_cpus, shuffle_v2
):
    # The join partitions on ``id``; grouping on ``id`` and ``grp`` is a superset.
    # The flag must be set before building the dataset, which snapshots the context.
    shuffle_v2.enable_map_groups_shuffle_elimination = True
    ds = _map_groups(_join(), ["id", "grp"], num_partitions=NUM_PARTITIONS)
    assert not _has_repartition_shuffle(ds)
    assert _has_sort_partition(ds)


def test_map_groups_keeps_shuffle_when_disabled(
    ray_start_regular_shared_2_cpus, shuffle_v2
):
    shuffle_v2.enable_map_groups_shuffle_elimination = False
    ds = _map_groups(_join(), ["id", "grp"], num_partitions=NUM_PARTITIONS)
    assert _has_repartition_shuffle(ds)
    assert not _has_sort_partition(ds)


@pytest.mark.parametrize(
    "keys,num_partitions",
    [
        # Not a superset of the partitioning columns.
        (["grp"], NUM_PARTITIONS),
        # The group-by asks for a different partition count.
        (["id", "grp"], NUM_PARTITIONS + 1),
    ],
)
def test_map_groups_keeps_shuffle_when_not_eliminable(
    ray_start_regular_shared_2_cpus, shuffle_v2, keys, num_partitions
):
    shuffle_v2.enable_map_groups_shuffle_elimination = True
    ds = _map_groups(_join(), keys, num_partitions=num_partitions)
    assert _has_repartition_shuffle(ds)
    assert not _has_sort_partition(ds)


# ---------------------------------------------------------------------------
# Correctness: same rows with and without the elimination
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "keys,num_partitions",
    [
        (["id", "grp"], NUM_PARTITIONS),
        (["id"], NUM_PARTITIONS),
        (["grp"], NUM_PARTITIONS),
        (["id", "grp"], NUM_PARTITIONS + 1),
    ],
)
def test_map_groups_output_matches_without_elimination(
    ray_start_regular_shared_2_cpus, shuffle_v2, keys, num_partitions
):
    shuffle_v2.enable_map_groups_shuffle_elimination = False
    expected = _map_groups(_join(), keys, num_partitions).to_pandas()

    shuffle_v2.enable_map_groups_shuffle_elimination = True
    actual = _map_groups(_join(), keys, num_partitions).to_pandas()

    _assert_same_rows(actual, expected)


# ---------------------------------------------------------------------------
# Key-grouped parquet reads
# ---------------------------------------------------------------------------


@pytest.fixture
def hive_dir(tmp_path):
    """Four ``k=`` directories, three files each, so each key spans files."""
    for k in range(4):
        key_dir = tmp_path / f"k={k}"
        key_dir.mkdir()
        for f in range(3):
            ids = np.arange(NUM_ROWS)
            pq.write_table(
                pa.table({"lval": ids * 2 + f, "rval": ids * 3}),
                key_dir / f"part{f}.parquet",
            )
    return tmp_path


@pytest.fixture
def key_grouped_read(monkeypatch):
    # ``read_parquet`` has no public way to request key grouping, so wrap the
    # constructor to set the private ``_group_keys`` that the reader reads.
    init = ParquetDatasourceV2.__init__

    def init_with_group_keys(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self._group_keys = ("k",)

    monkeypatch.setattr(ParquetDatasourceV2, "__init__", init_with_group_keys)


def test_map_groups_read_skips_shuffle_when_key_grouped(
    ray_start_regular_shared_2_cpus, shuffle_v2, hive_dir, key_grouped_read
):
    shuffle_v2.enable_map_groups_shuffle_elimination = True
    ds = _map_groups(ray.data.read_parquet(str(hive_dir)), ["k"])
    assert not _has_repartition_shuffle(ds)
    assert not _has_sort_partition(ds)


def test_map_groups_read_keeps_shuffle_without_key_grouping(
    ray_start_regular_shared_2_cpus, shuffle_v2, hive_dir
):
    shuffle_v2.enable_map_groups_shuffle_elimination = True
    ds = _map_groups(ray.data.read_parquet(str(hive_dir)), ["k"])
    assert _has_repartition_shuffle(ds)


def test_map_groups_read_output_matches_without_elimination(
    ray_start_regular_shared_2_cpus, shuffle_v2, hive_dir, key_grouped_read
):
    # Depends on the reader emitting each key as one sorted block; fails without it.
    shuffle_v2.enable_map_groups_shuffle_elimination = False
    expected = _map_groups(ray.data.read_parquet(str(hive_dir)), ["k"]).to_pandas()

    shuffle_v2.enable_map_groups_shuffle_elimination = True
    actual = _map_groups(ray.data.read_parquet(str(hive_dir)), ["k"]).to_pandas()

    _assert_same_rows(actual, expected)
