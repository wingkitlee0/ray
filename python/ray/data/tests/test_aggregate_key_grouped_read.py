import glob

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import ray
from ray.data._internal.datasource_v2.formats.parquet.parquet_datasource_v2 import (
    ParquetDatasourceV2,
)
from ray.data._internal.execution.operators.shuffle_operators.shuffle_map_operator import (  # noqa: E501
    ShuffleMapOp,
)
from ray.data._internal.logical.optimizers import get_execution_plan
from ray.data.context import ShuffleStrategy
from ray.data.tests.conftest import *  # noqa: F401, F403
from ray.tests.conftest import *  # noqa: F401, F403

NUM_ROWS = 500


def _has_shuffle(ds) -> bool:
    physical_plan, _ = get_execution_plan(ds._logical_plan)
    return any(
        isinstance(op, ShuffleMapOp) for op in physical_plan.dag.post_order_iter()
    )


@pytest.fixture
def shuffle_v2(restore_data_context):
    restore_data_context.shuffle_strategy = ShuffleStrategy.SHUFFLE_V2
    restore_data_context.use_disk_based_hash_shuffle = False
    return restore_data_context


@pytest.fixture
def hive_dir(tmp_path):
    """Four ``k=`` directories, three files each, so each key spans files."""
    rng = np.random.default_rng(0)
    for k in range(4):
        key_dir = tmp_path / f"k={k}"
        key_dir.mkdir()
        for f in range(3):
            pq.write_table(
                pa.table({"v": rng.integers(0, 100, NUM_ROWS)}),
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


def test_aggregate_skips_shuffle_when_key_grouped(
    ray_start_regular_shared_2_cpus, shuffle_v2, hive_dir, key_grouped_read
):
    shuffle_v2.enable_aggregate_shuffle_elimination = True
    ds = ray.data.read_parquet(str(hive_dir)).groupby("k").sum("v")
    assert not _has_shuffle(ds)


def test_aggregate_keeps_shuffle_without_key_grouping(
    ray_start_regular_shared_2_cpus, shuffle_v2, hive_dir
):
    shuffle_v2.enable_aggregate_shuffle_elimination = True
    ds = ray.data.read_parquet(str(hive_dir)).groupby("k").sum("v")
    assert _has_shuffle(ds)


def test_aggregate_output_matches_pandas_reference(
    ray_start_regular_shared_2_cpus, shuffle_v2, hive_dir, key_grouped_read
):
    # Each key is one block on the key-grouped path, so the one-phase aggregate
    # must produce the same sums as a plain pandas groupby over the files.
    frames = []
    for path in glob.glob(f"{hive_dir}/k=*/*.parquet"):
        key = int(path.split("k=")[1].split("/")[0])
        frame = pq.read_table(path).to_pandas()
        frame["k"] = key
        frames.append(frame)
    expected = pd.concat(frames).groupby("k", as_index=False)["v"].sum()
    # Hive partition values are read as strings.
    expected["k"] = expected["k"].astype(str)

    ds = ray.data.read_parquet(str(hive_dir)).groupby("k").sum("v")
    assert not _has_shuffle(ds)
    actual = ds.to_pandas().rename(columns={"sum(v)": "v"})
    pd.testing.assert_frame_equal(
        actual.sort_values("k").reset_index(drop=True),
        expected.sort_values("k").reset_index(drop=True),
        check_dtype=False,
    )
