import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import ray
from ray.data._internal.datasource_v2.formats.parquet.parquet_datasource_v2 import (
    ParquetDatasourceV2,
)
from ray.data.context import DataContext
from ray.data.datasource.partitioning import Partitioning
from ray.data.read_api import _read_datasource_v2

KEYS = ["a", "b", "c", "d"]
FILES_PER_KEY = 3
ROWS_PER_FILE = 5


@pytest.fixture
def hive_dataset(tmp_path):
    for key in KEYS:
        key_dir = tmp_path / f"k={key}"
        key_dir.mkdir()
        for i in range(FILES_PER_KEY):
            values = list(range(i * ROWS_PER_FILE, (i + 1) * ROWS_PER_FILE))
            pq.write_table(pa.table({"v": values}), key_dir / f"f{i}.parquet")
    return tmp_path


def test_group_keys_yields_one_block_per_key(
    ray_start_regular_shared, restore_data_context, hive_dataset
):
    # A small target size would split a group into several blocks if the
    # output stage re-shaped them.
    DataContext.get_current().target_max_block_size = 8 * 1024
    datasource = ParquetDatasourceV2(
        [str(hive_dataset)], partitioning=Partitioning("hive")
    )
    datasource._group_keys = ("k",)

    ds = _read_datasource_v2(datasource, parallelism=8)
    blocks = [ray.get(ref) for ref in ds.to_arrow_refs()]
    blocks = [block for block in blocks if block.num_rows > 0]

    keys_per_block = [set(block.column("k").to_pylist()) for block in blocks]
    assert all(len(keys) == 1 for keys in keys_per_block)
    assert sorted(keys.pop() for keys in keys_per_block) == KEYS
    assert sum(block.num_rows for block in blocks) == (
        len(KEYS) * FILES_PER_KEY * ROWS_PER_FILE
    )
