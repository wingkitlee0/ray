"""Tests for ``ray.data.checkpoint.partitioned``.

Mirrors the conventions of ``test_checkpoint.py`` (fixtures, ID_COL naming,
the write/fail/resume pattern), scoped to the partition-sharded manager and
filter defined in the module under test.
"""

import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import ray
from ray.data.checkpoint import CheckpointConfig
from ray.data.checkpoint.interfaces import InvalidCheckpointingConfig
from ray.data.checkpoint.partitioned import (
    PartitionedCheckpointFilter,
    PartitionedCheckpointManager,
    _extract_partition_shards,
    _flatten_struct_id_column,
    _sort_partition_group,
)
from ray.data.context import DataContext
from ray.data.tests.conftest import *  # noqa

ID_COL = "row_hash"
HASH_FIELD = PartitionedCheckpointManager.HASH_FIELD_NAME
PARTITION_FIELD = PartitionedCheckpointManager.PARTITION_FIELD_NAME

pytestmark = [
    pytest.mark.usefixtures("restore_data_context"),
    pytest.mark.timeout(300),
]


def _struct_id_column(hashes, partitions) -> pa.StructArray:
    return pa.StructArray.from_arrays(
        [
            pa.array(hashes, type=pa.uint64()),
            pa.array(partitions, type=pa.uint32()),
        ],
        names=[HASH_FIELD, PARTITION_FIELD],
    )


def _checkpoint_config(checkpoint_path) -> CheckpointConfig:
    return CheckpointConfig(
        id_column=ID_COL,
        checkpoint_path=str(checkpoint_path),
        checkpoint_manager_cls=PartitionedCheckpointManager,
        checkpoint_filter_cls=PartitionedCheckpointFilter,
    )


# --------------------------------------------------------------------------
# Pure-function unit tests: no Ray cluster required.
# --------------------------------------------------------------------------


def test_flatten_struct_id_column():
    batch = pa.table({ID_COL: _struct_id_column([1, 2, 3], [10, 10, 20])})
    flat = _flatten_struct_id_column(
        batch, id_column=ID_COL, hash_field=HASH_FIELD, partition_field=PARTITION_FIELD
    )
    assert flat["__ckpt_hash__"].to_pylist() == [1, 2, 3]
    assert flat["__ckpt_partition__"].to_pylist() == [10, 10, 20]


def test_sort_partition_group():
    from ray.data.checkpoint.partitioned import _HASH_COL, _PARTITION_COL

    batch = pa.table(
        {
            _PARTITION_COL: pa.array([7, 7, 7], type=pa.uint32()),
            _HASH_COL: pa.array([30, 10, 20], type=pa.uint64()),
        }
    )
    result = _sort_partition_group(batch)
    assert result.num_rows == 1
    assert result["__ckpt_partition__"][0].as_py() == 7
    sorted_hashes = result["__ckpt_sorted_hashes__"][0].values.to_numpy(
        zero_copy_only=False
    )
    assert list(sorted_hashes) == [10, 20, 30]


def test_extract_partition_shards(ray_start_10_cpus_shared):
    from ray.data.checkpoint.partitioned import _HASH_COL, _PARTITION_COL

    values_a = pa.array(np.array([10, 20, 30], dtype=np.uint64))
    values_b = pa.array(np.array([1, 2], dtype=np.uint64))
    sorted_col = pa.ListArray.from_arrays(
        pa.array([0, 3, 5], type=pa.int32()), pa.concat_arrays([values_a, values_b])
    )
    block = pa.table(
        {
            _PARTITION_COL: pa.array([7, 8], type=pa.uint32()),
            "__ckpt_sorted_hashes__": sorted_col,
        }
    )
    shard_map = ray.get(_extract_partition_shards.remote(block))
    assert set(shard_map.keys()) == {7, 8}
    shard_7_ref, shard_7_size = shard_map[7]
    shard_8_ref, shard_8_size = shard_map[8]
    np.testing.assert_array_equal(ray.get(shard_7_ref), [10, 20, 30])
    np.testing.assert_array_equal(ray.get(shard_8_ref), [1, 2])
    assert shard_7_size == 3 * 8
    assert shard_8_size == 2 * 8


# --------------------------------------------------------------------------
# Filter unit tests: construct a fake shard map directly, no manager/write
# path involved. This is the test that specifically exercises cross-partition
# isolation -- the core correctness property this design relies on.
# --------------------------------------------------------------------------


def test_filter_cross_partition_isolation(ray_start_10_cpus_shared, tmp_path):
    """The same hash value in two different partitions must be treated
    independently: checkpointing it in one partition must not filter it out
    of the other."""
    config = _checkpoint_config(tmp_path / "ckpt")

    # hash value 5 is checkpointed in partition 1, but NOT in partition 2.
    shard_map = {
        1: (ray.put(np.array([5, 6], dtype=np.uint64)), 16),
        2: (ray.put(np.array([], dtype=np.uint64)), 0),
    }
    checkpoint_ref = ray.put(shard_map)
    filt = PartitionedCheckpointFilter(config, checkpoint_ref)

    block = pa.table(
        {
            ID_COL: _struct_id_column([5, 5, 7], [1, 2, 1]),
            "value": ["a", "b", "c"],
        }
    )
    result = filt.filter_rows_for_block(block)

    # Row 0 (hash=5, partition=1): checkpointed -> filtered out.
    # Row 1 (hash=5, partition=2): NOT checkpointed in partition 2 -> kept.
    # Row 2 (hash=7, partition=1): not checkpointed -> kept.
    assert result["value"].to_pylist() == ["b", "c"]


def test_filter_unseen_partition_keeps_all_rows(ray_start_10_cpus_shared, tmp_path):
    """A partition with no entry in the shard map (never checkpointed yet)
    keeps every row."""
    config = _checkpoint_config(tmp_path / "ckpt")
    checkpoint_ref = ray.put({})
    filt = PartitionedCheckpointFilter(config, checkpoint_ref)

    block = pa.table({ID_COL: _struct_id_column([1, 2], [99, 99]), "value": ["a", "b"]})
    result = filt.filter_rows_for_block(block)
    assert result["value"].to_pylist() == ["a", "b"]


def test_filter_empty_block(tmp_path):
    config = _checkpoint_config(tmp_path / "ckpt")
    filt = PartitionedCheckpointFilter(config, None)
    block = pa.table({ID_COL: _struct_id_column([], []), "value": pa.array([], type=pa.string())})
    result = filt.filter_rows_for_block(block)
    assert result.num_rows == 0


# --------------------------------------------------------------------------
# Manager unit tests.
# --------------------------------------------------------------------------


def test_manager_no_checkpoint_dir(ray_start_10_cpus_shared, tmp_path):
    ctx = DataContext.get_current()
    config = _checkpoint_config(tmp_path / "does_not_exist")
    manager = PartitionedCheckpointManager(config, ctx)
    ref, size = manager.load_checkpoint()
    assert ref is None
    assert size == 0


def test_manager_validates_non_struct_id_column(ray_start_10_cpus_shared, tmp_path):
    ckpt_path = tmp_path / "ckpt"
    ckpt_path.mkdir()
    # A plain (non-struct) id column, as the default manager would expect.
    pa.parquet.write_table(
        pa.table({ID_COL: pa.array([1, 2, 3], type=pa.uint64())}),
        str(ckpt_path / "pre_checkpoint.parquet"),
    )
    ctx = DataContext.get_current()
    config = _checkpoint_config(ckpt_path)
    manager = PartitionedCheckpointManager(config, ctx)
    with pytest.raises(InvalidCheckpointingConfig, match="struct column"):
        manager.load_checkpoint()


def test_manager_validates_missing_struct_field(ray_start_10_cpus_shared, tmp_path):
    ckpt_path = tmp_path / "ckpt"
    ckpt_path.mkdir()
    bad_struct = pa.StructArray.from_arrays(
        [pa.array([1, 2], type=pa.uint64())], names=["hash"]
    )  # missing "partition" field
    pa.parquet.write_table(
        pa.table({ID_COL: bad_struct}),
        str(ckpt_path / "pre_checkpoint.parquet"),
    )
    ctx = DataContext.get_current()
    config = _checkpoint_config(ckpt_path)
    manager = PartitionedCheckpointManager(config, ctx)
    with pytest.raises(InvalidCheckpointingConfig, match="partition"):
        manager.load_checkpoint()


def test_manager_builds_shard_per_partition(ray_start_10_cpus_shared, tmp_path):
    ckpt_path = tmp_path / "ckpt"
    ckpt_path.mkdir()
    pa.parquet.write_table(
        pa.table({ID_COL: _struct_id_column([3, 1, 2, 20, 10], [0, 0, 0, 1, 1])}),
        str(ckpt_path / "pre_checkpoint.parquet"),
    )
    ctx = DataContext.get_current()
    config = _checkpoint_config(ckpt_path)
    manager = PartitionedCheckpointManager(config, ctx)
    ref, size = manager.load_checkpoint()
    assert ref is not None

    shard_map = ray.get(ref)
    assert set(shard_map.keys()) == {0, 1}
    shard_0 = ray.get(shard_map[0][0])
    shard_1 = ray.get(shard_map[1][0])
    np.testing.assert_array_equal(shard_0, [1, 2, 3])  # sorted
    np.testing.assert_array_equal(shard_1, [10, 20])  # sorted
    assert size == max(shard_map[0][1], shard_map[1][1])


# --------------------------------------------------------------------------
# End-to-end: real write + injected failure + resume, verifying no
# duplicates and no cross-partition leakage, mirroring
# test_checkpoint.py::test_partial_failure_no_duplicates.
# --------------------------------------------------------------------------


def test_partitioned_checkpoint_partial_failure_no_duplicates(
    ray_start_10_cpus_shared, tmp_path
):
    num_rows = 1000
    num_partitions = 8
    fail_threshold = 100

    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    checkpoint_path_dir = tmp_path / "checkpoint"
    for path in [input_path, output_path, checkpoint_path_dir]:
        path.mkdir(exist_ok=True)

    df = pd.DataFrame({"key": range(num_rows), "value": [f"row_{i}" for i in range(num_rows)]})
    df.to_parquet(input_path / "data.parquet", index=False)

    ctx = DataContext.get_current()
    ctx.checkpoint_config = _checkpoint_config(checkpoint_path_dir)
    ctx.checkpoint_config.delete_checkpoint_on_success = False

    def add_struct_id(batch):
        keys = batch["key"].to_numpy()
        batch[ID_COL] = _struct_id_column(
            keys.astype(np.uint64), (keys % num_partitions).astype(np.uint32)
        )
        return batch

    from ray.data.tests.test_checkpoint import FailAfterWriteParquetDatasink

    with pytest.raises(RuntimeError, match="Simulated failure"):
        ds = ray.data.read_parquet(str(input_path))
        ds = ds.repartition(50)
        ds = ds.map_batches(add_struct_id, batch_format="pyarrow", batch_size=None)
        failing_datasink = FailAfterWriteParquetDatasink(
            str(output_path), fail_threshold=fail_threshold
        )
        ds.write_datasink(failing_datasink, ray_remote_args={"max_retries": 0})

    ray.shutdown()
    ray.init()
    ctx = DataContext.get_current()
    ctx.checkpoint_config = _checkpoint_config(checkpoint_path_dir)

    ds2 = ray.data.read_parquet(str(input_path))
    ds2 = ds2.repartition(50)
    ds2 = ds2.map_batches(add_struct_id, batch_format="pyarrow", batch_size=None)
    ds2.write_parquet(str(output_path))

    ctx.checkpoint_config = None
    result = ray.data.read_parquet(str(output_path)).to_pandas()

    assert len(result) == num_rows
    assert result["key"].is_unique, (
        "Duplicate keys found: "
        f"{sorted(result[result.duplicated('key', keep=False)]['key'].unique().tolist())}"
    )
    assert sorted(result["key"].tolist()) == list(range(num_rows))
