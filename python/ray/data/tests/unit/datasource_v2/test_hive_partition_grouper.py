import pytest

from ray.data._internal.datasource_v2.common.hive_partition_grouper import (
    HivePartitionGrouper,
)
from ray.data._internal.datasource_v2.interfaces.file_manifest import FileManifest
from ray.data.datasource.partitioning import Partitioning


def _manifest(paths):
    return FileManifest.construct_manifest(
        paths=paths,
        sizes=[1] * len(paths),
        chunk_metadatas=[None] * len(paths),
    )


def _groups(grouper):
    grouper.finalize()
    groups = []
    while grouper.has_partition():
        groups.append(sorted(str(p) for p in grouper.next_partition().paths))
    return groups


def test_group_keys_groups_only_by_the_named_keys():
    grouper = HivePartitionGrouper(
        Partitioning("hive"),
        group_keys=("k",),
    )
    grouper.add_input(
        _manifest(["k=a/d=1/f0.parquet", "k=a/d=2/f1.parquet", "k=b/d=1/f2.parquet"])
    )

    assert _groups(grouper) == [
        ["k=a/d=1/f0.parquet", "k=a/d=2/f1.parquet"],
        ["k=b/d=1/f2.parquet"],
    ]


def test_without_group_keys_every_hive_key_splits_groups():
    grouper = HivePartitionGrouper(Partitioning("hive"))
    grouper.add_input(_manifest(["k=a/d=1/f0.parquet", "k=a/d=2/f1.parquet"]))

    assert _groups(grouper) == [
        ["k=a/d=1/f0.parquet"],
        ["k=a/d=2/f1.parquet"],
    ]


def test_path_missing_a_group_key_is_an_error():
    grouper = HivePartitionGrouper(Partitioning("hive"), group_keys=("d",))
    grouper.add_input(_manifest(["k=a/d=1/f0.parquet", "k=b/f1.parquet"]))

    with pytest.raises(ValueError, match="no hive value for group key"):
        grouper.finalize()


def test_group_key_that_is_not_a_hive_field_is_rejected_up_front():
    with pytest.raises(ValueError, match="not hive keys of the dataset"):
        HivePartitionGrouper(
            Partitioning("hive", field_names=["k", "d"]),
            group_keys=("x",),
        )


def test_null_partition_sorts_before_values_instead_of_failing():
    grouper = HivePartitionGrouper(
        Partitioning("hive", null_fallback="__NULL__"),
        group_keys=("k",),
    )
    grouper.add_input(
        _manifest(["k=a/f0.parquet", "k=__NULL__/f1.parquet", "k=b/f2.parquet"])
    )

    assert _groups(grouper) == [
        ["k=__NULL__/f1.parquet"],
        ["k=a/f0.parquet"],
        ["k=b/f2.parquet"],
    ]
