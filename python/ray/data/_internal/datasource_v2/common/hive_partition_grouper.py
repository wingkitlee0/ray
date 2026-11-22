from typing import Dict, List, Optional, Tuple

import pyarrow as pa

from ray.data._internal.datasource_v2.interfaces.file_manifest import (
    PATH_COLUMN_NAME,
    FileManifest,
)
from ray.data._internal.datasource_v2.interfaces.file_partitioner import (
    FilePartitioner,
)
from ray.data.datasource.partitioning import Partitioning, PathPartitionParser


def _nulls_first(
    key: Tuple[Tuple[str, object], ...]
) -> Tuple[Tuple[str, bool, object], ...]:
    """Sort key for group keys whose values may be ``None`` (a null partition).

    ``None`` never meets a typed value: the flag decides the order, and two ``None``
    values compare equal, so the value is only compared when both are set.
    """
    return tuple((name, value is not None, value) for name, value in key)


class HivePartitionGrouper(FilePartitioner):
    """Emits one manifest per distinct hive key tuple, so each key is one read unit.

    Keys are parsed from each listing row's path. Groups are emitted in sorted
    key order and rows within a group are sorted by path, so the output does
    not depend on listing order. Listing must see every file, so this
    partitioner requires global input.

    With ``group_keys``, rows are grouped by those keys only, and a row whose
    path lacks one of them is an error. Without it, rows are grouped by every
    hive key. If no path carries a hive key, there is nothing to group by, so
    the rows go to ``fallback`` instead (when given), and this partitioner
    passes them through.
    """

    def __init__(
        self,
        partitioning: Partitioning,
        fallback: Optional[FilePartitioner] = None,
        group_keys: Optional[Tuple[str, ...]] = None,
    ):
        if group_keys is not None:
            if not group_keys:
                raise ValueError("group_keys must name at least one hive key")
            if partitioning.field_names is not None:
                unknown = set(group_keys) - set(partitioning.field_names)
                if unknown:
                    raise ValueError(
                        f"group_keys {sorted(unknown)} are not hive keys of the "
                        f"dataset ({partitioning.field_names})"
                    )
        self._parser = PathPartitionParser(partitioning)
        self._fallback = None if group_keys is not None else fallback
        self._group_keys = group_keys
        self._tables: List[pa.Table] = []
        self._groups: List[FileManifest] = []
        self._fallback_used = False

    @property
    def requires_global_input(self) -> bool:
        return True

    def add_input(self, input_manifest: FileManifest) -> None:
        block = input_manifest.as_block()
        assert isinstance(block, pa.Table)
        self._tables.append(block)

    def has_partition(self) -> bool:
        if self._fallback_used:
            return self._fallback.has_partition()
        return len(self._groups) > 0

    def next_partition(self) -> FileManifest:
        if self._fallback_used:
            return self._fallback.next_partition()
        return self._groups.pop(0)

    def finalize(self) -> None:
        if not self._tables:
            return
        table = pa.concat_tables(self._tables)
        self._tables = []

        paths = table.column(PATH_COLUMN_NAME).to_pylist()
        rows_by_key: Dict[Tuple[Tuple[str, str], ...], List[int]] = {}
        for index, path in enumerate(paths):
            key = self._key_of(str(path))
            rows_by_key.setdefault(key, []).append(index)

        if self._fallback is not None and all(len(key) == 0 for key in rows_by_key):
            self._fallback_used = True
            self._fallback.add_input(FileManifest(table))
            self._fallback.finalize()
            return

        for key in sorted(rows_by_key, key=_nulls_first):
            rows = sorted(rows_by_key[key], key=lambda i: paths[i])
            self._groups.append(FileManifest(table.take(rows)))

    def _key_of(self, path: str) -> Tuple[Tuple[str, str], ...]:
        values = self._parser(path)
        if self._group_keys is None:
            return tuple(sorted(values.items()))
        missing = [k for k in self._group_keys if k not in values]
        if missing:
            raise ValueError(
                f"Path {path!r} has no hive value for group key(s) {missing}"
            )
        return tuple((k, values[k]) for k in self._group_keys)
