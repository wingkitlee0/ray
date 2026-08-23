"""Partition-scoped checkpoint manager and filter.

The default checkpointing path (:class:`~ray.data.checkpoint.IdColumnCheckpointManager`
/ :class:`~ray.data.checkpoint.NumpyArrayBasedCheckpointFilter`) coalesces every
checkpointed ID into a single sorted NumPy array, held in full by every
checkpoint filter actor. For a dataset with a coarse partition column and O(10B)
rows, that array (and the per-actor memory reservation derived from its size)
does not scale: see https://github.com/ray-project/ray/issues/60200 and
https://github.com/ray-project/ray/issues/61509.

This module implements a partition-scoped alternative: :class:`id_column` is a
*struct* column with two fields (by default named ``"hash"`` and
``"partition"``), computed upstream of the write operator. Checkpoint files are
written unchanged by the existing write path -- the struct is just an ordinary
Arrow column as far as the writer is concerned -- but on restore,
:class:`PartitionedCheckpointManager` groups checkpointed IDs by the
``partition`` field and sorts each partition's ``hash`` values independently,
producing one small shard per partition instead of one array for the whole
dataset. :class:`PartitionedCheckpointFilter` fetches only the shard(s) a
block's rows actually belong to.

Only rows within the same partition value are ever compared against each
other, so uniqueness of the ``hash`` field only needs to hold *within* a
partition, not across the whole dataset -- see the ``partition`` field's
docstring on :class:`PartitionedCheckpointManager` for the collision-safety
argument this relies on.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow
import pyarrow.compute as pc

import ray
from ray.data._internal.arrow_ops import transform_pyarrow
from ray.data._internal.execution.interfaces.ref_bundle import RefBundle
from ray.data.block import Block
from ray.data.checkpoint.checkpoint_filter import CheckpointFilter, CheckpointManager
from ray.data.checkpoint.interfaces import CheckpointConfig, InvalidCheckpointingConfig
from ray.types import ObjectRef
from ray.util.annotations import DeveloperAPI

logger = logging.getLogger(__name__)

# Internal column names used only within the shard-building pipeline below;
# never user-visible and never written to any checkpoint file.
_PARTITION_COL = "__ckpt_partition__"
_HASH_COL = "__ckpt_hash__"
_SORTED_HASHES_COL = "__ckpt_sorted_hashes__"

# Type alias for the (opaque, to the framework) object this module's manager
# returns and its filter consumes: a map from partition value to (a ref to
# that partition's sorted hash array, the array's size in bytes).
ShardMap = Dict[Any, Tuple[ObjectRef[np.ndarray], int]]


def _flatten_struct_id_column(
    batch: "pyarrow.Table",
    id_column: str,
    hash_field: str,
    partition_field: str,
) -> "pyarrow.Table":
    """Project the struct ``id_column`` into two flat top-level columns.

    ``Dataset.groupby`` groups by a top-level column name, not a nested
    struct field expression, so this is a required step before grouping.
    """
    struct_col = batch[id_column].combine_chunks()
    return pyarrow.table(
        {
            _PARTITION_COL: struct_col.field(partition_field),
            _HASH_COL: struct_col.field(hash_field),
        }
    )


def _sort_partition_group(batch: "pyarrow.Table") -> "pyarrow.Table":
    """Sort one partition's hash values, returning a single-row table whose
    ``_SORTED_HASHES_COL`` cell holds the entire sorted array as a list value.

    ``batch`` holds every row for exactly one ``_PARTITION_COL`` value (that
    is what ``GroupedData.map_groups`` guarantees), so it is safe to read the
    partition value from the first row.
    """
    partition_value = batch[_PARTITION_COL][0].as_py()
    hashes = transform_pyarrow.to_numpy(batch[_HASH_COL], zero_copy_only=False)
    sorted_hashes = np.sort(hashes)

    # Build the list-typed cell via from_arrays (zero-copy from the numpy
    # array) rather than a Python list round-trip, since a single partition's
    # array can be tens of millions of elements.
    values = pyarrow.array(sorted_hashes)
    offsets = pyarrow.array([0, len(sorted_hashes)], type=pyarrow.int32())
    sorted_col = pyarrow.ListArray.from_arrays(offsets, values)

    return pyarrow.table(
        {
            _PARTITION_COL: pyarrow.array([partition_value]),
            _SORTED_HASHES_COL: sorted_col,
        }
    )


@ray.remote
def _extract_partition_shards(block: "pyarrow.Table") -> ShardMap:
    """Split one block of (partition, sorted-hash-array) rows into
    independently-``ray.put``-able shards.

    Runs as a remote task per block of the grouped dataset -- not on the
    driver -- so no single process ever holds more than one block's worth of
    shards (bounded by Ray Data's normal target block size) at a time.
    """
    partition_array = block[_PARTITION_COL]
    sorted_list_array = block[_SORTED_HASHES_COL]
    shard_map: ShardMap = {}
    for i in range(len(block)):
        partition_value = partition_array[i].as_py()
        sorted_scalar = sorted_list_array[i]
        hashes = sorted_scalar.values.to_numpy(zero_copy_only=False)
        if hashes.dtype != np.uint64:
            hashes = hashes.astype(np.uint64, copy=False)
        shard_map[partition_value] = (ray.put(hashes), hashes.nbytes)
    return shard_map


@DeveloperAPI
class PartitionedCheckpointManager(CheckpointManager):
    """``CheckpointManager`` that shards checkpointed IDs by a partition field.

    Requires ``checkpoint_config.id_column`` to be a **struct** column with
    two fields: ``HASH_FIELD_NAME`` (default ``"hash"``, must be unique
    *within* each partition value -- see below) and ``PARTITION_FIELD_NAME``
    (default ``"partition"``, the value blocks are sharded by at restore
    time). Both are class attributes so a subclass can rename them without
    overriding any methods; :class:`PartitionedCheckpointFilter` must be
    subclassed to match if you do.

    Collision safety: because only rows sharing a partition value are ever
    compared, ``hash`` values only need to be unique within a partition, not
    across the whole dataset. A content hash of width ``h`` bits over a
    partition of ``n`` rows has expected colliding pairs
    ``n^2 / 2^(h+1)`` -- for ``P`` roughly equal partitions of a dataset with
    ``N`` total rows (``n = N / P``), the *dataset-wide* expected collisions
    are ``N^2 / (P * 2^(h+1))``: a factor of ``P`` improvement over treating
    the same hash as one flat, dataset-wide identity, because collisions
    between rows in different partitions are structurally impossible to ever
    be compared, not merely rare.

    Construction cost: building the shard map requires one distributed
    ``groupby`` + per-group sort over the checkpoint directory's contents (not
    a single-node coalesce), and one small extraction task per output block.
    This replaces the default manager's ``repartition(num_blocks=1)`` +
    single-task ``np.sort``, which cannot scale to a whole dataset's worth of
    checkpointed IDs. See the module docstring for the scalability problem
    this solves.

    Example::

        import pyarrow.compute as pc
        from ray.data.expressions import col
        from ray.data.checkpoint import CheckpointConfig
        from ray.data.checkpoint.partitioned import (
            PartitionedCheckpointManager,
            PartitionedCheckpointFilter,
        )

        ds = ds.with_column(
            "row_hash",
            # Any expression producing a struct with "hash" and "partition"
            # fields; both must be present in every row reaching the write
            # operator, and must flow through unchanged like any id_column.
            some_struct_expr,
        )
        config = CheckpointConfig(
            id_column="row_hash",
            checkpoint_path="s3://bucket/checkpoints",
            checkpoint_manager_cls=PartitionedCheckpointManager,
            checkpoint_filter_cls=PartitionedCheckpointFilter,
        )
    """

    HASH_FIELD_NAME: str = "hash"
    PARTITION_FIELD_NAME: str = "partition"

    def load_checkpoint(
        self,
        data_file_dir: Optional[str] = None,
        data_file_filesystem: Optional["pyarrow.fs.FileSystem"] = None,
    ) -> Tuple[Optional[ObjectRef[ShardMap]], int]:
        """Load checkpoint data as a map of per-partition shards.

        Returns:
            ObjectRef[ShardMap]: a ref to a ``{partition_value: (shard_ref,
                shard_size_bytes)}`` map. ``None`` if no checkpoint was loaded.
            int: the size, in bytes, of the *largest single shard* -- this
                sizes the per-actor memory reservation, since a filter actor
                whose blocks stay within one partition never holds more than
                one shard at a time. If your pipeline routes blocks that mix
                many partitions to the same actor, override this to size for
                the actor's actual expected working set instead.
        """
        logger.info(
            "Loading partitioned checkpoint from %s, this could take a while.",
            self.checkpoint_path,
        )
        start_t = time.time()

        if data_file_dir is not None:
            self._clean_pending_checkpoints(data_file_dir, data_file_filesystem)

        from pyarrow.fs import FileSelector, FileType

        entries = self.filesystem.get_file_info(
            FileSelector(
                self.checkpoint_path_unwrapped,
                recursive=self.checkpoint_path_partition_filter is not None,
                allow_not_found=True,
            )
        )
        if not any(f.type == FileType.File for f in entries):
            return None, 0

        checkpoint_ds: ray.data.Dataset = ray.data.read_parquet(
            self.checkpoint_path,
            filesystem=self.filesystem,
            partition_filter=self.checkpoint_path_partition_filter,
        )
        checkpoint_ds.set_name("partitioned_checkpoint_dataset")
        # Manually disable checkpointing for loading the checkpoint metadata,
        # to avoid recursively restoring checkpoints (matches the base class).
        checkpoint_ds.context.checkpoint_config = None

        checkpoint_ds = self._preprocess_data_pipeline(checkpoint_ds)
        self._validate_id_column_schema(checkpoint_ds.schema().base_schema)

        flat_ds = checkpoint_ds.map_batches(
            _flatten_struct_id_column,
            batch_format="pyarrow",
            fn_kwargs={
                "id_column": self.id_column,
                "hash_field": self.HASH_FIELD_NAME,
                "partition_field": self.PARTITION_FIELD_NAME,
            },
        )
        grouped_ds = flat_ds.groupby(_PARTITION_COL).map_groups(
            _sort_partition_group,
            batch_format="pyarrow",
        )

        ref_bundles: List[RefBundle] = list(grouped_ds.iter_internal_ref_bundles())
        if not ref_bundles or all(rb.num_rows() == 0 for rb in ref_bundles):
            return None, 0

        extract_refs = [
            _extract_partition_shards.remote(block.ref)
            for rb in ref_bundles
            for block in rb.blocks
        ]
        shard_map: ShardMap = {}
        for partial_map in ray.get(extract_refs):
            shard_map.update(partial_map)

        if not shard_map:
            return None, 0

        shard_map_ref = ray.put(shard_map)
        largest_shard_bytes = max(nbytes for _, nbytes in shard_map.values())

        logger.info(
            "Partitioned checkpoint loaded in %.2f seconds. %d partitions, "
            "largest shard = %d bytes.",
            time.time() - start_t,
            len(shard_map),
            largest_shard_bytes,
        )
        return shard_map_ref, largest_shard_bytes

    def _validate_id_column_schema(self, arrow_schema: "pyarrow.Schema") -> None:
        """Validate that ``id_column`` is a struct with the expected fields.

        Args:
            arrow_schema: The checkpoint dataset's underlying Arrow schema
                (i.e. ``Dataset.schema().base_schema``).

        Raises:
            InvalidCheckpointingConfig: if the column is missing or not a
                struct containing both ``HASH_FIELD_NAME`` and
                ``PARTITION_FIELD_NAME``.
        """
        try:
            field = arrow_schema.field(self.id_column)
        except KeyError:
            raise InvalidCheckpointingConfig(
                f"Checkpoint id_column {self.id_column!r} not found in "
                f"checkpoint schema {arrow_schema}."
            )
        if not pyarrow.types.is_struct(field.type):
            raise InvalidCheckpointingConfig(
                f"{type(self).__name__} requires id_column {self.id_column!r} "
                f"to be a struct column with fields {self.HASH_FIELD_NAME!r} "
                f"and {self.PARTITION_FIELD_NAME!r}, but got type {field.type}."
            )
        field_names = {field.type.field(i).name for i in range(field.type.num_fields)}
        missing = {self.HASH_FIELD_NAME, self.PARTITION_FIELD_NAME} - field_names
        if missing:
            raise InvalidCheckpointingConfig(
                f"{type(self).__name__} requires id_column {self.id_column!r} "
                f"struct to have fields {missing}, but it only has "
                f"{field_names}."
            )


@DeveloperAPI
class PartitionedCheckpointFilter(CheckpointFilter):
    """``CheckpointFilter`` that filters each block against only the
    per-partition shard(s) its rows actually belong to.

    Must be paired with :class:`PartitionedCheckpointManager` (or a subclass
    using the same ``HASH_FIELD_NAME`` / ``PARTITION_FIELD_NAME``), since it
    expects ``checkpoint_ref`` to resolve to a ``ShardMap`` as produced by
    that manager's ``load_checkpoint``.

    Shards are fetched lazily and cached per actor: a block only pays for
    ``ray.get`` on the partition(s) it actually contains, not the whole
    dataset's checkpoint state.
    """

    HASH_FIELD_NAME: str = "hash"
    PARTITION_FIELD_NAME: str = "partition"

    def __init__(
        self,
        checkpoint_config: CheckpointConfig,
        checkpoint_ref: Optional[ObjectRef[ShardMap]] = None,
    ):
        super().__init__(checkpoint_config, checkpoint_ref)
        self._shard_map: ShardMap = ray.get(checkpoint_ref) if checkpoint_ref else {}
        self._shard_cache: Dict[Any, np.ndarray] = {}

    def _get_shard(self, partition_value: Any) -> np.ndarray:
        if partition_value in self._shard_cache:
            return self._shard_cache[partition_value]
        entry = self._shard_map.get(partition_value)
        if entry is None:
            # No checkpointed rows for this partition yet -- nothing to
            # filter out. Cache the empty result so repeated blocks from an
            # un-checkpointed partition don't re-check the map every time.
            shard = np.empty(0, dtype=np.uint64)
        else:
            shard_ref, _ = entry
            shard = ray.get(shard_ref)
        self._shard_cache[partition_value] = shard
        return shard

    def filter_rows_for_block(self, block: Block) -> Block:
        if len(block) == 0:
            return block

        assert isinstance(block, pyarrow.Table)
        struct_col = block[self.id_column].combine_chunks()
        partitions = struct_col.field(self.PARTITION_FIELD_NAME)
        hashes = transform_pyarrow.to_numpy(
            struct_col.field(self.HASH_FIELD_NAME), zero_copy_only=False
        )

        mask = np.ones(len(block), dtype=bool)
        unique_partitions = pc.unique(partitions)
        for partition_scalar in unique_partitions:
            partition_value = partition_scalar.as_py()
            group_mask = pc.equal(partitions, partition_scalar).to_numpy(
                zero_copy_only=False
            )
            shard = self._get_shard(partition_value)
            if shard.shape[0] == 0:
                continue  # keep all rows for this partition

            group_hashes = hashes[group_mask]
            sorted_indices = np.searchsorted(shard, group_hashes)
            valid = sorted_indices < len(shard)
            potential_matches = sorted_indices[valid]
            matched = shard[potential_matches] == group_hashes[valid]

            group_result_mask = np.ones(len(group_hashes), dtype=bool)
            group_result_mask[valid] = ~matched
            mask[group_mask] = group_result_mask

        mask_array = pyarrow.array(mask)
        return block.filter(mask_array)
