import logging
from typing import List, Tuple, Union

from ray.data.block import (
    Block,
    BlockAccessor,
    BlockExecStats,
    BlockMetadata,
)
from ray.data._internal.sort import SortKey
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


def get_group_boundaries(
    keys: Union[np.ndarray, dict[str, np.ndarray]],
) -> Tuple[Union[np.ndarray, dict[str, np.ndarray]], np.ndarray]:
    """Find the boundaries of the groups.

    Args:
        keys: an array of keys or a dict of key columns

    Returns:
        grouped_keys: keys found
        indices: starting indices of each

    Note:
        This is a fast linear-time implementation that uses
        vectorization. This implementation does not relied
        the keys being sorted and unique.
        For example, [2, 2, 2, 1, 1, 3, 2] results in
        grouped_keys = [2, 1, 3, 2], indices = [0, 3, 5, 6]
    """

    if not isinstance(keys, dict):
        # Include the first index explicitly
        indices = np.hstack([[0], np.where(np.diff(keys) != 0)[0] + 1])
        return keys[indices], indices

    s = {0}
    for arr in keys.values():
        s |= set(np.where(np.diff(arr) != 0)[0] + 1)
    indices = np.array(sorted(s))

    grouped_keys = {key: value[indices] for key, value in keys.items()}
    return grouped_keys, indices


@dataclass
class BlockStat:
    keys: np.ndarray
    counts: np.ndarray
    indices: np.ndarray
    batch_id: int

    def __post_init__(self):
        if not (len(self.keys) == len(self.counts) == len(self.indices)):
            raise ValueError(
                f"length not match: ({len(self.keys)},{len(self.counts)},"
                f"{len(self.indices)}"
            )

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self) -> str:
        return f"BlockStat({self.keys}, {self.counts}, {self.indices}, {self.batch_id})"


@dataclass
class SubBlockStats:
    key: Union[int, str]
    count: int
    batch_id: list[int]  # batch_id corresponding to to_batch()
    min_batch_id: int

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self) -> str:
        return (
            f"SingleBlockStat(key={self.key}, cnt={self.count},"
            f"min_batch_id={self.min_batch_id})"
        )

    def __iadd__(self, obj: "SubBlockStats") -> "SubBlockStats":
        if self.key != obj.key:
            raise ValueError("keys must be equal")

        batch_id = sorted(self.batch_id + obj.batch_id)
        min_batch_id = batch_id[0]

        # use the obj's start_idx if it's an "earlier" block
        out = SubBlockStats(
            key=self.key,
            count=self.count + obj.count,
            batch_id=batch_id,
            min_batch_id=min_batch_id,
        )
        return out


def get_block_stats(
    keys: Union[np.ndarray, dict[str, np.ndarray]], batch_id: int
) -> BlockStat:
    """Given an array of keys of a given batch (arbitrary size), return the
    necessary statistics for merging
    """
    group_keys, indices = get_group_boundaries(keys)

    counts = np.diff(indices, append=len(keys))

    return BlockStat(group_keys, counts, indices, batch_id)


def _split_single_block_by_key(
    idx: int,
    block: Block,
    key: SortKey,
) -> Tuple[List[Block], BlockMetadata]:
    """Split the block based on the key

    Args:
        idx: block id (within one RefBundle)
        block: data
        output_num_blocks: not used as we will determine this from the key
        key: the column name for repartition
    """
    import numpy as np

    stats = BlockExecStats.builder()

    columns = key.get_columns()
    accessor = BlockAccessor.for_block(block)
    keys = accessor.to_numpy(columns)

    block_stats = get_block_stats(keys, idx)

    indices = np.hstack([block_stats["indices"], [len(keys)]])

    out_blocks = []
    for start, end in zip(indices[:-1], indices[1:]):
        out_blocks.append(accessor.slice(start, end))

    # metadata is coming the pre-split block
    meta = BlockAccessor.for_block(block).get_metadata(
        input_files=None, exec_stats=stats.build()
    )
    return out_blocks, meta


def _split_blocks_by_key(
    blocks: List[Block],
    key: SortKey,
) -> Tuple[List[List[Block]], List[BlockMetadata]]:
    """To be wrapped as a remote function"""

    subblocks_list, metadata_list = [], []
    for idx, block in enumerate(blocks):
        subblocks, metadata = _split_single_block_by_key(idx, block, key)
        subblocks_list.append(subblocks)
        metadata_list.append(metadata)

    return subblocks_list, metadata_list