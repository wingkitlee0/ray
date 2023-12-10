from dataclasses import dataclass
from typing import List, Tuple, TypeVar, Union

import numpy as np

from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.planner.exchange.interfaces import ExchangeTaskSpec
from ray.data._internal.sort import SortKey
from ray.data.block import Block, BlockAccessor, BlockExecStats, BlockMetadata

T = TypeVar("T")

logger = DatasetLogger(__name__)


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


class RepartitionByColTaskSpec(ExchangeTaskSpec):
    """Implementation of repartition-by-column

    Repartition-by-column is used to ensure the group boundaries
    are aligned with the block boundaries. This means that each
    group is fully contained in N blocks. We assume the keys are
    continuous within each block before repartition. This is not
    intended to be a general groupby ops.

    1. Map-stage: each block is split into blocks with a single
    group. For example, a dataset with N input blocks and K groups,
    the output of the map stage will be between max(N, K) and N * K,
    depending on how many splits per block.

    2. Merge-stage: Given a list of mapped-blocks, we merge them
    if they belong to the same group. The output of this stage is
    a list of blocks with a single group.

    Note:
        By default, We do not explicitly sort the blocks.
    """

    # TODO: Preserve the ordering of the input blocks. A naive implementation
    # is to add a single-value column to label the ordering
    # TODO: Allow control of target_max_block_size
    # For target_max_block_size, it could be implemented in both map
    # and reduce stages.
    # TODO: Allow setting output_num_blocks. When it is smaller than the
    # number of groups, each output block may contain more than one group
    # On the other hand, if output_num_blocks is large, one group may be
    # split into multiple blocks. User should have control to this behavior
    # TODO: Allow sorting within the block (based on a column other than the
    # partition key). See Spark RDD's repartitionAndSortWithinPartitions

    REPARTITION_BY_COLUMN_SPLIT_SUB_PROGRESS_BAR_NAME = "Repartition Split"
    REPARTITION_BY_COLUMN_MERGE_SUB_PROGRESS_BAR_NAME = "Repartition Merge"

    def __init__(
        self,
        keys: Union[str, List[str]],
    ):
        super().__init__(
            map_args=[keys],
            reduce_args=[keys],
        )
        print("Creating", self.__class__.__name__)

    @staticmethod
    def map(
        idx: int,
        block: Block,
        output_num_blocks: int,
        key: SortKey,
    ) -> List[Union[BlockMetadata, Block]]:
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

        out = []
        for start, end in zip(indices[:-1], indices[1:]):
            out.append(accessor.slice(start, end))

        # metadata is coming the pre-split block
        meta = BlockAccessor.for_block(block).get_metadata(
            input_files=None, exec_stats=stats.build()
        )
        return out + [meta]

    @staticmethod
    def reduce(
        key: SortKey,
        *mapper_outputs: List[Block],
        partial_reduce: bool = False,
    ) -> Tuple[Block, BlockMetadata]:
        """Reduce stage by merging blocks with the same key

        Args:
            key: column for repartition
            *mapper_outputs: list of blocks
            partial_reduce: When False, this is final reduce stage.
        """
        # Since the original output signature is a tuple of a single
        # block and its metadata, this implies that this function should
        # be called with a list of blocks with the same key.

        # if partial_reduce:

        # return BlockAccessor.for_block(mapper_outputs[0]).merge_sorted_blocks(
        #     mapper_outputs, key
        # )
