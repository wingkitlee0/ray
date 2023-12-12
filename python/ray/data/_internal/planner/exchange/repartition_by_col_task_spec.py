from typing import List, Tuple, Union

from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.planner.exchange.interfaces import ExchangeTaskSpec
from ray.data._internal.sort import SortKey
from ray.data.block import Block, BlockMetadata

logger = DatasetLogger(__name__)


class RepartitionByColTaskSpec(ExchangeTaskSpec):
    """A subclass of ExchangeTaskSpec for repartition-by-column

    See SplitRepartitionByColTaskScheduler for implementation details.
    """

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
        _logger = logger.get_logger()
        _logger.debug("Creating %s", self.__class__.__name__)

    @staticmethod
    def map(
        idx: int,
        block: Block,
        output_num_blocks: int,
        keys: SortKey,
    ) -> List[Union[BlockMetadata, Block]]:
        """Split the block based on the key

        Args:
            idx: block id (within one RefBundle)
            block: data
            output_num_blocks: not used as we will determine this from the key
            key: the column name for repartition
        """
        raise NotImplementedError

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
        raise NotImplementedError
