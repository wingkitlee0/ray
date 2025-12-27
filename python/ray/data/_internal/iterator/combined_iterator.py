from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple

from ray.data._internal.execution.interfaces.ref_bundle import RefBundle
from ray.data._internal.stats import DatasetStats
from ray.data.context import DataContext
from ray.data.iterator import DataIterator

if TYPE_CHECKING:

    from ray.data.dataset import Dataset, Schema


class CombinedDataIterator(DataIterator):
    """A DataIterator that cycles through multiple Dataset instances.

    This class generalizes all APIs of DataIterator by cycling through multiple
    Dataset instances. Each call to an iteration method (e.g., iter_batches(),
    iter_rows()) exhausts one dataset, then advances to the next dataset for
    subsequent calls.

    Examples:
        >>> import ray
        >>> ds1 = ray.data.range(5)
        >>> ds2 = ray.data.range(3)
        >>> combined = CombinedDataIterator([ds1, ds2], cycle=True)
        >>> # First call exhausts ds1
        >>> list(combined.iter_rows())
        [{'id': 0}, {'id': 1}, {'id': 2}, {'id': 3}, {'id': 4}]
        >>> # Second call exhausts ds2
        >>> list(combined.iter_rows())
        [{'id': 0}, {'id': 1}, {'id': 2}]
        >>> # With cycle=False, stops after one pass through all datasets
        >>> combined = CombinedDataIterator([ds1, ds2], cycle=False)
        >>> list(combined.iter_rows())  # Exhausts ds1
        [{'id': 0}, {'id': 1}, {'id': 2}, {'id': 3}, {'id': 4}]
        >>> list(combined.iter_rows())  # Exhausts ds2
        [{'id': 0}, {'id': 1}, {'id': 2}]
        >>> list(combined.iter_rows())  # Raises StopIteration
        []
    """

    def __init__(
        self,
        datasets: List["Dataset"],
        validate_schema: bool = False,
        cycle: bool = True,
    ):
        """Initialize CombinedDataIterator with a list of Dataset instances.

        Args:
            datasets: List of Dataset instances to combine.
            validate_schema: If True, validate that all datasets have the same schema.
                Raises ValueError if schemas don't match. Defaults to False.
            cycle: If True, cycle through datasets indefinitely (wrapping around).
                If False, stop after exhausting all datasets once. Defaults to True.
        """
        if not datasets:
            raise ValueError("At least one dataset is required")

        # Validate schemas if requested
        if validate_schema:
            schemas = [ds.schema() for ds in datasets]
            # Filter out None schemas for validation
            valid_schemas = [s for s in schemas if s is not None]
            if valid_schemas:
                self._validate_schemas(valid_schemas)

        # Convert all datasets to iterators
        self._iterators = [ds.iterator() for ds in datasets]
        self._datasets = datasets
        self._dataset_idx = 0
        self._cycle = cycle
        self._has_completed_cycle = False

    def _validate_schemas(self, schemas: List["Schema"]) -> None:
        """Validate that all schemas are the same.

        Args:
            schemas: List of schemas to validate.

        Raises:
            ValueError: If schemas don't match.
        """
        if not schemas:
            return

        reference_schema = schemas[0]
        mismatches = []

        for i, schema in enumerate(schemas[1:], start=1):
            if reference_schema != schema:
                mismatches.append((i, schema))

        if mismatches:
            error_msg = "Schemas do not match across datasets.\n"
            error_msg += f"Dataset 0 schema: {reference_schema}\n"
            for idx, schema in mismatches:
                error_msg += f"Dataset {idx} schema: {schema}\n"
            raise ValueError(error_msg)

    def __repr__(self) -> str:
        dataset_reprs = [repr(ds) for ds in self._datasets]
        return f"CombinedDataIterator({', '.join(dataset_reprs)})"

    def _to_ref_bundle_iterator(
        self,
    ) -> Tuple[Iterator[RefBundle], Optional[DatasetStats], bool]:
        """Returns an iterator for the current dataset, advancing to next on exhaustion.

        Returns:
            A tuple containing:
            - An iterator over RefBundles from the current dataset
            - DatasetStats from the current dataset
            - Boolean indicating if blocks can be safely cleared after use

        Raises:
            StopIteration: If cycle=False and all datasets have been exhausted once.
        """
        # If cycle=False and completed, return an empty iterator
        # that raises StopIteration immediately
        if not self._cycle and self._has_completed_cycle and self._dataset_idx == 0:

            def empty_iterator() -> Iterator[RefBundle]:
                yield from ()

            empty_gen = empty_iterator()
            return empty_gen, None, False

        # Get iterator for current dataset
        (
            ref_bundles_iter,
            stats,
            blocks_owned,
        ) = self._iterators[self._dataset_idx]._to_ref_bundle_iterator()

        def wrapped_iterator() -> Iterator[RefBundle]:
            """Advance dataset_idx when exhausted"""
            try:
                yield from ref_bundles_iter
            finally:
                # Advance to next dataset when exhausted
                next_idx = (self._dataset_idx + 1) % len(self._iterators)
                # Check if we've wrapped around (completed a cycle)
                if next_idx == 0:
                    self._has_completed_cycle = True
                self._dataset_idx = next_idx

        return wrapped_iterator(), stats, blocks_owned

    def stats(self) -> str:
        """Returns execution timing information from the current dataset.

        Returns:
            A string containing stats from the current dataset.
        """
        return self._iterators[self._dataset_idx].stats()

    def schema(self) -> "Schema":
        """Return the schema of the current dataset.

        Returns:
            The schema from the current dataset.
        """
        return self._iterators[self._dataset_idx].schema()

    def get_context(self) -> DataContext:
        """Return the DataContext.

        Returns the context from the current dataset.

        Returns:
            The DataContext from the current dataset.
        """
        return self._iterators[self._dataset_idx].get_context()

    def _get_dataset_tag(self) -> str:
        """Return the dataset tag for the current dataset.

        Returns:
            A string with the tag from the current dataset.
        """
        tag = self._iterators[self._dataset_idx]._get_dataset_tag()
        return f"combined_iter_{self._dataset_idx}_{tag}"

    def get_current_dataset_idx(self) -> int:
        """Return the index of the current dataset.

        Returns:
            An integer with the index of the current dataset.
        """
        return self._dataset_idx
