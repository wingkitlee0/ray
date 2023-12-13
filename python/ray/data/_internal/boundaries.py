from typing import Tuple, Union
import numpy as np


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
        indices = np.hstack([[0], np.where(np.diff(keys) != 0)[0] + 1, [len(keys)]])
        return keys[indices[:-1]], indices

    s = {0}
    for arr in keys.values():
        s |= set(np.where(np.diff(arr) != 0)[0] + 1)
    s = sorted(s)
    s.append(len(arr))
    indices = np.array(s)

    grouped_keys = {
        key: value[indices[:-1]] for key, value in keys.items() if len(value) > 0
    }
    return grouped_keys, indices