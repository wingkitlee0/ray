from typing import TYPE_CHECKING, Any, Tuple

if TYPE_CHECKING:
    import mlx.core as mx


def _mlx_array_deserializer(
    np_array: Any,
    dtype_str: str,
    shape: Tuple[int, ...],
) -> "mx.array":
    """
    Reconstructs an mlx.core.array from a numpy array (buffer).

    Args:
        np_array: Generic numpy array acting as the buffer.
        dtype_str: string representation of the mlx dtype.
        shape: Shape of the array.

    Returns:
        New mlx.core.array (currently involves a copy during construction).
    """
    try:
        import mlx.core as mx
    except ImportError:
        raise ImportError("mlx must be installed to deserialize MLX arrays.")

    return mx.array(np_array)


def mlx_array_reducer(array: "mx.array"):
    """
    Reducer for mlx.core.array.

    Converts MLX array to a Numpy array to leverage Ray's optimized
    numpy serialization (zero-copy to plasma).
    """
    import numpy as np

    np_ver = np.array(array, copy=False)

    return _mlx_array_deserializer, (np_ver, str(array.dtype), array.shape)
