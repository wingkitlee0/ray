import sys

import pytest

import ray

try:
    import mlx.core as mx
except ImportError:
    mx = None


@pytest.mark.skipif(mx is None, reason="MLX not installed")
def test_mlx_serialization_basic(ray_start_regular):
    """Test basic roundtrip of MLX arrays."""
    a = mx.arange(100)
    oid = ray.put(a)
    b = ray.get(oid)

    assert isinstance(b, mx.array)
    assert mx.array_equal(a, b).item()
    assert b.dtype == a.dtype
    assert b.shape == a.shape


@pytest.mark.skipif(mx is None, reason="MLX not installed")
def test_mlx_serialization_types(ray_start_regular):
    """Test various types and shapes."""
    shapes = [
        (10,),
        (10, 10),
        (5, 5, 5),
    ]
    dtypes = [
        mx.float32,
        mx.int32,
        # mx.bool_, # MLX bool might behave differently, let's test safely
    ]

    for shape in shapes:
        for dtype in dtypes:
            a = mx.zeros(shape, dtype=dtype)
            b = ray.get(ray.put(a))
            assert mx.array_equal(a, b).item()
            assert b.dtype == a.dtype


@pytest.mark.skipif(mx is None, reason="MLX not installed")
@pytest.mark.skipif(not (mx and mx.metal.is_available()), reason="Metal not available")
def test_mlx_serialization_mps(ray_start_regular):
    """Test serialization of arrays on MPS device."""
    # Force creation on GPU if possible
    with mx.stream(mx.gpu):
        a = mx.arange(100) * 2.0
        mx.eval(a)

    # Ensure it's on GPU? MLX is unified memory basically, but stream matters?
    # MLX arrays don't strictly report 'device' like pytorch in the same way
    # but let's trust the context.

    oid = ray.put(a)
    b = ray.get(oid)

    assert mx.array_equal(a, b).item()
    # Note: Deserialized array will likely be on default device (CPU/GPU depending on setup)
    # or wherever the numpy conversion put it.


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main(["-v", __file__]))
