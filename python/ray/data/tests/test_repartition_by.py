import pandas as pd

import ray
import ray.data
from ray.data.aggregate import Sum
from ray.data.tests.conftest import *  # noqa
from ray.tests.conftest import *  # noqa

RANDOM_SEED = 123


def test_repartition_by_multi_blocks(ray_start_10_cpus):
    ds = ray.data.from_pandas(
        [
            pd.DataFrame(
                {
                    "x": [i, i, i + 100, i + 100],
                    "y": [0, 1, 2, 3],
                }
            )
            for i in range(5)
        ]
    )
    assert ds.num_blocks() == 5

    ds1 = ds.repartition_by("x").materialize()
    assert ds1.num_blocks() == 10

    assert ds1.aggregate(Sum("y")) == {"sum(y)": 30}

    result = ds1.groupby("x").count().take_all()
    result = sorted(result, key=lambda d: d["x"])
    assert result == [
        {"count()": 2, "x": 0},
        {"count()": 2, "x": 1},
        {"count()": 2, "x": 2},
        {"count()": 2, "x": 3},
        {"count()": 2, "x": 4},
        {"count()": 2, "x": 100},
        {"count()": 2, "x": 101},
        {"count()": 2, "x": 102},
        {"count()": 2, "x": 103},
        {"count()": 2, "x": 104},
    ]

    result = ds1.map_batches(
        fn=lambda d: {
            "x": [d["x"][0]],
            "count()": [len(d["x"])],
        },
        batch_size=None,
    ).take_all()
    result = sorted(result, key=lambda d: d["x"])
    assert result == [
        {"count()": 2, "x": 0},
        {"count()": 2, "x": 1},
        {"count()": 2, "x": 2},
        {"count()": 2, "x": 3},
        {"count()": 2, "x": 4},
        {"count()": 2, "x": 100},
        {"count()": 2, "x": 101},
        {"count()": 2, "x": 102},
        {"count()": 2, "x": 103},
        {"count()": 2, "x": 104},
    ]


def test_repartition_by_parallelism_1(ray_start_regular_shared):
    ds = ray.data.range(10, parallelism=1)
    ds = ds.repartition_by("id")

    assert ds.count() == 10
    assert ds.num_blocks() == 10

    ds = ray.data.from_items([1, 1, 1, 2, 2, 2, 3, 3, 3], parallelism=1)
    ds = ds.repartition_by("item")

    assert ds.count() == 9
    assert ds.num_blocks() == 3
