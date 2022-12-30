"""Common strategies"""
from typing import cast

import numpy as np
from hypothesis import strategies as hst
from hypothesis.extra import numpy as hnp
from hypothesis.strategies import composite, DrawFn, SearchStrategy
from numpy._typing import NDArray

from zema_emc_annotated.data_types import UncertainArray


@composite
def uncertain_arrays(
    draw: DrawFn,
    greater_than: float = -1e2,
    less_than: float = 1e2,
    samples: int | None = None,
    scaler: int | None = None,
) -> SearchStrategy[UncertainArray]:
    if samples is None:
        samples = draw(hst.integers(min_value=1, max_value=10))
    if scaler is None:
        scaler = draw(hst.integers(min_value=1, max_value=10))
    values: NDArray[np.float64] = cast(
        NDArray[np.float64],
        draw(
            hnp.arrays(
                dtype=np.float64,
                shape=(samples, scaler * 11),
                elements=hst.floats(min_value=greater_than, max_value=less_than),
            )
        ),
    )
    std_uncertainties = cast(
        NDArray[np.float64],
        draw(
            hnp.arrays(
                dtype=np.float64,
                shape=values.shape,
                elements=hst.floats(
                    min_value=np.abs(values).min() * 1e-3,
                    max_value=np.abs(values).min() * 1e2,
                ),
            )
        ),
    )
    return cast(
        SearchStrategy[UncertainArray],
        UncertainArray(values, std_uncertainties),
    )
