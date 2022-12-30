"""An API for accessing the data in the ZeMA remaining-useful life dataset"""

__all__ = [
    "ExtractionDataType",
    "LOCAL_ZEMA_DATASET_PATH",
    "ZeMASamples",
    "ZEMA_DATASET_HASH",
    "ZEMA_DATASET_URL",
    "ZEMA_QUANTITIES",
]

import operator
import os
import pickle
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from os.path import dirname, exists
from pathlib import Path
from typing import cast

import h5py
import numpy as np
from h5py import Dataset
from numpy._typing import NDArray

from zema_emc_annotated.data_types import RealMatrix, RealVector, UncertainArray

LOCAL_ZEMA_DATASET_PATH = Path(dirname(__file__), "datasets")
ZEMA_DATASET_HASH = (
    "sha256:fb0e80de4e8928ae8b859ad9668a1b6ea6310028a6690bb8d4c1abee31cb8833"
)
ZEMA_DATASET_URL = "https://zenodo.org/record/5185953/files/axis11_2kHz_ZeMA_PTB_SI.h5"
ZEMA_QUANTITIES = (
    "Acceleration",
    "Active_Current",
    "Force",
    "Motor_Current",
    "Pressure",
    "Sound_Pressure",
    "Velocity",
)


class ExtractionDataType(Enum):
    """Identifiers of data types in ZeMA dataset

    Attributes
    ----------
    VALUES : str
        with value ``qudt:value``
    UNCERTAINTIES : str
        with value ``qudt:standardUncertainty``
    """

    VALUES = "qudt:value"
    UNCERTAINTIES = "qudt:standardUncertainty"


@dataclass
class ZeMASamples:
    """Extracts requested number of samples of values with associated uncertainties

    The underlying dataset is the annotated "Sensor data set of one electromechanical
    cylinder at ZeMA testbed (ZeMA DAQ and Smart-Up Unit)" by Dorst et al. [Dorst2021]_.

    Parameters
    ----------
    n_samples : int, optional
        number of samples each containing size_scaler readings from each of the
        eleven sensors with associated uncertainties, defaults to 1
    size_scaler : int, optional
        number of sensor readings from each of the individual sensors per sample,
        defaults to 1
    normalize : bool, optional
        if ``True``, then data is centered around zero and scaled to unit std,
        defaults to False

    Attributes
    ----------
    uncertain_values : UncertainArray
        The collection of samples of values with associated uncertainties,
        will be of shape (n_samples, 11 x size_scaler)
    """

    uncertain_values: UncertainArray

    def __init__(
        self, n_samples: int = 1, size_scaler: int = 1, normalize: bool = False
    ):

        self.normalize = normalize
        self.n_samples = n_samples
        self.size_scaler = size_scaler
        # if cached_data := _check_and_load_cache(n_samples, size_scaler):
        #     return cached_data
        dataset_full_path = (
            "/home/bjorn/code/zema_emc_annotated/src/zema_emc_annotated/"
            "datasets/394da54b1fc044fc498d60367c4e292d-axis11_2kHz_ZeMA_PTB_SI.h5"
        )
        # retrieve(
        #     url=ZEMA_DATASET_URL,
        #     known_hash=ZEMA_DATASET_HASH,
        #     path=LOCAL_ZEMA_DATASET_PATH,
        #     progressbar=True,
        # )
        assert exists(dataset_full_path)
        self._uncertainties = np.empty((n_samples, 0))
        self._values = np.empty((n_samples, 0))
        relevant_datasets = (
            ["ZeMA_DAQ", quantity, datatype.value]
            for quantity in ZEMA_QUANTITIES
            for datatype in ExtractionDataType
        )
        self._treating_uncertainties: bool = False
        self._treating_values: bool = False
        self._normalization_divisors: dict[str, NDArray[np.double] | float] = {}
        with h5py.File(dataset_full_path, "r") as h5f:
            for dataset_descriptor in relevant_datasets:
                self._current_dataset: Dataset = cast(
                    Dataset, reduce(operator.getitem, dataset_descriptor, h5f)
                )
                if ExtractionDataType.VALUES.value in self._current_dataset.name:
                    self._treating_values = True
                    print(f"    Extract values from {self._current_dataset.name}")
                elif (
                    ExtractionDataType.UNCERTAINTIES.value in self._current_dataset.name
                ):
                    self._treating_values = False
                    print(
                        f"    Extract uncertainties from {self._current_dataset.name}"
                    )
                else:
                    raise RuntimeError(
                        "Somehow there is unexpected data in the dataset to be"
                        f"processed. Did not expect to find "
                        f"{self._current_dataset.name}"
                    )
                if self._current_dataset.shape[0] == 3:
                    for idx, sensor in enumerate(self._current_dataset):
                        self._normalize_if_requested_and_append(
                            sensor, self._extract_sub_dataset_name(idx)
                        )
                else:
                    self._normalize_if_requested_and_append(
                        self._current_dataset,
                        self._strip_data_type_from_dataset_descriptor(),
                    )
                if self._treating_values:
                    print("    Values extracted")
                else:
                    print("    Uncertainties extracted")
        self._store_cache(
            uncertain_values := UncertainArray(self._values, self._uncertainties)
        )
        self.uncertain_values = uncertain_values

    def _normalize_if_requested_and_append(
        self, data: Dataset, dataset_descriptor: str
    ) -> None:
        """Normalize the provided data and append according to current state"""
        _potentially_normalized_data = data[
            np.s_[1 : self.size_scaler + 1, : self.n_samples]
        ]
        if self._treating_values:
            if self.normalize:
                _potentially_normalized_data -= np.mean(
                    data[:, : self.n_samples], axis=0
                )
                data_std = np.std(data[:, : self.n_samples], axis=0)
                data_std[data_std == 0] = 1.0
                self._normalization_divisors[dataset_descriptor] = data_std
                _potentially_normalized_data /= self._normalization_divisors[
                    dataset_descriptor
                ]
            self._values = np.append(
                self._values, _potentially_normalized_data.transpose(), axis=1
            )
        else:
            if self.normalize:
                _potentially_normalized_data /= self._normalization_divisors[
                    dataset_descriptor
                ]
            self._uncertainties = np.append(
                self._uncertainties, _potentially_normalized_data.transpose(), axis=1
            )

    def _extract_sub_dataset_name(self, idx: int) -> str:
        return str(
            self._strip_data_type_from_dataset_descriptor()
            + self._current_dataset.attrs["si:label"]
            .split(",")[idx]
            .strip("[")
            .strip("]")
            .replace(" ", "")
            .replace('"', "")
            .replace("uncertainty", "")
        ).replace("\n", "")

    def _strip_data_type_from_dataset_descriptor(self) -> str:
        return str(
            self._current_dataset.name.replace(
                ExtractionDataType.UNCERTAINTIES.value, ""
            ).replace(ExtractionDataType.VALUES.value, "")
        )

    @property
    def values(self) -> RealVector:
        """The values of the stored :class:`UncertainArray` object"""
        return self.uncertain_values.values

    @property
    def uncertainties(self) -> RealMatrix | RealVector:
        """The uncertainties of the stored :class:`UncertainArray` object"""
        return self.uncertain_values.uncertainties

    @staticmethod
    def _check_and_load_cache(
        n_samples: int, size_scaler: int
    ) -> UncertainArray | None:
        """Checks if corresponding file for n_samples exists and loads it with pickle"""
        if os.path.exists(
            cache_path := ZeMASamples._cache_path(n_samples, size_scaler)
        ):
            with open(cache_path, "rb") as cache_file:
                return cast(UncertainArray, pickle.load(cache_file))
        return None

    @staticmethod
    def _cache_path(n_samples: int, size_scaler: int) -> Path:
        """Local file system path for a cache file containing n ZeMA samples

        The result does not guarantee, that the file at the specified location exists,
        but can be used to check for existence or creation.
        """
        return LOCAL_ZEMA_DATASET_PATH.joinpath(
            f"{str(n_samples)}_samples_with_{str(size_scaler)}_values_per_sensor.pickle"
        )

    @staticmethod
    def _store_cache(uncertain_values: UncertainArray) -> None:
        """Dumps provided uncertain tenor to corresponding pickle file"""
        with open(
            ZeMASamples._cache_path(
                uncertain_values.values.shape[0],
                int(uncertain_values.values.shape[1] / 11),
            ),
            "wb",
        ) as cache_file:
            pickle.dump(uncertain_values, cache_file)
