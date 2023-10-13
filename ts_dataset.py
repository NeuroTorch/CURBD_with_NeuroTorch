import os
from copy import deepcopy
from typing import Optional, Iterable

import numpy as np
import numpy.random as npr
import torch
import neurotorch as nt
from neurotorch import to_tensor
from pythonbasictools.google_drive import GoogleDriveDownloader
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset


class TSDataset(Dataset):
    ROOT_FOLDER = "data/ts/"
    FILE_ID_NAME = {
        "SampleZebrafishData_PaulDeKoninckLab_2020-12-16.npy": "1-3jgAZiNU__NxxhXub7ezAJUqDMFpMCO",
        "Stimulus_data_2022_02_23_fish3_1.npy": "19DnsoI_z4IAWGSLP-32JR4rEMMBMp2Gf",
        "Stimulus_data_2022_02_23_fish3_1.hdf5": "1rliot_adOAYMzHZjoKrIAJuXnxTmwIpX",
    }

    def __init__(
            self,
            n_units: int = 3 * 100,
            n_time_steps: Optional[int] = None,
            seed: int = 0,
            filename: Optional[str] = None,
            smoothing_sigma: float = 0.0,
            download: bool = True,
            amp_in_wn=0.01,
            dt_data=0.01,
            tau=0.1,
            units: Optional[Iterable[int]] = None,
            **kwargs
    ):
        self.n_units = n_units
        self.dt_data = dt_data
        self.tau = tau
        self.amp_in_wn = amp_in_wn
        self.kwargs = kwargs
        verbose = kwargs.get("verbose", False)
        self.ROOT_FOLDER = kwargs.get("root_folder", self.ROOT_FOLDER)
        if filename is None:
            filename = list(self.FILE_ID_NAME.keys())[0]
            download = True
        if os.path.exists(filename):
            path = filename
        else:
            path = os.path.join(self.ROOT_FOLDER, filename)
        if os.path.exists(path):
            download = False
        if download:
            assert filename in self.FILE_ID_NAME, \
                f"File {filename} not found in the list of available files: {list(self.FILE_ID_NAME.keys())}."
            GoogleDriveDownloader(
                self.FILE_ID_NAME[filename], path, skip_existing=True, verbose=verbose
            ).download()
        self.filename = filename
        self.hdf5_file = None
        if filename.endswith(".npy"):
            self.ts = np.load(path)
        elif filename.endswith(".hdf5"):
            import h5py
            self.hdf5_file = h5py.File(path, 'r')
            self.ts = np.asarray(self.hdf5_file["timeseries"])
        else:
            raise ValueError(f"File format not supported: {filename}")
        self.original_time_series = deepcopy(self.ts)
        if self.kwargs.get("rm_dead_units", True):
            self.ts = self.ts[np.sum(np.abs(self.ts), axis=-1) > 0, :]
        self.total_n_neurons, self.total_n_time_steps = self.ts.shape

        self.seed = seed
        random_generator = np.random.RandomState(seed)

        if units is not None:
            units = list(units)
            assert n_units is None or len(
                units
            ) == n_units, "Number of units and number of units in units must be equal"
            n_units = len(units)
        elif n_units is not None:
            if n_units < 0:
                n_units = self.total_n_neurons + n_units + 1
            units = random_generator.randint(self.total_n_neurons, size=n_units)
        else:
            n_units = self.total_n_neurons
            units = random_generator.randint(self.total_n_neurons, size=n_units)

        if n_time_steps is None or n_time_steps > self.total_n_time_steps or n_time_steps < 0:
            self.n_time_steps = self.total_n_time_steps
        else:
            self.n_time_steps = n_time_steps

        self._given_n_time_steps = n_time_steps
        self.n_units = n_units
        self.units_indexes = units
        self.norm_eps = self.kwargs.get("eps", 1e-5)
        self._initial_data = deepcopy(self.ts[self.units_indexes].T)
        self.data = None
        self.reset_data_()
        self._sigma = None
        self.sigma = smoothing_sigma

        self.data = nt.to_tensor(self.data, dtype=torch.float32)
        self.target_skip_first = kwargs.get("target_skip_first", True)

        self._is_simulated = False
        self._noise_is_simulated = False
        self.simulation = None
        self.input_noise_data = None
        self.dtype = kwargs.get("dtype", np.float32)
        self.re_hh = kwargs.get("re_hh", False)

    def simulate(self):
        self.input_noise_data = self.exec_input_noise()
        self._noise_is_simulated = True

    def exec_input_noise(self):
        ampWN = np.sqrt(self.tau / self.dt_data)
        iWN = ampWN * npr.randn(self.n_units, self.n_time_steps)
        inputWN = np.ones((self.n_units, self.n_time_steps))
        for tt in range(1, self.n_time_steps):
            inputWN[:, tt] = iWN[:, tt] + (inputWN[:, tt - 1] - iWN[:, tt]) * np.exp(- (self.dt_data / self.tau))
        inputWN = self.amp_in_wn * inputWN
        return inputWN

    def get_activity(self):
        if not self._is_simulated:
            self.simulate()
        activity = self.data
        return activity  # shapes: (time, units)

    def __len__(self):
        return 1

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma: float):
        assert sigma >= 0, "Sigma must be null or positive"
        self._sigma = sigma
        self.reset_data_()
        self.normalize_data_()
        self.smooth_data_()
        self.cast_data_to_tensor_()

    @property
    def normalize_mth(self):
        return self.kwargs.get("normalize_mth", "abs_max")

    def normalize_data_(self):
        self.cast_data_to_numpy_()
        normalize_mth_dict = {
            "min_max": self.normalize_min_max,
            "abs_max": self.normalize_abs_max,
            "standard": self.normalize_standard,
            "minus_one_one": self.normalize_minus_one_one,
        }
        if self.kwargs.get("normalize", True):
            if self.kwargs.get("normalize_by_unit", False):
                for neuron in range(self.data.shape[-1]):
                    self.data[:, neuron] = normalize_mth_dict[self.normalize_mth](self.data[:, neuron])
            else:
                self.data = normalize_mth_dict[self.normalize_mth](self.data)
        self.cast_data_to_tensor_()

    def normalize_min_max(self, data: np.ndarray):
        normed_data = data - np.min(data)
        normed_data = normed_data / (np.max(normed_data) + self.norm_eps)
        return normed_data

    def normalize_abs_max(self, data: np.ndarray):
        normed_data = data / (np.max(np.abs(data)) + self.norm_eps)
        return normed_data

    def normalize_standard(self, data: np.ndarray):
        normed_data = data - np.mean(data)
        normed_data = normed_data / (np.std(normed_data) + self.norm_eps)
        return normed_data

    def normalize_minus_one_one(self, data: np.ndarray):
        normed_data = data - np.mean(data)
        normed_data = normed_data / (np.max(np.abs(normed_data)) + self.norm_eps)
        return normed_data

    def smooth_data_(self):
        assert self.sigma >= 0, "Sigma must be null or positive"
        if not np.isclose(self.sigma, 0.0):
            self.cast_data_to_numpy_()
            self.data = gaussian_filter1d(self.data, sigma=self.sigma, mode="nearest", axis=0)
            self.cast_data_to_tensor_()

    def __getitem__(self, idx):
        if not self._is_simulated:
            self.simulate()
        activity = self.get_activity()
        input_noise = deepcopy(self.input_noise_data).T
        x = to_tensor(input_noise)[1:]  # shape: (time, units)
        h0, y = to_tensor(activity)[0], to_tensor(activity)[1:]  # shape: (time, units)
        if self.re_hh:
            return deepcopy(x), deepcopy(h0), deepcopy(y)
        else:
            return deepcopy(x), deepcopy(y)

    def get_initial_condition(self):
        return deepcopy(to_tensor(self.get_activity()[0]))  # shape: (units, )

    def set_params_from_self(self, kwargs: dict):
        kwargs["n_units"] = self.n_units
        kwargs["n_time_steps"] = self.n_time_steps
        kwargs["smoothing_sigma"] = self.sigma
        kwargs["filename"] = self.filename
        return kwargs

    def cast_data_to_tensor_(self, dtype=torch.float32):
        self.data = nt.to_tensor(self.data, dtype=dtype)
        return self.data

    def cast_data_to_numpy_(self, dtype=np.float32):
        self.data = nt.to_numpy(self.data, dtype=dtype)
        return self.data

    def reset_data_(self):
        self.data = deepcopy(self._initial_data)
        self.cast_data_to_tensor_()
        return self.data

    def __repr__(self):
        return f"{self.__class__.__name__}(n_units={self.n_units}, n_time_steps={self.n_time_steps}, " \
               f"sigma={self.sigma}, filename={self.filename})"
























