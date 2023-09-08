from copy import deepcopy

import numpy as np
import numpy.random as npr
from neurotorch import to_tensor
from torch.utils.data import Dataset


class CURBD3RegionsDataset(Dataset):
    """
    Generate a dataset according to CURBD repository https://github.com/rajanlab/CURBD.
    """

    def __init__(
            self,
            n_units: int = 3 * 100,
            ga=1.8,
            gb=1.5,
            gc=1.5,
            tau=0.1,
            frac_inter_reg=0.05,
            amp_inter_reg=0.02,
            frac_external=0.5,
            amp_in_b=1,
            amp_in_c=-1,
            dt_data=0.01,
            time_seconds=10,
            lead_time=2,
            bump_std=0.2,
            amp_in_wn=0.01,
            **kwargs
    ):
        self.n_units = n_units
        self.ga = ga
        self.gb = gb
        self.gc = gc
        self.tau = tau
        self.frac_inter_reg = frac_inter_reg
        self.amp_inter_reg = amp_inter_reg
        self.frac_external = frac_external
        self.amp_in_b = amp_in_b
        self.amp_in_c = amp_in_c
        self.dt_data = dt_data
        self.time_seconds = time_seconds
        self.lead_time = lead_time
        self.bump_std = bump_std
        self.amp_in_wn = amp_in_wn
        self.n_time_steps = int((self.time_seconds + self.lead_time) / self.dt_data) + 1
        self.kwargs = kwargs

        self._is_simulated = False
        self.simulation = None
        self.input_noise_data = None
        self.dtype = kwargs.get("dtype", np.float32)

        assert self.n_units % 3 == 0, "Number of units must be divisible by 3"

    def simulate(self):
        self.simulation = self.exec_simulation()
        self.input_noise_data = self.exec_input_noise()
        self._is_simulated = True

    def exec_simulation(self):
        time_data = np.arange(0, (self.time_seconds + self.dt_data), self.dt_data)

        # for now it only works if the networks are the same size
        N = int(self.n_units / 3)
        Na = Nb = Nc = N

        # set up RNN A (chaotic responder)
        Ja = npr.randn(Na, Na)
        Ja = self.ga / np.sqrt(Na) * Ja
        hCa = 2 * npr.rand(Na, 1) - 1  # start from random state

        # set up RNN B (driven by sequence)
        Jb = npr.randn(Nb, Nb)
        Jb = self.gb / np.sqrt(Na) * Jb
        hCb = 2 * npr.rand(Nb, 1) - 1  # start from random state

        # set up RNN C (driven by fixed point)
        Jc = npr.randn(Nc, Nc)
        Jc = self.gb / np.sqrt(Na) * Jc
        hCc = 2 * npr.rand(Nc, 1) - 1  # start from random state

        # generate external inputs
        # set up sequence-driving network
        xBump = np.zeros((Nb, len(time_data)))
        sig = self.bump_std * Nb  # width of bump in N units

        norm_by = 2 * sig ** 2
        cut_off = int(np.ceil(len(time_data) / 2)) - 100
        for i in range(Nb):
            stuff = (i - sig - Nb * time_data / (time_data[-1] / 2)) ** 2 / norm_by
            xBump[i, :] = np.exp(-stuff)
            xBump[i, cut_off:] = xBump[i, cut_off]

        hBump = np.log((xBump + 0.01) / (1 - xBump + 0.01))
        hBump = hBump - np.min(hBump)
        hBump = hBump / np.max(hBump)

        # set up fixed points driving network

        xFP = np.zeros((Nc, len(time_data)))
        cut_off = int(np.ceil(len(time_data) / 2)) + 100
        for i in range(Nc):
            front = xBump[i, 10] * np.ones((1, cut_off))
            back = xBump[i, 300] * np.ones((1, len(time_data) - cut_off))
            xFP[i, :] = np.concatenate((front, back), axis=1)
        hFP = np.log((xFP + 0.01) / (1 - xFP + 0.01))
        hFP = hFP - np.min(hFP)
        hFP = hFP / np.max(hFP)

        # add the lead time
        extratData = np.arange(time_data[-1] + self.dt_data, self.time_seconds + self.lead_time, self.dt_data)
        time_data = np.concatenate((time_data, extratData))

        newmat = np.tile(hBump[:, 1, np.newaxis], (1, int(np.ceil(self.lead_time / self.dt_data))))
        hBump = np.concatenate((newmat, hBump), axis=1)

        newmat = np.tile(hFP[:, 1, np.newaxis], (1, int(np.ceil(self.lead_time / self.dt_data))))
        hFP = np.concatenate((newmat, hFP), axis=1)

        # build connectivity between RNNs
        Nfrac = int(self.frac_inter_reg * N)

        rand_idx = npr.permutation(N)
        w_A2B = np.zeros((N, 1))
        w_A2B[rand_idx[0:Nfrac]] = 1

        rand_idx = npr.permutation(N)
        w_A2C = np.zeros((N, 1))
        w_A2C[rand_idx[0:Nfrac]] = 1

        rand_idx = npr.permutation(N)
        w_B2A = np.zeros((N, 1))
        w_B2A[rand_idx[0:Nfrac]] = 1

        rand_idx = npr.permutation(N)
        w_B2C = np.zeros((N, 1))
        w_B2C[rand_idx[0:Nfrac]] = 1

        rand_idx = npr.permutation(N)
        w_C2A = np.zeros((N, 1))
        w_C2A[rand_idx[0:Nfrac]] = 1

        rand_idx = npr.permutation(N)
        w_C2B = np.zeros((N, 1))
        w_C2B[rand_idx[0:Nfrac]] = 1

        # Sequence only projects to B
        Nfrac = int(self.frac_external * N)
        rand_idx = npr.permutation(N)
        w_Seq2B = np.zeros((N, 1))
        w_Seq2B[rand_idx[0:Nfrac]] = 1

        # Fixed point only projects to A
        Nfrac = int(self.frac_external * N)
        rand_idx = npr.permutation(N)
        w_Fix2C = np.zeros((N, 1))
        w_Fix2C[rand_idx[0:Nfrac]] = 1

        # generate time series simulated data
        Ra = np.empty((Na, len(time_data)))
        Ra[:] = np.NaN

        Rb = np.empty((Nb, len(time_data)))
        Rb[:] = np.NaN

        Rc = np.empty((Nc, len(time_data)))
        Rc[:] = np.NaN

        for tt in range(len(time_data)):
            Ra[:, tt, np.newaxis] = np.tanh(hCa)
            Rb[:, tt, np.newaxis] = np.tanh(hCb)
            Rc[:, tt, np.newaxis] = np.tanh(hCc)
            # chaotic responder
            JRa = Ja.dot(Ra[:, tt, np.newaxis])
            JRa += self.amp_inter_reg * w_B2A * Rb[:, tt, np.newaxis]
            JRa += self.amp_inter_reg * w_C2A * Rc[:, tt, np.newaxis]
            hCa = hCa + self.dt_data * (-hCa + JRa) / self.tau

            # sequence driven
            JRb = Jb.dot(Rb[:, tt, np.newaxis])
            JRb += self.amp_inter_reg * w_A2B * Ra[:, tt, np.newaxis]
            JRb += self.amp_inter_reg * w_C2B * Rc[:, tt, np.newaxis]
            JRb += self.amp_in_b * w_Seq2B * hBump[:, tt, np.newaxis]
            hCb = hCb + self.dt_data * (-hCb + JRb) / self.tau

            # fixed point driven
            JRc = Jc.dot(Rc[:, tt, np.newaxis])
            JRc += self.amp_inter_reg * w_B2C * Rb[:, tt, np.newaxis]
            JRc += self.amp_inter_reg * w_A2C * Ra[:, tt, np.newaxis]
            JRc += self.amp_in_c * w_Fix2C * hFP[:, tt, np.newaxis]
            hCc = hCc + self.dt_data * (-hCc + JRc) / self.tau

        # package up outputs
        Rseq = hBump.copy()
        Rfp = hFP.copy()
        # normalize
        Ra = Ra / np.max(Ra)
        Rb = Rb / np.max(Rb)
        Rc = Rc / np.max(Rc)
        Rseq = Rseq / np.max(Rseq)
        Rfp = Rfp / np.max(Rfp)

        out_params = {'Na': Na, 'Nb': Nb, 'Nc': Nc, 'ga': self.ga, 'gb': self.gb, 'gc': self.gc, 'tau': self.tau,
                      'fracInterReg': self.frac_inter_reg, 'ampInterReg': self.amp_inter_reg,
                      'fracExternal': self.frac_external, 'ampInB': self.amp_in_b, 'ampInC': self.amp_in_c,
                      'dtData': self.dt_data, 'T': self.time_seconds, 'leadTime': self.lead_time,
                      'bumpStd': self.bump_std}

        out = {'Ra': Ra, 'Rb': Rb, 'Rc': Rc, 'Rseq': Rseq, 'Rfp': Rfp, 'tData': time_data, 'Ja': Ja, 'Jb': Jb, 'Jc': Jc,
               'w_A2B': w_A2B, 'w_A2C': w_A2C, 'w_B2A': w_B2A, 'w_B2C': w_B2C, 'w_C2A': w_C2A, 'w_C2B': w_C2B,
               'w_Fix2C': w_Fix2C, 'w_Seq2B': w_Seq2B, 'params': out_params}
        return out

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
        activity = np.concatenate((self.simulation['Ra'], self.simulation['Rb'], self.simulation['Rc']), 0).T
        activity = activity / np.max(activity)
        activity = np.minimum(activity, 0.999)
        activity = np.maximum(activity, -0.999)
        activity = activity.astype(self.dtype)
        return activity  # shapes: (time, units)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if not self._is_simulated:
            self.simulate()
        activity = np.concatenate((self.simulation['Ra'], self.simulation['Rb'], self.simulation['Rc']), 0).T
        input_noise = deepcopy(self.input_noise_data).T
        x = to_tensor(input_noise)[1:]  # shape: (time, units)
        y = to_tensor(activity)[1:]  # shape: (time, units)
        return deepcopy(x), deepcopy(y)

    def get_initial_condition(self):
        return deepcopy(to_tensor(self.get_activity()[0]))  # shape: (units, )
