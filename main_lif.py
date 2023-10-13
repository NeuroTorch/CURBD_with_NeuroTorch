"""
This file is used to reproduce the results of the CURBD repository (https://github.com/rajanlab/CURBD)
using the NeuroTorch library.
"""

import numpy as np
import torch
import neurotorch as nt
from neurotorch import LIFLayer

from curbd_dataset import CURBD3RegionsDataset
from curbd_training import train_with_curbd
from ts_dataset import TSDataset


class TakeItemFromBatch(torch.nn.Module):
    """
    Take the last element of a sequence in the given dimension.
    """
    def __init__(self, dim: int = 1, idx: int = -1):
        super().__init__()
        self.dim = dim
        self.idx = idx

    def make_slice(self, x):
        if isinstance(x, torch.Tensor):
            slice_ = [slice(None)] * len(x.shape)
            slice_[self.dim] = self.idx
            return tuple(slice_)
        elif isinstance(x, dict):
            return {
                k: self.make_slice(v)
                for k, v in x.items()
            }
        elif isinstance(x, list):
            return [self.make_slice(v) for v in x]
        elif isinstance(x, tuple):
            return tuple(self.make_slice(v) for v in x)
        else:
            raise ValueError("Inputs must be a torch.Tensor or a dictionary.")

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            out_item = x[self.make_slice(x)]
        elif isinstance(x, dict):
            out_item = {k: self(v) for k, v in x.items()}
        elif isinstance(x, list):
            out_item = [self(v) for v in x]
        elif isinstance(x, tuple):
            out_item = tuple(self(v) for v in x)
        else:
            raise ValueError("Inputs must be a torch.Tensor or a dictionary.")
        return out_item

    def extra_repr(self) -> str:
        return f"dim={self.dim}, idx={self.idx}"


class SeqLayer(nt.modules.layers.BaseLayer):
    def __init__(
            self,
            layer: nt.modules.layers.BaseLayer,
            foresight_time_steps: int = 10,
            **kwargs
    ):
        super().__init__(
            input_size=layer.input_size,
            output_size=layer.output_size,
            name=layer.name,
            device=layer.device,
            **kwargs
        )
        self.seq = nt.SequentialRNN(
            layers=[layer],
            foresight_time_steps=foresight_time_steps,
            output_transform=kwargs.pop("output_transform", [nt.transforms.ReduceMean(dim=1)]),
            **kwargs
        ).build()
        self.take_last = TakeItemFromBatch(dim=1, idx=-1)

    def forward(self, inputs: torch.Tensor, state: torch.Tensor = None,  **kwargs):
        out, hh = nt.utils.unpack_out_hh(self.seq(inputs, state, **kwargs))
        hh = self.take_last(hh)
        return nt.utils.maybe_unpack_singleton_dict(out), nt.utils.maybe_unpack_singleton_dict(hh)


class UnpackTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nt.utils.maybe_unpack_singleton_dict(x)


if __name__ == '__main__':
    number_units = 3 * 100
    tau = 0.1
    n_time_steps = 1400
    # n_time_steps = 100
    amp_in_wn = 0.01
    dt = 0.01

    dataset = CURBD3RegionsDataset(
        n_units=number_units,
        tau=tau,
        dt_data=dt,
        time_seconds=int(n_time_steps * dt),
        lead_time=2,
        amp_in_wn=amp_in_wn,
        dtype=np.float32,
        re_hh=False,
    )
    # dataset = TSDataset(
    #     n_units=number_units,
    #     tau=tau,
    #     amp_in_wn=amp_in_wn,
    #     dt_data=dt,
    #     re_hh=False,
    #     filename="spikes.npy",
    # )
    h0 = dataset.get_initial_condition().reshape(1, -1)
    layer = nt.SpyLIFLayer(
        input_size=dataset.n_units,
        output_size=dataset.n_units,
        tau_m=tau,
        dt=dt,
        force_dale_law=False,
        use_recurrent_connection=True,
        use_rec_eye_mask=False,
        forward_weights=torch.eye(number_units, dtype=torch.float32),
        # recurrent_weights=J,
        # activation=torch.nn.Tanh(),
        hh_init="given",
        h0=(torch.rand_like(h0), torch.rand_like(h0), h0),
    ).build()
    layer.forward_weights.requires_grad = False
    seq_layer = SeqLayer(
        layer=layer,
        foresight_time_steps=10,
    ).build()
    model = nt.SequentialRNN(layers=[seq_layer], output_transform=[UnpackTransform()]).build()
    print(f"Model:\n{model}")
    model, history, trainer = train_with_curbd(
        model,
        dataset,
        force_overwrite=True,
        learning_algorithm=nt.TBPTT(
            default_optimizer_cls=torch.optim.AdamW,
            params_lr=2e-4,
            backward_time_steps=1,
            optim_time_steps=1
        )
    )
    history.plot(show=True)


