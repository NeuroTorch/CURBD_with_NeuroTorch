"""
This file is used to reproduce the results of the CURBD repository (https://github.com/rajanlab/CURBD)
using the NeuroTorch library.
"""

import numpy as np
import torch
import neurotorch as nt
from neurotorch import WilsonCowanCURBDLayer

from curbd_dataset import CURBD3RegionsDataset
from curbd_training import train_with_curbd, make_eval_prediction
from figures_script import complete_report

if __name__ == '__main__':
    number_units = 3 * 100
    tau = 0.1
    n_time_steps = 1400
    amp_in_wn = 0.01
    dt = 0.01

    dataset = CURBD3RegionsDataset(
        n_units=number_units,
        tau=tau,
        time_seconds=int(n_time_steps * dt),
        amp_in_wn=amp_in_wn,
        dtype=np.float32
    )

    layer = WilsonCowanCURBDLayer(
        input_size=dataset.n_units,
        output_size=dataset.n_units,
        tau=tau,
        dt=dt,
        learn_tau=False,
        force_dale_law=False,
        use_recurrent_connection=True,
        use_rec_eye_mask=False,
        forward_weights=torch.eye(number_units, dtype=torch.float32),
        # recurrent_weights=J,
        activation=torch.nn.Tanh(),
        hh_init="given",
        h0=(dataset.get_initial_condition().reshape(1, -1), ),
    ).build()
    layer.forward_weights.requires_grad = False
    model = nt.SequentialRNN(layers=[layer], checkpoint_folder=f"data/curbd/checkpoints").build()
    print(f"Model:\n{model}")
    model, history, trainer = train_with_curbd(model, dataset, force_overwrite=True)

    eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_pred, y_target = make_eval_prediction(model, dataset, device=eval_device)
    wc_viz_pred, wc_viz_target = complete_report(model, y_pred=y_pred, y_target=y_target)
    history.plot(show=True)
