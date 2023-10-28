"""
This file is used to reproduce the results of the CURBD repository (https://github.com/rajanlab/CURBD)
using the NeuroTorch library.
"""
import pickle

import numpy as np
import torch
import neurotorch as nt
from neurotorch import WilsonCowanCURBDLayer

from curbd_dataset import CURBD3RegionsDataset
from curbd_training import train_with_curbd, make_eval_prediction
from figures_script import complete_report
from ts_dataset import TSDataset
from util import model_summary, TBPTTHook

if __name__ == '__main__':
    number_units = -1
    # number_units = 600
    tau = 0.1
    # n_time_steps = 1400
    n_time_steps = -1
    amp_in_wn = 0.01
    dt = 0.01

    # dataset = CURBD3RegionsDataset(
    #     n_units=number_units,
    #     tau=tau,
    #     time_seconds=int(n_time_steps * dt),
    #     lead_time=2,
    #     amp_in_wn=amp_in_wn,
    #     dtype=np.float32
    # )

    dataset = TSDataset(
        n_units=number_units,
        tau=tau,
        amp_in_wn=amp_in_wn,
        dt_data=dt,
        re_hh=False,
        # filename="spikes.npy",
        filename="dff_matrix_N1644_T1975.npy",
        # filename="dff_matrix_N1019_T1975.npy",
        # filename="dff_matrix_N600_T1974.npy",
        # smoothing_sigma=1,
        # smoothing_sigma=2*3.57,
        smoothing_sigma=2*3.57,
        normalize=True,
        normalize_mth="min_max",
        # normalize_mth="standard",
        normalize_by_unit=True,
    )
    nbu_str = f"u" if dataset.kwargs.get("normalize_by_unit", False) else f"d"
    norm_mth_to_activation_funcs = {
        "min_max": torch.nn.Sigmoid(),
        "abs_max": torch.nn.Tanh(),
        "standard": torch.nn.Tanh(),
        "minus_one_one": torch.nn.Tanh(),
    }
    print(f"Dataset:\n{dataset}")
    layer = nt.WilsonCowanLayer(
        input_size=dataset.n_units,
        output_size=dataset.n_units,
        tau=tau,
        dt=dt,
        learn_tau=True,
        force_dale_law=True,
        use_recurrent_connection=True,
        use_rec_eye_mask=False,
        forward_weights=torch.eye(dataset.n_units, dtype=torch.float32),
        forward_sign=torch.ones((dataset.n_units, 1), dtype=torch.float32),
        # recurrent_weights=J,
        # activation=torch.nn.Sigmoid(),
        # activation=torch.nn.Tanh(),
        activation=norm_mth_to_activation_funcs[dataset.normalize_mth],
        hh_init="given",
        h0=(dataset.get_initial_condition().reshape(1, -1), ),
    ).build()
    layer.get_forward_weights_parameter().requires_grad = False
    if layer.force_dale_law:
        layer.get_forward_sign_parameter().requires_grad = False
        layer.get_recurrent_sign_parameter().requires_grad = True
    dale_str = f"_dale" if layer.force_dale_law else f""
    model = nt.SequentialRNN(
        layers=[layer],
        device=torch.device("cpu" if layer.force_dale_law else "cuda"),
        # device=torch.device("cuda"),
        checkpoint_folder=f"data/{dataset.filename[:-4]}/checkpoints"
                          f"_{dataset.normalize_mth}{nbu_str}"
                          f"_{layer.name}"
                          f"{dale_str}"
                          f"_sig{str(dataset.sigma).replace('.', '-')}"
                          # f"_mixtr",
    ).build()
    print(f"Model:\n{model},\nCheckpoint folder: {model.checkpoint_folder}")
    print(model_summary(model, (1, dataset.n_units)))
    model, history, trainer = train_with_curbd(
        model, dataset,
        force_overwrite=True,
        n_iterations=1_000 if model.device.type == "cpu" else 300,
        lr=1.0,
        **{
            "learning_algorithm":
                TBPTTHook(
                    params=nt.utils.filter_parameters(list(model.parameters()), requires_grad=True),
                    backward_time_steps=1,
                    optim_time_steps=1,
                    default_optimizer_cls=torch.optim.AdamW,
                    params_lr=1e-2,
                    grad_norm_clip_value=0.5,
                ) if layer.force_dale_law else None,
            # "callbacks": [
            #         TBPTTHook(
            #             # params=[layer.get_recurrent_sign_parameter()],
            #             params=[
            #                 p for p in
            #                 nt.utils.filter_parameters(list(model.parameters()), requires_grad=True)
            #                 if p is not layer.get_recurrent_weights_parameter()
            #             ],
            #             backward_time_steps=1,
            #             optim_time_steps=1,
            #             default_optimizer_cls=torch.optim.AdamW,
            #             params_lr=1e-2,
            #         )
            #     ],  # if not layer.force_dale_law else [],
            "use_lr_scheduler": layer.force_dale_law,
            "lr_schedule_start": 0.0,
        },
    )
    # model, history, trainer = train_with_curbd(
    #     model, dataset,
    #     force_overwrite=True,
    #     n_iterations=100 if model.device.type == "cpu" else 300,
    #     lr=1.0,
    #     **{
    #         "learning_algorithm":
    #             nt.TBPTT(
    #                 params=nt.utils.filter_parameters(list(model.parameters()), requires_grad=True),
    #                 backward_time_steps=1,
    #                 optim_time_steps=1,
    #                 default_optimizer_cls=torch.optim.AdamW,
    #                 params_lr=1e-2,
    #                 grad_norm_clip_value=0.5,
    #             ),
    #         "use_lr_scheduler": layer.force_dale_law,
    #         "lr_schedule_start": 0.0,
    #     },
    # )
    eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_pred, y_target = make_eval_prediction(model, dataset, device=eval_device)
    wc_viz_pred, wc_viz_target = complete_report(model, y_pred=y_pred, y_target=y_target)
    model, history, trainer = train_with_curbd(
        model, dataset,
        force_overwrite=False,
        n_iterations=300 if model.device.type == "cpu" else 900,
        lr=1.0,
    )
    eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_pred, y_target = make_eval_prediction(model, dataset, device=eval_device)
    wc_viz_pred, wc_viz_target = complete_report(model, y_pred=y_pred, y_target=y_target)
    history.plot(show=True)


