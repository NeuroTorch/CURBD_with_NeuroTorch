"""
This file is used the same pipeline as the main.py file, but with a different dataset where you can
specify the filename of the dataset to use.
"""

import neurotorch as nt
import torch

from curbd_training import train_with_curbd, make_eval_prediction
from figures_script import complete_report
from ts_dataset import TSDataset
from utils import model_summary

if __name__ == '__main__':
    number_units = -1
    tau = 0.1
    n_time_steps = -1
    amp_in_wn = 0.01
    dt = 0.01

    dataset = TSDataset(
        n_units=number_units,
        tau=tau,
        amp_in_wn=amp_in_wn,
        dt_data=dt,
        re_hh=False,
        filename="Stimulus_data_2022_02_23_fish3_1.npy",
        smoothing_sigma=2 * 3.57,
        normalize=True,
        normalize_mth="min_max",
        normalize_by_unit=True,
    )
    print(f"Dataset:\n{dataset}")
    layer = nt.WilsonCowanLayer(
        input_size=dataset.n_units,
        output_size=dataset.n_units,
        tau=tau,
        dt=dt,
        learn_tau=True,
        force_dale_law=False,
        use_recurrent_connection=True,
        use_rec_eye_mask=False,
        forward_weights=torch.eye(dataset.n_units, dtype=torch.float32),
        forward_sign=torch.ones((dataset.n_units, 1), dtype=torch.float32),
        activation=torch.nn.Sigmoid(),
        hh_init="given",
        h0=(dataset.get_initial_condition().reshape(1, -1),),
    ).build()
    layer.get_forward_weights_parameter().requires_grad = False
    if layer.force_dale_law:
        layer.get_forward_sign_parameter().requires_grad = False
        layer.get_recurrent_sign_parameter().requires_grad = True
    dale_str = f"_dale" if layer.force_dale_law else f""
    model = nt.SequentialRNN(
        layers=[layer],
        device=torch.device("cpu" if torch.cuda.is_available() else "cuda"),
        checkpoint_folder=f"data/{dataset.filename[:-4]}/checkpoints"
                          f"_{layer.name}"
                          f"{dale_str}"
                          f"_sig{str(dataset.sigma).replace('.', '-')}"
    ).build()
    print(f"Model:\n{model},\nCheckpoint folder: {model.checkpoint_folder}")
    print(model_summary(model, (1, dataset.n_units)))
    model, history, trainer = train_with_curbd(
        model, dataset,
        force_overwrite=True,
        n_iterations=1_000 if model.device.type == "cpu" else 3_000,
        lr=1.0,
        **{
            "learning_algorithm":
                nt.TBPTT(
                    params=nt.utils.filter_parameters(list(model.parameters()), requires_grad=True),
                    backward_time_steps=1,
                    optim_time_steps=1,
                    default_optimizer_cls=torch.optim.AdamW,
                    params_lr=1e-2,
                    grad_norm_clip_value=0.5,
                    use_hooks=True,
                ) if layer.force_dale_law else None,
            "use_lr_scheduler": layer.force_dale_law,
            "lr_schedule_start": 0.0,
        },
    )
    eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_pred, y_target = make_eval_prediction(model, dataset, device=eval_device)
    wc_viz_pred, wc_viz_target = complete_report(model, y_pred=y_pred, y_target=y_target)
    history.plot(show=True)
