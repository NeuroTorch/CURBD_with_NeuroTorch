from typing import Tuple

import neurotorch as nt
import numpy as np
import torch
from neurotorch.callbacks.convergence import ConvergenceTimeGetter
from neurotorch.callbacks.lr_schedulers import LRSchedulerOnMetric

from curbd_dataset import CURBD3RegionsDataset
from utils import SaveObjsCallback


def train_with_curbd(
        model: nt.SequentialRNN, curbd_dataset: CURBD3RegionsDataset, **kwargs
) -> Tuple[nt.SequentialRNN, nt.TrainingHistory, nt.Trainer]:
    """
    Train a model with the CURBD dataset.

    :param model: The model to train.
    :param curbd_dataset: The CURBD dataset.
    :param kwargs: Additional arguments.
    :return: The trained model and the trainer.
    """
    layer = model.get_layer()
    la = kwargs.get("learning_algorithm", None)
    lr = kwargs.get("params_lr", kwargs.get("lr", 1.0))
    if la is None:
        la = nt.RLS(
            params=[layer.get_recurrent_weights_parameter()],
            # params=nt.utils.filter_parameters(list(model.parameters()), requires_grad=True),
            delta=kwargs.get("P0", 1.0),
            strategy="outputs",
            params_lr=lr,
            default_optimizer_cls=torch.optim.SGD,
            # default_optimizer_cls=torch.optim.AdamW,
            default_optim_kwargs={"lr": lr},
            is_recurrent=True,
            # device=torch.device("cpu"),
            use_hooks=True,
        )
    if kwargs.get("use_lr_scheduler", False):
        lr_scheduler = LRSchedulerOnMetric(
            'val_p_var',
            metric_schedule=np.linspace(kwargs.get("lr_schedule_start", 0.80), 1.0, 100),
            min_lr=[2e-4],
            retain_progress=True,
            priority=la.priority + 1,
        )
    else:
        lr_scheduler = None
    n_iterations = kwargs.get("n_iterations", 1_000)
    checkpoint_manager = nt.CheckpointManager(
        checkpoint_folder=model.checkpoint_folder,
        checkpoints_meta_path=model.checkpoints_meta_path,
        metric="val_p_var",
        minimise_metric=False,
        # save_freq=max(1, int(n_iterations / 10)),
        save_freq=1,
        start_save_at=0,
        save_best_only=True,
    )
    es_threshold = nt.callbacks.early_stopping.EarlyStoppingThreshold(
        metric="val_p_var", threshold=kwargs.get("early_stopping_threshold", 0.99), minimize_metric=False
    )
    es_nan = nt.callbacks.early_stopping.EarlyStoppingOnNaN(metric="val_p_var")
    es_stagnation = nt.callbacks.early_stopping.EarlyStoppingOnStagnation(metric="val_p_var", patience=50, tol=1e-6)
    convergence_timer = ConvergenceTimeGetter(
        metric="val_p_var", threshold=kwargs.get("convergence_threshold", kwargs.get("early_stopping_threshold", 0.9)),
        minimize_metric=False,
    )
    save_objs = SaveObjsCallback(model, curbd_dataset)
    callbacks = [
        la,
        checkpoint_manager,
        es_threshold,
        es_nan,
        es_stagnation,
        convergence_timer,
        save_objs,
        *kwargs.get("callbacks", []),
    ]
    # if len([callback for callback in callbacks if isinstance(callback, LearningAlgorithm)]) > 1:
    #     callbacks.append(MixTrainingLearningAlgorithm(callbacks))
    if lr_scheduler is not None:
        callbacks.append(lr_scheduler)
    trainer = nt.Trainer(
        model,
        callbacks=callbacks,
        metrics=[nt.metrics.RegressionMetrics(model, "p_var")],
    )
    print(f"Trainer:\n{trainer}")
    dataloader = torch.utils.data.DataLoader(curbd_dataset, batch_size=1, shuffle=False)
    history = trainer.train(
        dataloader,
        dataloader,
        n_iterations=n_iterations,
        load_checkpoint_mode=nt.LoadCheckpointMode.LAST_ITR,
        force_overwrite=kwargs.get("force_overwrite", False),
        desc=kwargs.get("desc", "Training"),
    )
    history.plot(save_path=f"{model.checkpoint_folder}/figures/tr_history.png", show=False)
    return model, history, trainer


@torch.no_grad()
def make_eval_prediction(
        model,
        dataset,
        load_checkpoint: bool = True,
        device: torch.device = None,
):
    if load_checkpoint:
        try:
            model.load_checkpoint(load_checkpoint_mode=nt.LoadCheckpointMode.BEST_ITR, verbose=False)
        except:
            try:
                model.load_checkpoint(load_checkpoint_mode=nt.LoadCheckpointMode.LAST_ITR, verbose=False)
            except:
                pass
    initial_device = model.device
    if device is not None:
        model.to(device)
    model.eval()
    x, y = dataset[0]
    with torch.no_grad():
        try:
            out = model.get_prediction_trace(nt.to_tensor(x).unsqueeze(0))
        except torch.cuda.OutOfMemoryError:
            model.to(torch.device("cpu"))
            out = model.get_prediction_trace(nt.to_tensor(x).unsqueeze(0))
        pred_ts = nt.to_numpy(nt.utils.maybe_unpack_singleton_dict(out))
    model.to(initial_device)
    return pred_ts, np.expand_dims(nt.to_numpy(y), axis=0)


@torch.no_grad()
def evaluate_model(
        model,
        dataset,
        load_checkpoint: bool = True,
        device: torch.device = None,
):
    y_pred, y_target = make_eval_prediction(model, dataset, load_checkpoint, device)
    p_var = nt.metrics.PVarianceLoss()(
        y_pred, y_target
    ).item()
    return p_var
