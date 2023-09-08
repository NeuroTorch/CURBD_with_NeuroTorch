import os
from typing import Tuple

import neurotorch as nt
from neurotorch.callbacks.convergence import ConvergenceTimeGetter

from curbd_dataset import CURBD3RegionsDataset
import torch


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
    rls_la = nt.RLS(
        params=[layer.recurrent_weights],
        delta=kwargs.get("P0", 1.0),
        strategy="outputs",
        params_lr=1.0,
        default_optimizer_cls=torch.optim.SGD,
        default_optim_kwargs={"lr": 1.0},
        is_recurrent=True,
    )
    n_iterations = kwargs.get("n_iterations", 1000)
    checkpoint_manager = nt.CheckpointManager(
        checkpoint_folder=model.checkpoint_folder,
        checkpoints_meta_path=model.checkpoints_meta_path,
        metric="val_p_var",
        minimise_metric=False,
        save_freq=max(1, int(n_iterations / 10)),
        start_save_at=1,
        save_best_only=False,
    )
    es_threshold = nt.callbacks.early_stopping.EarlyStoppingThreshold(
        metric="val_p_var", threshold=kwargs.get("early_stopping_threshold", 0.99), minimize_metric=False
    )
    es_nan = nt.callbacks.early_stopping.EarlyStoppingOnNaN(metric="val_p_var")
    es_stagnation = nt.callbacks.early_stopping.EarlyStoppingOnStagnation(metric="val_p_var", patience=10, tol=1e-4)
    convergence_timer = ConvergenceTimeGetter(
        metric="val_p_var", threshold=kwargs.get("convergence_threshold", kwargs.get("early_stopping_threshold", 0.9)),
        minimize_metric=False,
    )
    trainer = nt.Trainer(
        model,
        callbacks=[
            rls_la, checkpoint_manager, es_threshold, es_nan, es_stagnation, convergence_timer,
        ],
        metrics=[nt.metrics.RegressionMetrics(model, "p_var")],
    )
    dataloader = torch.utils.data.DataLoader(curbd_dataset, batch_size=1, shuffle=False)
    os.makedirs(f"{model.checkpoint_folder}/infos", exist_ok=True)
    with open(f"{model.checkpoint_folder}/infos/trainer_repr.txt", "w+") as f:
        f.write(repr(trainer))
    history = trainer.train(
        dataloader,
        dataloader,
        n_iterations=n_iterations,
        load_checkpoint_mode=nt.LoadCheckpointMode.LAST_ITR,
        force_overwrite=kwargs.get("force_overwrite", False),
    )
    history.plot(save_path=f"{model.checkpoint_folder}/figures/tr_history.png", show=False)
    with open(f"{model.checkpoint_folder}/infos/trainer_repr.txt", "w+") as f:
        f.write(repr(trainer))
    return model, history, trainer





