import os
from typing import Tuple, Iterable, Optional, List

import neurotorch as nt
import numpy as np
from neurotorch.callbacks.convergence import ConvergenceTimeGetter
from neurotorch.callbacks.lr_schedulers import LRSchedulerOnMetric
import pythonbasictools as pbt
import threading as th

from neurotorch.learning_algorithms.learning_algorithm import LearningAlgorithm

from curbd_dataset import CURBD3RegionsDataset
from figures_script import complete_report
from util import save_str_to_file, model_summary, RLSHook
import torch


class SayMetricValue(nt.callbacks.BaseCallback):
    DEFAULT_PRIORITY = nt.callbacks.BaseCallback.DEFAULT_LOW_PRIORITY

    def __init__(self, metric: str, freq: int = 10, cache_file: str = "./.cache/gtts/tts.mp3"):
        super().__init__()
        self.metric = metric
        self.freq = freq
        self.cache_file = cache_file

    def on_iteration_end(self, trainer: nt.Trainer, **kwargs):
        if trainer.state.iteration % self.freq == 0:
            audio_metric_name = self.metric.replace("_", " ")
            audio_value = f"{trainer.training_history[self.metric][-1]:.2f}"
            try:
                # run this try in a thread to avoid blocking the training
                th.Thread(
                    target=pbt.simple_tts.say,
                    args=(
                        f"The metric {audio_metric_name} is at a value of {audio_value} "
                        f"for the iteration {trainer.state.iteration}.",
                    ),
                    kwargs={
                        "cache_file": self.cache_file,
                        "delay": 0.1,
                        "raise_error": False,
                        "rm_cache_file": True,
                        "n_trials": 3,
                    },
                ).start()
            except Exception as e:
                pass
            finally:
                if self.cache_file is not None:
                    if os.path.exists(self.cache_file):
                        os.remove(self.cache_file)

    def extra_repr(self) -> str:
        return f"metric={self.metric}, freq={self.freq}"


class SaveObjsCallback(nt.callbacks.BaseCallback):
    DEFAULT_PRIORITY = nt.callbacks.BaseCallback.DEFAULT_LOW_PRIORITY

    def __init__(
            self,
            model,
            dataset,
            pkl_sub_folder="pickles",
            repr_sub_folder="infos",
            eval_device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.pkl_sub_folder = pkl_sub_folder
        self.repr_sub_folder = repr_sub_folder
        self.eval_device = eval_device

    def to_pickle(self):
        import pickle

        folder = os.path.join(self.model.checkpoint_folder, self.pkl_sub_folder)
        os.makedirs(folder, exist_ok=True)
        y_pred, y_target = make_eval_prediction(self.model, self.dataset, device=self.eval_device)
        wc_viz_pred, wc_viz_target = complete_report(self.model, y_pred=y_pred, y_target=y_target)
        pickle.dump(self.model, open(f"{folder}/model.pkl", "wb"))
        pickle.dump(self.dataset, open(f"{folder}/dataset.pkl", "wb"))
        pickle.dump(wc_viz_pred, open(f"{folder}/viz_pred.pkl", "wb"))
        pickle.dump(wc_viz_target, open(f"{folder}/viz_target.pkl", "wb"))

    def to_txt(self):
        summary = model_summary(self.model, (1, self.dataset.n_units))
        folder = os.path.join(self.model.checkpoint_folder, "infos")
        os.makedirs(folder, exist_ok=True)

        save_str_to_file(f"{folder}/model_repr.txt", repr(self.model))
        save_str_to_file(f"{folder}/dataset_repr.txt", repr(self.dataset))
        save_str_to_file(f"{folder}/trainer_repr.txt", repr(self.trainer))
        save_str_to_file(f"{folder}/trainer_state_repr.txt", repr(self.trainer.state))
        save_str_to_file(f"{folder}/model_summary.txt", summary)
        
    def try_save_all(self):
        try:
            self.to_pickle()
        except:
            pass
        try:
            self.to_txt()
        except:
            pass

    def start(self, trainer: nt.Trainer, **kwargs):
        super().start(trainer, **kwargs)
        self.try_save_all()

    def close(self, trainer: nt.Trainer, **kwargs):
        super().close(trainer, **kwargs)
        self.try_save_all()

    def __del__(self):
        self.try_save_all()
        super().__del__()

    def extra_repr(self) -> str:
        return f"pkl_sub_folder={self.pkl_sub_folder}, repr_sub_folder={self.repr_sub_folder}"


class MixTrainingLearningAlgorithm(LearningAlgorithm):
    DEFAULT_PRIORITY = nt.callbacks.BaseCallback.DEFAULT_HIGH_PRIORITY

    def __init__(
            self,
            learning_algorithms: Optional[Iterable[LearningAlgorithm]] = None,
            n_epochs_per_la: Optional[int] = 10,
    ):
        super().__init__(save_state=False)
        self._learning_algorithms = learning_algorithms
        self.n_epochs_per_la = n_epochs_per_la

    def filter_learning_algorithms(self, callbacks: Iterable[nt.callbacks.BaseCallback]) -> List[LearningAlgorithm]:
        la_list = [callback for callback in callbacks if isinstance(callback, LearningAlgorithm)]
        # make sure that self is not in the list
        la_list = [la for la in la_list if not isinstance(la, MixTrainingLearningAlgorithm)]
        if self in la_list:
            la_list.remove(self)
        return la_list

    def start(self, trainer: nt.Trainer, **kwargs):
        super().start(trainer, **kwargs)
        if self._learning_algorithms is None:
            self._learning_algorithms = trainer.callbacks
        self._learning_algorithms = self.filter_learning_algorithms(self._learning_algorithms)
        for la in self._learning_algorithms:
            la.start(trainer, **kwargs)

        trainer.update_state_(
            n_epochs=max(trainer.state.n_epochs, self.n_epochs_per_la * len(self._learning_algorithms))
        )

        # remove the learning algorithms from the trainer's callbacks
        self.add_learning_algorithms_to_trainer(trainer, **kwargs)

    def remove_learning_algorithms_from_trainer(self, trainer: nt.Trainer, **kwargs):
        callbacks_to_remove = self.filter_learning_algorithms(trainer.callbacks)
        for la in callbacks_to_remove:
            trainer.callbacks.remove(la)

    def add_learning_algorithms_to_trainer(self, trainer: nt.Trainer, **kwargs):
        for la in self._learning_algorithms:
            if la not in trainer.callbacks:
                trainer.callbacks.append(la)

    def on_epoch_begin(self, trainer, **kwargs):
        super().on_iteration_begin(trainer, **kwargs)
        self.remove_learning_algorithms_from_trainer(trainer, **kwargs)
        # curr_la = self._learning_algorithms[trainer.state.epoch % len(self._learning_algorithms)]
        la_idx = (trainer.state.epoch // self.n_epochs_per_la) % len(self._learning_algorithms)
        curr_la = self._learning_algorithms[la_idx]
        trainer.callbacks.append(curr_la)
        curr_la.on_epoch_begin(trainer, **kwargs)
        p_bar = trainer.state.objects.get("p_bar", None)
        if p_bar is not None:
            if p_bar.postfix is None:
                curr_postfix = {}
            else:
                curr_postfix = {
                    p_str.split("=")[0].strip(): p_str.split("=")[1].strip()
                    for p_str in p_bar.postfix.split(",")
                    if "=" in p_str
                }
            curr_postfix["la"] = getattr(curr_la, "name", curr_la.__class__.__name__)
            curr_postfix["epoch"] = f"{trainer.state.epoch+1}/{trainer.state.n_epochs}"
            p_bar.set_postfix(curr_postfix)

    def on_epoch_end(self, trainer, **kwargs):
        super().on_iteration_end(trainer, **kwargs)
        self.remove_learning_algorithms_from_trainer(trainer, **kwargs)
        self.add_learning_algorithms_to_trainer(trainer, **kwargs)

    def extra_repr(self) -> str:
        if self._learning_algorithms is None:
            return f"learning_algorithms=None"
        learning_algorithms = self.filter_learning_algorithms(self._learning_algorithms)
        la_names = [getattr(la, "name", la.__class__.__name__) for la in learning_algorithms]
        return f"learning_algorithms={la_names}, n_epochs_per_la={self.n_epochs_per_la}"


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
        la = RLSHook(
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
        # SayMetricValue("val_p_var", freq=10),
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


