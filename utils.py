import os
from collections import OrderedDict
from typing import List, Optional, Iterable

import numpy as np
import torch
import neurotorch as nt
from neurotorch.learning_algorithms.learning_algorithm import LearningAlgorithm
from neurotorch.modules import BaseModel
from neurotorch.utils import unpack_out_hh, recursive_detach_, list_insert_replace_at
from torch.utils.hooks import RemovableHandle


def save_str_to_file(save_path: str, string: str):
    """
    Save a string to a file.

    :param save_path: The path to save the string to.
    :param string: The string to save.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w+") as f:
        f.write(string)


@torch.no_grad()
def pvar_mean_std(x, y, epsilon: float = 1e-8, negative: bool = False):
    """
    Calculate the mean and standard deviation of the P-Variance loss over the batch.

    :param x: The first input.
    :param y: The second input.
    :param epsilon: A small value to avoid division by zero.
    :param negative: If True, the negative of the P-Variance loss is returned.

    :return: The mean and standard deviation of the P-Variance loss over the batch.
    """
    import neurotorch as nt

    x, y = nt.to_tensor(x), nt.to_tensor(y)
    x_reshape, y_reshape = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
    mse_loss = torch.mean((x_reshape - y_reshape)**2, dim=-1)
    var = y_reshape.var(dim=-1)
    loss = 1 - (mse_loss / (var + epsilon))
    if negative:
        loss = -loss
    return loss.mean(), loss.std()


def model_summary(
        model: BaseModel,
        input_size=None,
        batch_size=-1,
        device="cpu",
):
    """
    Print a summary of the model.

    Note: This function is derived from https://github.com/sksq96/pytorch-summary.

    :param model:
    :param input_size:
    :param batch_size:
    :param device:
    :return:
    """
    if input_size is None:
        input_size = (1, model.input_size)

    # create properties
    summary = OrderedDict()
    hooks = []

    def register_hook(module):

        def hook(module, input, output):
            if input is None or output is None:
                return
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            output, hh = nt.utils.unpack_out_hh(output)
            input = [nt.utils.unpack_out_hh(_in)[0] for _in in input]

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            if isinstance(hh, (list, tuple)):
                summary[m_key]["hidden_shape"] = [
                    [-1] + list(o.size())[1:] for o in hh
                ]
            elif hh is not None:
                summary[m_key]["hidden_shape"] = list(hh.size())
                summary[m_key]["hidden_shape"][0] = batch_size

            params = 0
            # if hasattr(module, "weight") and hasattr(module.weight, "size"):
            #     params += torch.prod(torch.LongTensor(list(module.weight.size())))
            #     summary[m_key]["trainable"] = module.weight.requires_grad
            # if hasattr(module, "bias") and hasattr(module.bias, "size"):
            #     params += torch.prod(torch.LongTensor(list(module.bias.size())))
            for p in module.parameters():
                params += torch.prod(torch.LongTensor(list(p.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, torch.nn.Sequential)
            and not isinstance(module, torch.nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    output_str = ""
    output_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Hidden Shape", "Param #")
    output_str += line_new + "\n"
    output_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    total_hidden = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            str(summary[layer].get("hidden_shape", "None")),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        total_hidden += np.prod(summary[layer].get("hidden_shape", [0]))
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        output_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    output_str += "================================================================" + "\n"
    output_str += "Total params: {0:,}".format(total_params) + "\n"
    output_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    output_str += "Non-trainable params: {0:,}".format(total_params - trainable_params) + "\n"
    output_str += "----------------------------------------------------------------" + "\n"
    output_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    output_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    output_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    output_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    output_str += "----------------------------------------------------------------" + "\n"
    return output_str


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
        from curbd_training import make_eval_prediction
        from figures_script import complete_report

        folder = os.path.join(self.model.checkpoint_folder, self.pkl_sub_folder)
        os.makedirs(folder, exist_ok=True)
        y_pred, y_target = make_eval_prediction(self.model, self.dataset, device=self.eval_device)
        wc_viz_pred, wc_viz_target = complete_report(self.model, y_pred=y_pred, y_target=y_target)
        pickle.dump(self.model, open(f"{folder}/model.pkl", "wb"))
        pickle.dump(self.dataset, open(f"{folder}/dataset.pkl", "wb"))
        pickle.dump(wc_viz_pred, open(f"{folder}/viz_pred.pkl", "wb"))
        pickle.dump(wc_viz_target, open(f"{folder}/viz_target.pkl", "wb"))

    def to_txt(self):
        # summary = model_summary(self.model, (1, self.dataset.n_units))
        folder = os.path.join(self.model.checkpoint_folder, "infos")
        os.makedirs(folder, exist_ok=True)

        save_str_to_file(f"{folder}/model_repr.txt", repr(self.model))
        save_str_to_file(f"{folder}/dataset_repr.txt", repr(self.dataset))
        save_str_to_file(f"{folder}/trainer_repr.txt", repr(self.trainer))
        save_str_to_file(f"{folder}/trainer_state_repr.txt", repr(self.trainer.state))
        # save_str_to_file(f"{folder}/model_summary.txt", summary)

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
            curr_postfix["epoch"] = f"{trainer.state.epoch + 1}/{trainer.state.n_epochs}"
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
