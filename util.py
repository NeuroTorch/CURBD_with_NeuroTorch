import os
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import torch
import neurotorch as nt
from neurotorch import RLS
from neurotorch.modules import BaseModel
from neurotorch.utils import unpack_out_hh, recursive_detach_, list_insert_replace_at, ConnectivityConvention
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
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
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


class TBPTTHook(nt.TBPTT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forwards_hooks: List[RemovableHandle] = []

    def decorate_forwards(self):
        if self.trainer.model.training:
            if not self._forwards_decorated:
                self._initialize_original_forwards()
            self._hidden_layer_names.clear()

            for layer in self.layers:
                hook = layer.register_forward_hook(self._hidden_hook, with_kwargs=True)
                self._hidden_layer_names.append(layer.name)
                self.forwards_hooks.append(hook)

            for layer in self.output_layers:
                hook = layer.register_forward_hook(self._output_hook, with_kwargs=True)
                self.forwards_hooks.append(hook)
            self._forwards_decorated = True

    def undecorate_forwards(self):
        for hook in self.forwards_hooks:
            hook.remove()
        self._forwards_decorated = False

    def _hidden_hook(self, module, args, kwargs, output) -> None:
        t, forecasting = kwargs.get("t", None), kwargs.get("forecasting", False)
        if t is None:
            return

        out_tensor, hh = unpack_out_hh(output)
        hh = recursive_detach_(hh)
        return

    def _output_hook(self, module, args, kwargs, output) -> None:
        t, forecasting = kwargs.get("t", None), kwargs.get("forecasting", False)
        if t is None:
            return

        layer_name = module.name
        out_tensor, hh = unpack_out_hh(output)
        list_insert_replace_at(self._layers_buffer[layer_name], t % self.backward_time_steps, out_tensor)
        self._optim_counter += 1
        if len(self._layers_buffer[layer_name]) == self.backward_time_steps:
            self._backward_at_t(t, self.backward_time_steps, layer_name)
            output = recursive_detach_(output)
        if self._optim_counter >= self.optim_time_steps:
            self._make_optim_step()
        return


class RLSHook(RLS, TBPTTHook):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _output_hook(self, module, args, kwargs, output) -> None:
        t, forecasting = kwargs.get("t", None), kwargs.get("forecasting", False)
        if t is None:
            return

        layer_name = module.name
        out_tensor, hh = unpack_out_hh(output)
        list_insert_replace_at(self._layers_buffer[layer_name], t % self.backward_time_steps, out_tensor)
        if len(self._layers_buffer[layer_name]) == self.backward_time_steps:
            self._backward_at_t(t, self.backward_time_steps, layer_name)
            if self.strategy in ["grad", "jacobian", "scaled_jacobian"]:
                output = recursive_detach_(output)

    @staticmethod
    def curbd_step_method(
            inv_corr: torch.Tensor,
            post_activation: torch.Tensor,
            error: torch.Tensor,
            connectivity_convention: ConnectivityConvention = ConnectivityConvention.ItoJ,
            **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        delta_weights, delta_inv_corr = RLS.curbd_step_method(
            inv_corr=inv_corr,
            post_activation=post_activation,
            error=error,
            connectivity_convention=connectivity_convention,
            **kwargs
        )
        delta_weights_sign = torch.sign(delta_weights)
        delta_weights = delta_weights_sign * torch.sqrt(torch.abs(delta_weights))
        return delta_weights, delta_inv_corr
