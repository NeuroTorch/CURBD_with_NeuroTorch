from typing import Tuple, List

import numpy as np
import torch
import neurotorch as nt
from neurotorch import WilsonCowanCURBDLayer

from curbd_training import train_with_curbd
from ts_dataset import TSDataset
import gc


parameters_space = [
    ("filename", ["spikes.npy", "dff_matrix.npy"]),
    ("normalize_mth", ["min_max", "abs_max", "standard", "minus_one_one"]),
    ("learning_algorithm", [
        # nt.RLS,
        nt.BPTT,
        nt.TBPTT
    ]),
    ("lr", np.linspace(2e-4, 1.0, 4)),
]


def on_start(ga_instance):
    print("on_start()")


def on_fitness(ga_instance, population_fitness):
    print("on_fitness()")


def on_parents(ga_instance, selected_parents):
    print("on_parents()")


def on_crossover(ga_instance, offspring_crossover):
    print("on_crossover()")


def on_mutation(ga_instance, offspring_mutation):
    print("on_mutation()")


def on_generation(ga_instance):
    print("on_generation()")


def on_stop(ga_instance, last_population_fitness):
    print("on_stop()")


def solution_to_params(solution, parameters_space: List[Tuple[str, List]]):
    params = {}
    for i, (param_name, param_values) in enumerate(parameters_space):
        params[param_name] = param_values[solution[i] % len(param_values)]
    return params


def parameters_space_to_function_inputs(parameters_space: List[Tuple[str, List]]):
    out_dict = {
        "num_genes": len(parameters_space),
        "init_range_low": 0,
        "init_range_high": max(len(param_values) for _, param_values in parameters_space),
        "gene_type": int,
    }
    return out_dict


def fitness_func(ga_instance, solution, solution_idx):
    number_units = -1
    tau = 0.1
    n_time_steps = 1400
    # n_time_steps = 100
    amp_in_wn = 0.01
    dt = 0.01

    params = solution_to_params(solution, parameters_space)

    dataset = TSDataset(
        n_units=number_units,
        tau=tau,
        amp_in_wn=amp_in_wn,
        dt_data=dt,
        re_hh=False,
        # filename="spikes.npy",
        # filename="dff_matrix.npy",
        filename=params.get("filename", "dff_matrix.npy"),
        # smoothing_sigma=1,
        smoothing_sigma=2 * 3.57,
        normalize=True,
        # normalize_mth="min_max",
        normalize_mth=params.get("normalize_mth", "min_max"),
        normalize_by_unit=True,
    )

    layer = WilsonCowanCURBDLayer(
        input_size=dataset.n_units,
        output_size=dataset.n_units,
        tau=tau,
        dt=dt,
        learn_tau=True,
        force_dale_law=False,
        use_recurrent_connection=True,
        use_rec_eye_mask=False,
        forward_weights=torch.eye(dataset.n_units, dtype=torch.float32),
        # recurrent_weights=J,
        activation=torch.nn.Sigmoid(),
        # activation=torch.nn.Tanh(),
        hh_init="given",
        h0=(dataset.get_initial_condition().reshape(1, -1),),
    ).build()
    layer.forward_weights.requires_grad = True
    model = nt.SequentialRNN(
        layers=[layer],
        device=torch.device("cpu"),
        checkpoint_folder=f"checkpoints/{solution_idx}",
    ).build()
    # print(f"Model:\n{model}")

    lr = params.get("params_lr", params.get("lr", 1.0))
    learning_algorithm_cls = params.get("learning_algorithm", nt.RLS)
    if learning_algorithm_cls == nt.RLS:
        learning_algorithm = nt.RLS(
            params=[layer.recurrent_weights],
            delta=1.0,
            strategy="outputs",
            params_lr=lr,
            default_optimizer_cls=torch.optim.SGD,
            default_optim_kwargs={"lr": lr},
            is_recurrent=True,
        )
    elif learning_algorithm_cls == nt.BPTT:
        learning_algorithm = nt.BPTT(
            backward_time_steps=1,
            optim_time_steps=1,
            default_optimizer_cls=torch.optim.AdamW,
            params_lr=lr,
        )
    elif learning_algorithm_cls == nt.TBPTT:
        learning_algorithm = nt.TBPTT(
            backward_time_steps=1,
            optim_time_steps=1,
            default_optimizer_cls=torch.optim.AdamW,
            params_lr=lr,
        )
    else:
        raise ValueError(f"Unknown learning algorithm: {learning_algorithm_cls}")

    model, history, trainer = train_with_curbd(
        model, dataset,
        force_overwrite=True,
        use_lr_scheduler=True,
        n_iterations=10,
        learning_algorithm=learning_algorithm,
        desc=f"Solution {solution_idx}",
    )
    torch.cuda.empty_cache()
    gc.collect()
    return history["val_p_var"][-1]


if __name__ == '__main__':
    import pygad

    ga_instance = pygad.GA(
        num_generations=3,
        num_parents_mating=5,
        fitness_func=fitness_func,
        sol_per_pop=10,
        mutation_num_genes=2,
        # parallel_processing=["process", 2],
        # on_start=on_start,
        # on_fitness=on_fitness,
        # on_parents=on_parents,
        # on_crossover=on_crossover,
        # on_mutation=on_mutation,
        # on_generation=on_generation,
        # on_stop=on_stop,
        **parameters_space_to_function_inputs(parameters_space),
    )
    ga_instance.run()
    print(ga_instance.best_solution(ga_instance.last_generation_fitness))
    print(ga_instance.summary())
    ga_instance.plot_fitness()

