import os.path
from copy import deepcopy
from typing import Tuple, List, Optional

import numpy as np
import torch
import neurotorch as nt
from neurotorch import WilsonCowanCURBDLayer
from tqdm import tqdm

from curbd_training import train_with_curbd, evaluate_model
from ts_dataset import TSDataset
import gc
import pygad
from pygad import torchga


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


class GeneticModel:
    FILENAME_EXT = ".pkl"

    def __init__(
            self,
            base_model,
            eval_dataset,
            filename: Optional[str] = None,
            ga_instance_kwargs=None,
            n_tr_iterations: int = 10,
            **kwargs,
    ):
        self.base_model = base_model
        self.eval_dataset = eval_dataset
        self.p_bar = None
        self.filename = filename
        self.num_solutions = kwargs.get("sol_per_pop", kwargs.get("population_size", 10))
        self.ga_instance_kwargs = ga_instance_kwargs or {
            "num_generations": 300,
            "num_parents_mating": 5,
            "parent_selection_type": "sss",
            # "crossover_type": "single_point",
            "crossover_type": "scattered",
            # "mutation_type": "random",
            "mutation_type": "adaptive",
            "sol_per_pop": self.num_solutions,
            "mutation_percent_genes": 10,
            "mutation_probability": [0.6, 0.15],
            "keep_parents": -1,
        }
        self.torch_ga = None
        self.load_torch_ga()
        self.ga_instance = self.load_ga_instance()
        self.n_tr_iterations = n_tr_iterations
        self.dataloader = None
        self.trainer = None
        self.last_history = None
        self.make_trainer()

    @property
    def num_genes(self) -> int:
        return int(np.sum([
            p.numel()
            for p in nt.utils.filter_parameters(self.base_model.parameters(), requires_grad=True)
        ], dtype=int))

    @property
    def fitness_inputs(self) -> dict:
        out_dict = {
            "num_genes": self.num_genes,
            "init_range_low": -1.0,
            "init_range_high": 1.0,
            "gene_type": float,
        }
        return out_dict

    @property
    def filepath(self):
        if self.filename is None:
            return None
        return self.filename + self.FILENAME_EXT

    def load_torch_ga(self):
        self.torch_ga = torchga.TorchGA(model=self.base_model, num_solutions=self.num_solutions)
        return self.torch_ga

    def create_population(self) -> np.ndarray:
        model_weights_vector = self.get_model_weights_as_vector()
        population = [model_weights_vector, ]
        high = self.fitness_inputs.get("init_range_high", 1.0)
        low = self.fitness_inputs.get("init_range_low", -1.0)
        scale = (high - low) / 2
        for idx in range(self.num_solutions - 1):
            # net_weights = deepcopy(model_weights_vector) + np.random.uniform(
            #     low=self.fitness_inputs.get("init_range_low", -1.0),
            #     high=self.fitness_inputs.get("init_range_high", 1.0),
            #     size=model_weights_vector.size
            # )
            net_weights = deepcopy(model_weights_vector) + np.random.normal(
                scale=scale,
                size=model_weights_vector.size
            )
            population.append(net_weights)
        return np.asarray(population)

    def get_model_weights_as_vector(self):
        model_weights_as_vector = []
        for p in nt.utils.filter_parameters(self.base_model.parameters(), requires_grad=True):
            model_weights_as_vector.extend(nt.to_numpy(p).flatten())
        return np.array(model_weights_as_vector)

    def load_ga_instance(self):
        if self.filename is not None and os.path.exists(self.filepath):
            ga_instance = pygad.load(filename=self.filename)
        else:
            ga_instance = pygad.GA(
                fitness_func=self.fitness_func,
                # initial_population=self.torch_ga.population_weights,
                initial_population=self.create_population(),
                # mutation_num_genes=2,
                # parallel_processing=["thread", 5],
                on_start=self.on_start,
                on_fitness=self.on_fitness,
                # on_parents=on_parents,
                # on_crossover=on_crossover,
                # on_mutation=on_mutation,
                on_generation=self.on_generation,
                on_stop=self.on_stop,
                **self.fitness_inputs,
                **self.ga_instance_kwargs,
            )
        self.ga_instance = ga_instance
        return ga_instance

    def update_population_(self, population: np.ndarray) -> pygad.GA:
        self.ga_instance.population = self.ga_instance.round_genes(population)
        return self.ga_instance

    def make_trainer(self):
        if self.n_tr_iterations <= 0:
            return
        self.dataloader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=1, shuffle=False)
        callbacks = [
            nt.callbacks.early_stopping.EarlyStoppingOnNaN(metric="val_p_var"),
            nt.CheckpointManager(
                checkpoint_folder=self.base_model.checkpoint_folder,
                checkpoints_meta_path=self.base_model.checkpoints_meta_path,
                metric="val_p_var",
                minimise_metric=False,
                save_freq=-1,
                start_save_at=self.n_tr_iterations + 1,
                save_best_only=False,
            ),
            nt.BPTT(
                backward_time_steps=1,
                optim_time_steps=1,
                default_optimizer_cls=torch.optim.AdamW,
                params_lr=2e-4,
            ),
        ]
        self.trainer = nt.Trainer(
            model=self.base_model,
            callbacks=callbacks,
            metrics=[nt.metrics.RegressionMetrics(self.base_model, "p_var")],
            verbose=False,
        )

    def update_ga_instance_(self, ga_instance=None):
        self.ga_instance = ga_instance or self.ga_instance
        if self.filename is not None:
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            self.ga_instance.save(filename=self.filename)

    def update_model_from_solution_(self, model_weights_as_vector: np.ndarray):
        """
        Update the model parameters from the solution.

        :param model_weights_as_vector: Vector of parameters of length `num_genes`.
        :type model_weights_as_vector: np.ndarray like
        :return: None
        """
        vector = torch.tensor(model_weights_as_vector, dtype=torch.float32)
        for p in nt.utils.filter_parameters(self.base_model.parameters(), requires_grad=True):
            p.data = vector[:p.numel()].reshape(p.shape).clone().to(p.data.device)
            vector = vector[p.numel():]

    def fitness_func(self, ga_instance, solution, solution_idx):
        self.update_model_from_solution_(solution)
        # model_weights_dict = torchga.model_weights_as_dict(model=self.base_model, weights_vector=solution)
        # self.base_model.load_state_dict(model_weights_dict)

        if self.trainer is not None:
            self.trainer.verbose = False
            self.last_history = self.train_model(n_iterations=self.n_tr_iterations, verbose=False)

        pvar = evaluate_model(self.base_model, self.eval_dataset)
        if np.isnan(pvar):
            pvar = -99999.999
        return pvar

    def train_model(self, n_iterations=None, verbose=False):
        n_iterations = n_iterations or self.n_tr_iterations
        _old_n_tr_iterations = self.n_tr_iterations
        self.n_tr_iterations = n_iterations
        if n_iterations > 0 and self.trainer is None:
            self.make_trainer()
        self.n_tr_iterations = _old_n_tr_iterations
        if self.trainer is not None:
            self.trainer.verbose = verbose
            self.last_history = self.trainer.train(
                self.dataloader,
                self.dataloader,
                n_iterations=n_iterations,
                load_checkpoint_mode=nt.LoadCheckpointMode.LAST_ITR,
                force_overwrite=True,
                # desc=kwargs.get("desc", "Training"),
                exec_metrics_on_train=False,
                verbose=verbose,
            )
            return self.last_history
        return None

    def on_start(self, ga_instance):
        self.update_ga_instance_(ga_instance)
        self.p_bar = tqdm(
            total=ga_instance.num_generations,
            desc="Genetic Evolution",
            unit="gen",
        )

    def on_fitness(self, ga_instance, population_fitness):
        self.p_bar.set_postfix({"best_fitness": np.nanmax(population_fitness)})

    def on_generation(self, ga_instance):
        self.update_ga_instance_(ga_instance)
        self.p_bar.update(1)

    def on_stop(self, ga_instance, last_population_fitness):
        self.update_ga_instance_(ga_instance)
        self.p_bar.set_postfix({"best_fitness": np.nanmax(last_population_fitness)})
        self.p_bar.close()


if __name__ == '__main__':
    import json

    number_units = -1
    tau = 0.1
    n_time_steps = -1
    # n_time_steps = 100
    amp_in_wn = 0.01
    dt = 0.01

    dataset = TSDataset(
        n_units=number_units,
        n_time_steps=n_time_steps,
        tau=tau,
        amp_in_wn=amp_in_wn,
        dt_data=dt,
        re_hh=False,
        # filename="spikes.npy",
        filename="dff_matrix.npy",
        # smoothing_sigma=1.0,
        smoothing_sigma=2 * 3.57,
        # normalize=True,
        normalize=True,
        normalize_mth="min_max",
        normalize_by_unit=True,
    )
    print(dataset)
    device = torch.device("cuda")

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
        device=device,
    ).build()
    layer.forward_weights.requires_grad = False
    model = nt.SequentialRNN(
        layers=[layer],
        device=device,
        checkpoint_folder=f"checkpoints/genetic",
    ).build()

    gen_model = GeneticModel(
        base_model=model,
        eval_dataset=dataset,
        filename="data/ga_instance",
        n_tr_iterations=0,
    )
    print(gen_model.base_model)
    print(gen_model.ga_instance.summary())
    gen_model.train_model(300, verbose=True)
    gen_model.update_population_(gen_model.create_population())
    # gen_model.load_torch_ga()
    gen_model.ga_instance.run()
    best_solution, best_solution_fitness, best_match_idx = gen_model.ga_instance.best_solution(
        gen_model.ga_instance.last_generation_fitness
    )
    print(f"{best_solution_fitness = }, {best_solution = }, {best_match_idx = }")
    print(gen_model.ga_instance.summary())
    gen_model.ga_instance.plot_fitness()
    results = {
        "best_solution": best_solution,
        "best_solution_fitness": best_solution_fitness,
        "best_match_idx": best_match_idx,
        "ga_instance.summary": gen_model.ga_instance.summary(),
        "last_pvar": gen_model.last_history["val_p_var"][-1],
        "dataset": repr(dataset),
        "model": repr(gen_model.base_model),
    }

    try:
        with open('data/gen_results.txt', 'w+') as fp:
            fp.write(str(results))
    except Exception as e:
        print(e)

    try:
        with open('data/gen_results.json', 'w') as fp:
            json.dump(results, fp, indent=4)
    except Exception as e:
        print(e)

