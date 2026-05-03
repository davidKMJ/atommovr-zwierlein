# Object for running benchmarking rounds and saving data

import math
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from atommovr.utils.errormodels import ZeroNoise, ErrorModel
from atommovr.utils.core import (
    _count_wrong_places,
    generate_random_target_configs,
    generate_random_init_configs,
    PhysicalParams,
    Configurations,
    CONFIGURATION_PLOT_LABELS,
)
from atommovr.utils.AtomArray import AtomArray
from atommovr.algorithms.Algorithm_class import Algorithm


def evaluate_moves(array: AtomArray, move_list: list):
    """
    Apply a sequence of moves to an ``AtomArray`` and collect timing stats.

    Parameters
    ----------
    array : AtomArray
        The tweezer array to update in place.
    move_list : list
        Sequence of move sets returned by an algorithm. Each element is a list of
        individual moves that can be executed in parallel.

    Returns
    -------
    AtomArray
        The updated array after all moves have been executed.
    float
        Total time taken to execute all move sets in parallel.
    list
        Two-element list ``[N_parallel_moves, N_non_parallel_moves]`` counting the
        number of parallel move sets and the total number of individual moves.
    """
    # making reference time
    t_total = 0
    N_parallel_moves = 0
    N_non_parallel_moves = 0

    # iterating through moves and updating matrix
    for _, move_set in enumerate(move_list):

        # performing the move
        [_, _], move_time = array.move_atoms(move_set)
        N_parallel_moves += 1
        N_non_parallel_moves += len(move_set)

        # calculating the time to complete the move set in parallel
        t_total += move_time

    return array, float(t_total), [N_parallel_moves, N_non_parallel_moves]


class BenchmarkingFigure:
    """

    NB: this is a placeholder class to mark an opportunity for future feature development (see CONTRIBUTING.md). It is not currently operational.

    Class that specifies plot parameters and figure types to be used in conjunction with the `Benchmarking` class.

    This class just specifies what you want to plot, to actually plot you have to pass it to an instance of the
    `Benchmarking` class and call the `plot_results()` function.

    ## Parameters
    - `y_axis_variables` (list):
        the observables to plot. Must be in ['Success rate', 'Filling fraction', 'Time', 'Wrong places #', 'Total atoms']
    - `figure_type` (str):
        The kind of figure you want to make. Options are histogram ('hist'), a plot comparing different algorithms ('scale'), or a plot comparing different target configurations for the same algorithm ('pattern').
    """

    def __init__(self, variables: list[str] | None = None, figure_type: str = "scale"):
        """Configure plotting options for benchmarking outputs.

        Parameters
        ----------
        variables : list, optional
            Observables to plot. Must be a subset of
            ``['Success rate', 'Filling fraction', 'Time', 'Wrong places #', 'Total atoms']``.
        figure_type : str, optional
            Desired plot style. Supported values are ``'hist'`` (histogram),
            ``'scale'`` (compare algorithms), and ``'pattern'`` (compare target
            configurations).

        Raises
        ------
        KeyError
            If an unsupported variable name is provided.
        """
        if variables is None:
            variables = ["Success rate"]
        for variable in variables:
            if variable not in [
                "Success rate",
                "Filling fraction",
                "Time",
                "Wrong places #",
                "Total atoms",
            ]:
                raise KeyError(
                    f"Variable '{variable}' is not recognized. The only allowed variables are the following: ['Success rate', 'Filling fraction', 'time', 'Wrong places #', 'Total atoms']."
                )
        self.y_axis_variables = variables
        self.figure_type = figure_type

    def generate_scaling_figure(
        self,
        x_axis,
        benchmarking_results,
        title,
        x_label,
        save,
        savename="Algorithm_scaling",
    ):
        """Create scaling plots comparing algorithms across system sizes.

        Parameters
        ----------
        x_axis : list or array-like
            Values for the horizontal axis (typically system sizes).
        benchmarking_results : list of dict
            Results dictionaries produced by ``Benchmarking.run`` for each algorithm.
        title : str
            Plot title.
        x_label : str
            Label for the x-axis.
        save : bool
            Whether to save the figure to disk.
        savename : str, optional
            Filename (without extension) used when ``save`` is ``True``.
        """

        # Iterate over the y-axis variables
        _, ax = plt.subplots(
            len(self.y_axis_variables), 1, figsize=(5, 5 * len(self.y_axis_variables))
        )
        for varind, y_var in enumerate(self.y_axis_variables):
            n_datapoints_added = 0
            y_axis = []

            # Iterate over the benchmarking results of each algorithm
            for algo_results in benchmarking_results:
                # If the y-axis variable is a list (e.g. filling fraction), take its average
                if type(algo_results[y_var]) is list:
                    algo_results[y_var] = np.mean(algo_results[y_var])

                if math.isnan(algo_results[y_var]):
                    raise Exception(
                        "Data to plot contains nan, indicating that something went wrong in your benchmarking. Please examine data and try again."
                    )
                y_axis.append(algo_results[y_var])

                n_datapoints_added += 1

                # If all the results of the algorithm are collected, plot the results
                if n_datapoints_added % len(x_axis) == 0:
                    try:
                        ax[varind].scatter(
                            x_axis,
                            y_axis,
                            marker="o",
                            label=algo_results["algorithm"].__class__.__name__,
                        )
                    except TypeError:
                        ax.scatter(
                            x_axis,
                            y_axis,
                            marker="o",
                            label=algo_results["algorithm"].__class__.__name__,
                        )
                    y_axis = []

            try:
                ax[varind].set_xlabel(x_label)
                ax[varind].set_ylabel(y_var.capitalize())
                ax[varind].set_title(title)
                ax[varind].legend(loc="best")
            except TypeError:
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_var.capitalize())
                ax.set_title(title)
                ax.legend(loc="best")

        if save:
            plt.savefig("./figs/" + savename)

    def generate_histogram_figure(
        self, benchmarking_results, title, x_label, save=False, savename="Histogram"
    ):
        """Create histograms for selected observables.

        Parameters
        ----------
        benchmarking_results : list of dict
            Results dictionaries produced by ``Benchmarking.run`` for each algorithm.
        title : str
            Plot title (unused but kept for API symmetry).
        x_label : str
            Label for the x-axis (unused but kept for API symmetry).
        save : bool, optional
            Whether to save the figure to disk.
        savename : str, optional
            Filename (without extension) used when ``save`` is ``True``.
        """
        hist_data = []
        algos_name = []
        _, ax = plt.subplots(
            len(self.y_axis_variables), 1, figsize=(5, 5 * len(self.y_axis_variables))
        )
        for varind, y_var in enumerate(self.y_axis_variables):

            for algo_results in benchmarking_results:
                hist_data.append(algo_results[y_var])
                algos_name.append(str(algo_results["algorithm"]))

            try:
                ax[varind].set_xlabel(y_var.capitalize())
                ax[varind].set_ylabel("Frequency")
                ax[varind].set_title(f"{y_var.capitalize()} histogram")
                ax[varind].hist(hist_data, bins=10, label=algos_name)
                ax[varind].legend()
            except TypeError:
                ax.set_xlabel(y_var.capitalize())
                ax.set_ylabel("Frequency")
                ax.set_title(f"{y_var.capitalize()} histogram")
                ax.hist(hist_data, bins=10, label=algos_name)
                ax.legend()

        if save:
            plt.savefig(f"./figs/{savename}")

    def generate_pattern_figure(
        self,
        x_axis,
        benchmarking_results,
        title,
        x_label,
        save=False,
        savename="Pattern_scaling",
    ):
        """Compare performance across different target configurations.

        Parameters
        ----------
        x_axis : list or array-like
            Values for the horizontal axis (typically system sizes).
        benchmarking_results : list of dict
            Results dictionaries for each target configuration.
        title : str
            Base title for the plots.
        x_label : str
            Label for the x-axis.
        save : bool, optional
            Whether to save the figure to disk.
        savename : str, optional
            Filename (without extension) used when ``save`` is ``True``.
        """

        _, ax = plt.subplots(
            len(self.y_axis_variables), 1, figsize=(5, 5 * len(self.y_axis_variables))
        )
        # Iterate over the y-axis variables
        for varind, y_var in enumerate(self.y_axis_variables):
            separate_pattern_flag = 0
            y_axis = []

            # Iterate over the benchmarking results of each target pattern
            for pattern_results in benchmarking_results:

                # If the y-axis variable is a list (e.g. filling fraction), take its average
                if type(pattern_results[y_var]) is list:
                    pattern_results[y_var] = np.mean(pattern_results[y_var])

                y_axis.append(pattern_results[y_var])
                separate_pattern_flag += 1

                # If all the results of the algorithm are collected, plot the results
                if separate_pattern_flag % len(x_axis) == 0:
                    try:
                        ax[varind].scatter(
                            x_axis,
                            y_axis,
                            marker="o",
                            label=CONFIGURATION_PLOT_LABELS[pattern_results["target"]],
                        )
                    except TypeError:
                        ax.scatter(
                            x_axis,
                            y_axis,
                            marker="o",
                            label=CONFIGURATION_PLOT_LABELS[pattern_results["target"]],
                        )
                    y_axis = []
            try:
                ax[varind].set_xlabel(x_label)
                ax[varind].set_ylabel(y_var.capitalize())
                ax[varind].set_title(f"{title} - {y_var.capitalize()}")
                ax[varind].legend(loc="best")
            except TypeError:
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_var.capitalize())
                ax.set_title(f"{title} - {y_var.capitalize()}")
                ax.legend(loc="best")

        if save:
            plt.savefig(f"./figs/{savename}")


# Set up the algorithms, target configurations, and system sizes
class Benchmarking:
    """
    An environment for studying the performance of rearrangement algorithms.

    Can be used to compare the scaling behavior of different algorithms, compare the time it takes for a single algorithm to prepare different target configurations, etc.

    ## Parameters
    - `algos` (list of `Algorithm` objects):
        the algorithms to compare.
    - `figure_output` (`BenchmarkingFigure`):
        an object for plotting.
    - `target_configs` (list of `Configurations` objects OR a list of np.ndarrays representing the explicit target configs.):
        the target patterns to prepare.
        IF a list of np.ndarrays, must provide targets for all system sizes; i.e. must have shape (len(sys_sizes), #targets), where #targets is the number of target configs.
    - `sys_sizes` (range):
        lengths of the square arrays that you want to look at (sqrt(N), where N is the number of tweezer sites).
    - `exp_params` (`PhysicalParams`):
        error and experimental parameters.
    - `n_shots` (int, default 100):
        number of repetitions per (algorithm or target config) per system size.
    - `n_species` (int, default 1):
        number of atomic species.
    - `check_sufficient_atoms` (bool, default True):
        if True, checks whether initial configurations have enough atoms, and regenerates new ones if not.

    ## Example Usage

    Creates an instance of the class and runs a benchmarking round.
        `instance = Benchmarking()`
        `instance.run()`
    """

    def __init__(
        self,
        algos: list[Algorithm] | None = None,
        target_configs: list[Configurations] | None = None,
        error_models_list: list[ErrorModel] | None = None,
        phys_params_list: list[PhysicalParams] | None = None,
        sys_sizes: list[int] | None = None,
        rounds_list: list[int] | None = None,
        figure_output: BenchmarkingFigure | None = None,
        n_shots: int = 100,
        n_species: int = 1,
        check_sufficient_atoms: bool = True,
    ) -> None:
        """Initialize benchmarking sweeps over algorithms, targets, and system sizes.

        Parameters
        ----------
        algos : list of Algorithm, optional
            Algorithms to benchmark.
        target_configs : list or ndarray, optional
            Target configurations as ``Configurations`` enums.
        error_models_list : list, optional
            Error models to evaluate.
        phys_params_list : list, optional
            Physical parameter sets to sweep.
        sys_sizes : list, optional
            Square lattice side lengths to test.
        rounds_list : list, optional
            Allowed numbers of rearrangement rounds per run.
        figure_output : BenchmarkingFigure, optional
            Plot configuration helper.
        n_shots : int, optional
            Number of Monte Carlo shots per configuration.
        n_species : int, optional
            Number of atomic species.
        check_sufficient_atoms : bool, optional
            If ``True``, regenerate initial states until enough atoms are loaded.

        Raises
        ------
        IndexError
            If explicit ``target_configs`` shapes do not match ``sys_sizes``.
        TypeError
            If ``target_configs`` is neither a list nor an ndarray.
        """
        if algos is None:
            algos = [Algorithm()]
        if target_configs is None:
            target_configs = [Configurations.MIDDLE_FILL]
        if error_models_list is None:
            error_models_list = [ZeroNoise()]
        if phys_params_list is None:
            phys_params_list = [PhysicalParams()]
        if sys_sizes is None:
            sys_sizes = list(range(10, 16))
        if rounds_list is None:
            rounds_list = [1]
        if figure_output is None:
            figure_output = BenchmarkingFigure()
        # initializing the sweep modules (minus target configs, see below)
        self.algos, self.n_algos = algos, len(algos)
        self.system_size_range, self.n_sizes = sys_sizes, len(sys_sizes)
        self.error_models_list, self.n_models = error_models_list, len(
            error_models_list
        )
        self.phys_params_list, self.n_parsets = phys_params_list, len(phys_params_list)
        self.rounds_list, self.n_rounds = rounds_list, len(rounds_list)

        # initializing other variables
        self.n_shots = n_shots
        self.check_sufficient_atoms = check_sufficient_atoms
        self.figure_output = figure_output
        self.tweezer_array = AtomArray(n_species=n_species)

        # initializing target configs depending on whether they were explicitly specified
        if isinstance(target_configs, list):
            self.istargetlist = True
            self.target_configs, self.n_targets = target_configs, len(target_configs)
        elif isinstance(target_configs, np.ndarray):
            self.istargetlist = False
            self.target_configs = target_configs
            self.n_targets = len(target_configs[0])
            if len(target_configs) != self.n_sizes:
                raise IndexError(
                    f"Number of system sizes {self.n_sizes} and shape of `target_configs` {np.shape(target_configs)} does not match. `target_configs` must have shape (len(sys_sizes), [number of target configs]). "
                )
        else:
            raise TypeError(
                "`target_configs` must be a list of Configuration objects or an np.ndarray."
            )

    def save(self, savename: str) -> None:
        """Persist benchmarking results to ``data/<savename>.nc``.

        Parameters
        ----------
        savename : str
            Base filename for the NetCDF output. ``.nc`` extension is optional.
        """
        if savename[-3:] == ".nc":
            savename = savename[0:-3]

        # NetCDF backends cannot serialize arbitrary Python objects (e.g.
        # Algorithm instances or per-shot Python lists), so write a temporary
        # string-cast copy while keeping in-memory results untouched.
        serializable_results = self.benchmarking_results.copy(deep=True)

        for coord_name in serializable_results.coords:
            coord = serializable_results.coords[coord_name]
            if coord.dtype == object:
                coord_values = np.vectorize(str, otypes=[str])(coord.values)
                serializable_results = serializable_results.assign_coords(
                    {coord_name: coord_values}
                )

        for data_var_name in serializable_results.data_vars:
            data_var = serializable_results[data_var_name]
            if data_var.dtype == object:
                serializable_results[data_var_name] = xr.DataArray(
                    np.vectorize(str, otypes=[str])(data_var.values),
                    dims=data_var.dims,
                    coords=data_var.coords,
                )

        serializable_results.to_netcdf(f"data/{savename}.nc")
        print(f"Benchmarking object saved to `data/{savename}.nc`")

    def load(self, loadname: str) -> None:
        """Load benchmarking results from ``data/<loadname>.nc`` into the object.

        Parameters
        ----------
        loadname : str
            Base filename for the NetCDF input. ``.nc`` extension is optional.
        """
        if loadname[-3:] == ".nc":
            loadname = loadname[0:-3]
        self.benchmarking_results = xr.open_dataset(f"data/{loadname}.nc")
        print(f"Data from `data/{loadname}.nc` loaded to `self.benchmarking_results`.")

    def load_params_from_dataset(self, dataset: xr.Dataset) -> None:
        """Import sweep parameters from an existing benchmarking dataset.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset previously produced by ``Benchmarking.run``.

        Notes
        -----
        This overwrites current algorithm, target, system size, error model, and
        physical parameter settings to match the provided dataset.
        """
        self.algos = dataset["algorithm"].values
        self.target_configs = dataset["target"].values
        self.istargetlist = True
        if isinstance(self.target_configs[0], np.ndarray):
            self.istargetlist = False
        self.system_size_range = dataset["sys size"].values
        self.error_models_list = dataset["error model"].values
        self.phys_params_list = dataset["physical params"].values
        rounds_list = dataset["num rounds"].values
        self.rounds_list = []
        for round in rounds_list:
            self.rounds_list.append(int(round))
        self.n_shots = len(dataset["filling fraction"].values[0][0][0][0][0][0])

    def set_observables(self, observables: list) -> None:
        """Set which observables should be plotted in downstream figures.

        Parameters
        ----------
        observables : list
            Subset of available metrics to place on the y-axis when plotting.
        """
        self.figure_output.y_axis_variables = observables

    def get_result_array_dims(self) -> None:
        """Update bookkeeping for result array dimensions based on current sweeps."""
        self.n_algos = len(self.algos)
        if self.istargetlist:
            self.n_targets = len(self.target_configs)
        else:
            self.n_targets = len(self.target_configs[0])
        if isinstance(self.target_configs, list) or not isinstance(
            self.target_configs[0], np.ndarray
        ):
            self.istargetlist = True
            self.n_targets = len(self.target_configs)
        elif isinstance(self.target_configs, np.ndarray):
            self.istargetlist = False
            self.n_targets = len(self.target_configs[0])
            if len(self.target_configs) != self.n_sizes:
                raise IndexError(
                    f"Number of system sizes {self.n_sizes} and shape of `target_configs` {np.shape(self.target_configs)} does not match. `target_configs` ust have shape (len(sys_sizes), [number of target configs]). "
                )
        else:
            raise TypeError(
                "`target_configs` must be a list of Configuration objects or an np.ndarray."
            )
        self.n_sizes = len(self.system_size_range)
        self.n_models = len(self.error_models_list)
        self.n_parsets = len(self.phys_params_list)
        self.n_rounds = len(self.rounds_list)

    def run(self, do_ejection: bool = False) -> None:
        """Execute benchmarking sweeps and populate ``self.benchmarking_results``.

        Parameters
        ----------
        do_ejection : bool, optional
            If ``True``, allow ejection moves when evaluating success.
        """

        # initializing result arrays
        self.get_result_array_dims()
        result_array_dims = [
            self.n_algos,
            self.n_targets,
            self.n_sizes,
            self.n_models,
            self.n_parsets,
            self.n_rounds,
        ]
        success_rate_array = np.zeros(result_array_dims, dtype="float")
        time_array = np.zeros(result_array_dims, dtype="float")
        fill_fracs_array = np.zeros(result_array_dims, dtype="object")
        wrong_places_array = np.zeros(result_array_dims, dtype="object")
        n_atoms_array = np.zeros(result_array_dims, dtype="object")
        n_targets_array = np.zeros(result_array_dims, dtype="object")
        sufficient_atom_rate = np.zeros(result_array_dims, dtype="float")

        # for xarray object
        dims = (
            "algorithm",
            "target",
            "sys size",
            "error model",
            "physical params",
            "num rounds",
        )
        if self.istargetlist:
            coord_targets = self.target_configs
        else:
            coord_targets = [f"Custom{i}" for i in range(self.n_targets)]
        coords = {
            "algorithm": self.algos,
            "target": coord_targets,
            "sys size": self.system_size_range,
            "error model": self.error_models_list,
            "physical params": self.phys_params_list,
            "num rounds": self.rounds_list,
        }

        # iterating through sweep parameters and running benchmarking rounds
        for param_ind, parset in enumerate(self.phys_params_list):
            self.tweezer_array.params = parset
            self.init_config_storage = generate_random_init_configs(
                self.n_shots,
                load_prob=self.tweezer_array.params.loading_prob,
                max_sys_size=np.max(self.system_size_range),
                n_species=self.tweezer_array.n_species,
            )
            for targ_ind in range(self.n_targets):
                target = None
                if self.istargetlist:
                    target = self.target_configs[targ_ind]
                    if target == Configurations.RANDOM:
                        self.target_config_storage = generate_random_target_configs(
                            self.n_shots,
                            targ_occup_prob=self.tweezer_array.params.target_occup_prob,
                            shape=self.tweezer_array.shape,
                        )
                for model_ind, error_model in enumerate(self.error_models_list):
                    self.tweezer_array.error_model = error_model

                    for size_ind, size in enumerate(self.system_size_range):
                        self.tweezer_array.shape = [size, size]
                        if not self.istargetlist:
                            self.tweezer_array.target = self.target_configs[
                                size_ind, targ_ind
                            ]
                        for alg_ind, algo in enumerate(self.algos):
                            for round_ind, num_rounds in enumerate(self.rounds_list):
                                (
                                    success_rate,
                                    mean_success_time,
                                    fill_fracs,
                                    wrong_places,
                                    atoms_in_arrays,
                                    atoms_in_target,
                                    sufficient_rate,
                                ) = self._run_benchmark_round(
                                    algo,
                                    do_ejection=do_ejection,
                                    pattern=target,
                                    num_rounds=num_rounds,
                                )
                                # populating result arrays
                                success_rate_array[
                                    alg_ind,
                                    targ_ind,
                                    size_ind,
                                    model_ind,
                                    param_ind,
                                    round_ind,
                                ] = success_rate
                                time_array[
                                    alg_ind,
                                    targ_ind,
                                    size_ind,
                                    model_ind,
                                    param_ind,
                                    round_ind,
                                ] = mean_success_time
                                fill_fracs_array[
                                    alg_ind,
                                    targ_ind,
                                    size_ind,
                                    model_ind,
                                    param_ind,
                                    round_ind,
                                ] = fill_fracs
                                wrong_places_array[
                                    alg_ind,
                                    targ_ind,
                                    size_ind,
                                    model_ind,
                                    param_ind,
                                    round_ind,
                                ] = wrong_places
                                n_atoms_array[
                                    alg_ind,
                                    targ_ind,
                                    size_ind,
                                    model_ind,
                                    param_ind,
                                    round_ind,
                                ] = atoms_in_arrays
                                n_targets_array[
                                    alg_ind,
                                    targ_ind,
                                    size_ind,
                                    model_ind,
                                    param_ind,
                                    round_ind,
                                ] = atoms_in_target
                                sufficient_atom_rate[
                                    alg_ind,
                                    targ_ind,
                                    size_ind,
                                    model_ind,
                                    param_ind,
                                    round_ind,
                                ] = sufficient_rate

        success_rates_da = xr.DataArray(success_rate_array, dims=dims, coords=coords)
        success_times_da = xr.DataArray(time_array, dims=dims, coords=coords)
        fill_fracs_da = xr.DataArray(fill_fracs_array, dims=dims, coords=coords)
        wrong_places_da = xr.DataArray(wrong_places_array, dims=dims, coords=coords)
        n_atoms_da = xr.DataArray(n_atoms_array, dims=dims, coords=coords)
        n_targets_da = xr.DataArray(n_targets_array, dims=dims, coords=coords)
        sufficient_atom_rate_da = xr.DataArray(
            sufficient_atom_rate, dims=dims, coords=coords
        )

        self.benchmarking_results = xr.Dataset(
            {
                "success rate": success_rates_da,
                "time": success_times_da,
                "filling fraction": fill_fracs_da,
                "wrong places": wrong_places_da,
                "n atoms": n_atoms_da,
                "n targets": n_targets_da,
                "sufficient rate": sufficient_atom_rate_da,
            }
        )

    def _run_benchmark_round(
        self, algorithm, do_ejection: bool = False, pattern=None, num_rounds=1
    ) -> tuple[float, float, list, list, list, list]:
        """Run repeated shots for a single algorithm/target/size combination.

        Parameters
        ----------
        algorithm : Algorithm
            Algorithm instance to generate moves.
        do_ejection : bool, optional
            Whether ejection moves are permitted.
        pattern : Configurations or ndarray, optional
            Target configuration when using enumerated targets.
        num_rounds : int, optional
            Maximum number of rearrangement rounds allowed per shot.

        Returns
        -------
        tuple
            ``(success_rate, mean_success_time, filling_fractions, wrong_places,
            atoms_in_arrays, atoms_in_targets, sufficient_rate)`` where
            ``filling_fractions``, ``wrong_places``, ``atoms_in_arrays``, and
            ``atoms_in_targets`` are lists per shot.

        Raises
        ------
        ValueError
            If ``num_rounds`` is non-positive or non-integer.
        """
        success_times = []
        success_flags = []
        filling_fractions = []
        wrong_places = []
        atoms_in_arrays = []
        atoms_in_targets = []
        sufficient_flags = []

        if self.istargetlist:
            if pattern != Configurations.RANDOM:
                self.tweezer_array.generate_target(
                    pattern,
                    occupation_prob=self.tweezer_array.params.loading_prob,
                    middle_size=self.tweezer_array.params.middle_size,
                )

        for shot in range(self.n_shots):
            # getting initial and final target configs
            initial_config = self.init_config_storage[shot][
                : self.tweezer_array.shape[0], : self.tweezer_array.shape[1]
            ].copy()
            self.tweezer_array.matrix = initial_config.reshape(
                [
                    self.tweezer_array.shape[0],
                    self.tweezer_array.shape[1],
                    self.tweezer_array.n_species,
                ]
            )
            if self.istargetlist:
                if pattern == Configurations.RANDOM:
                    self.tweezer_array.target = self.target_config_storage[shot][
                        : self.tweezer_array.shape[0], : self.tweezer_array.shape[1]
                    ].reshape(
                        [self.tweezer_array.shape[0], self.tweezer_array.shape[1], 1]
                    )
            if self.check_sufficient_atoms:
                # loop to ensure that the initial configuration has sufficient atoms.
                init_count = 0
                while (
                    np.sum(initial_config) < np.sum(self.tweezer_array.target)
                    and init_count < 100
                ):
                    self.tweezer_array.load_tweezers()
                    initial_config = self.tweezer_array.matrix
                    init_count += 1
                if init_count == 100:
                    print(
                        f"[WARNING] could not find initial configuration with enough atoms ({np.sum(self.tweezer_array.target)}) in target). \
                          Consider aborting run and choosing more suitable parameters. If this is intentional, however, you can turn off this check by setting `check_sufficient_atoms` to False when calling `Benchmarking()`."
                    )
            round_count = 0
            if num_rounds <= 0 or not isinstance(num_rounds, int):
                raise ValueError(
                    f"Number of rearrangement rounds (entered as {num_rounds}) cannot be 0, negative, nor a non-integer value."
                )
            while round_count < num_rounds:
                # generating and evaluating moves
                if self.tweezer_array.n_species == 1:
                    _, move_list, algo_success_flag = algorithm.get_moves(
                        self.tweezer_array, do_ejection=do_ejection
                    )
                else:
                    _, move_list, algo_success_flag = algorithm.get_moves(
                        self.tweezer_array
                    )
                t_total, _ = self.tweezer_array.evaluate_moves(move_list)
                success_flag = Algorithm.get_success_flag(
                    self.tweezer_array.matrix,
                    self.tweezer_array.target,
                    do_ejection=do_ejection,
                    n_species=self.tweezer_array.n_species,
                )
                if success_flag == 1:
                    break
                round_count += 1

            success_flags.append(success_flag)
            if success_flag:
                success_times.append(t_total)

            # calculate filling fraction
            filling_fraction_config = np.multiply(
                self.tweezer_array.matrix, self.tweezer_array.target
            )
            filling_fractions.append(
                float(
                    np.sum(filling_fraction_config) / np.sum(self.tweezer_array.target)
                )
            )

            # Identify wrong places (atoms that are not in the target configuration)
            wrong_places.append(
                _count_wrong_places(
                    self.tweezer_array.matrix, self.tweezer_array.target, do_ejection
                )
            )

            atoms_in_arrays.append(int(np.sum(self.tweezer_array.matrix)))
            atoms_in_targets.append(int(np.sum(self.tweezer_array.target)))

            if np.sum(initial_config) < np.sum(self.tweezer_array.target):
                sufficient_flags.append(False)
            else:
                sufficient_flags.append(True)

        return (
            float(np.mean(success_flags)),
            float(np.mean(success_times)),
            filling_fractions,
            wrong_places,
            atoms_in_arrays,
            atoms_in_targets,
            float(np.mean(sufficient_flags)),
        )

    def plot_results(self, save: bool = False, savename: str | None = None) -> None:
        """Dispatch plotting based on ``figure_output.figure_type``.

        Parameters
        ----------
        save : bool, optional
            Whether to save the generated plot(s).
        savename : str, optional
            Base filename to use when saving figures; defaults depend on figure type.
        """
        if self.figure_output.figure_type == "scale":
            if savename is None:
                savename = "scaling"
            self.figure_output.generate_scaling_figure(
                list(self.system_size_range),
                self.benchmarking_results,
                "Benchmarking results",
                "Array length (# atoms)",
                savename=savename,
                save=save,
            )

        elif self.figure_output.figure_type == "hist":
            if savename is None:
                savename = "histogram"
            self.figure_output.generate_histogram_figure(
                self.benchmarking_results,
                "Benchmarking results",
                "Array length (# atoms)",
            )

        elif self.figure_output.figure_type == "pattern":
            if savename is None:
                savename = "pattern"
            self.figure_output.generate_pattern_figure(
                list(self.system_size_range),
                self.benchmarking_results,
                "Benchmarking results",
                "Array length (# atoms)",
            )
