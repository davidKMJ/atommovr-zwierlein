# Parent class for all algorithms.
# For developers:
#   Feel free to use this as a template for new algorithms.
#   Each function below describes the requirements/what you need to put in it.

# Author: Nikhil Harle

import numpy as np
from atommovr.utils.AtomArray import AtomArray


class Algorithm:
    """
    Parent class for all algorithms.

    NB: The following functions are placeholders for illustrative purposes only and
    should be overwritten for your particular algorithm.

    If your algorithm can only prepare select target configurations, please list them here.

    e.g:

    Supported configurations: Middle Fill (see `atommovr.utils.core.Configurations`
    for a list of configurations).
    """

    def __init__(self):
        pass

    def __repr__(self) -> str:
        return "Insert the name of your algorithm here. This is what will show up on your benchmarking plots."

    def get_moves(
        self, atom_array, do_ejection: bool = False
    ) -> tuple[AtomArray, list, bool]:
        """
        This is the main function for the algorithm.

        ## Parameters
        **atom_array** : AtomArray
            object containing the initial configuration `atom_array.matrix`
            and the target configuration `atom_array.target`.

        **do_ejection** : bool, optional (default = False)
            argument to run an ejection subroutine(see
            `atommovr.algorithms.source.ejection.py` for the protocol).

        any other (optional!) kwargs you see fit to include :)

        ## Returns
        **config** : AtomArray
            the final configuration after all moves have been applied
            (ideally, this should just be the target configuration)

        **move_set** : list[list[Move, Move...], list[Move], ...]
            contains all the moves to transform the initial configuration into the final
            configuration.
            each list inside `move_set` is a set of moves that will be done in parallel.
            If you're confused by lists inside of lists, consider the following example:
                `small_move_list = [Move1]`
                `small_move_list1 = [Move2, Move3]`
                `small_move_list2 = [Move4, Move5]`
                `move_set = [small_move_list, small_move_list1, small_move_list2]`
            When this is read by the framework, it will first execute Move1, then will execute
            Move2 and Move3 in parallel, then after that Move4 and Move5 will be executed in
            parallel.
        **success_flag** : bool
            simple sanity check. This should be set to True if the algorithm prepares the
            final configuration and `False` if it does not. This is helpful during benchmarking.
        """
        config = AtomArray(shape=atom_array.shape, n_species=atom_array.n_species)
        move_set = []
        success_flag = False

        # your code here #

        return config, move_set, success_flag

    # Utility function common to all algorithms
    @staticmethod
    def get_success_flag(
        state: np.ndarray,
        target: np.ndarray,
        do_ejection: bool = False,
    ) -> bool:
        """
        Checks if the target configuration was prepared and returns a flag.
        """
        if np.shape(state) != np.shape(target):
            print(
                f"Mismatch in shapes {np.shape(state)} and {np.shape(target)}. Reshaping."
            )
            state = state.reshape(np.shape(target))

        # If do_ejection is True, we expect that the array is same as target.
        if do_ejection:
            return np.array_equal(state, target)
        else:
            # If do_ejection is False, we expect that the array has 1s at least where the target has 1s, but it can have more 1s (i.e. extra atoms that need to be ejected).
            return np.array_equal(np.multiply(state, target), target)
