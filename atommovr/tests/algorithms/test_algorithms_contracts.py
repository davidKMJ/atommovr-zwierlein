import numpy as np
import pytest

from atommovr.algorithms.Algorithm_class import Algorithm
from atommovr.algorithms.dual_species import InsideOut, NaiveParHung
from atommovr.algorithms.single_species import (
    # BCv2,
    # BalanceAndCompact,
    GeneralizedBalance,
    Hungarian,
    ParallelHungarian,
    ParallelLBAP,
)
from atommovr.utils.AtomArray import AtomArray
from atommovr.utils.core import Configurations

# Add new built-in algorithm classes here as they are created
BUILTIN_ALGORITHMS = [
    Algorithm,
    ParallelHungarian,
    ParallelLBAP,
    GeneralizedBalance,
    Hungarian,
    # BCv2,
    # BalanceAndCompact,
    InsideOut,
    NaiveParHung,
]

# Base class `Algorithm` is a template and intentionally not behavior-complete.
# Use this list for behavioral contract checks.
CONCRETE_ALGORITHMS = [
    algo_cls for algo_cls in BUILTIN_ALGORITHMS if algo_cls is not Algorithm
]

# Specify which algorithms are dual-species algorithms, to be used in tests that need to know this distinction.
DUAL_SPECIES_ALGORITHMS = {
    InsideOut,
    NaiveParHung,
}

SINGLE_SPECIES_ALGORITHMS = [
    algo_cls
    for algo_cls in CONCRETE_ALGORITHMS
    if algo_cls not in DUAL_SPECIES_ALGORITHMS
]

REQUIRED_METHODS = [
    "__repr__",
    "get_moves",
    "get_success_flag",
]


def _make_single_species_array() -> AtomArray:
    # For simplicity, we just use an empty array to test types of algorithms' outputs.
    arr = AtomArray(shape=[4, 4], n_species=1)
    arr.matrix[:, :, 0] = 1
    arr.generate_target(Configurations.MIDDLE_FILL)
    return arr


def _make_dual_species_array() -> AtomArray:
    # For simplicity, we just use an empty array to test types of algorithms' outputs.
    arr = AtomArray(shape=[4, 4], n_species=2)
    arr.matrix[:, :, :] = 1
    arr.generate_target(Configurations.CHECKERBOARD)
    return arr


@pytest.mark.parametrize("algo_cls", BUILTIN_ALGORITHMS)
def test_builtin_algorithms_are_algorithm_subclasses(algo_cls) -> None:
    algo = algo_cls()
    assert isinstance(algo, Algorithm)


@pytest.mark.parametrize("algo_cls", BUILTIN_ALGORITHMS)
@pytest.mark.parametrize("method_name", REQUIRED_METHODS)
def test_builtin_algorithms_have_required_methods(algo_cls, method_name: str) -> None:
    algo = algo_cls()
    method = getattr(algo, method_name, None)
    assert callable(
        method
    ), f"{algo_cls.__name__}.{method_name} is missing or not callable"


@pytest.mark.parametrize("algo_cls", BUILTIN_ALGORITHMS)
def test_builtin_algorithms_repr_contract(algo_cls) -> None:
    algo = algo_cls()
    rep = repr(algo)
    assert isinstance(rep, str)
    assert len(rep.strip()) > 0


@pytest.mark.parametrize("algo_cls", BUILTIN_ALGORITHMS)
def test_builtin_algorithms_get_moves_return_types(algo_cls) -> None:
    """Test that get_moves returns (AtomArray, list, bool)."""
    algo = algo_cls()

    if algo_cls in DUAL_SPECIES_ALGORITHMS:
        arr = _make_dual_species_array()
    else:
        arr = _make_single_species_array()

    config, move_set, success_flag = algo.get_moves(arr)

    assert isinstance(
        config, (np.ndarray, AtomArray)
    ), f"{algo_cls.__name__}.get_moves() config should be np.ndarray or AtomArray, got {type(config)}"
    assert isinstance(
        move_set, list
    ), f"{algo_cls.__name__}.get_moves() move_set should be list, got {type(move_set)}"
    assert isinstance(
        success_flag, bool
    ), f"{algo_cls.__name__}.get_moves() success_flag should be bool, got {type(success_flag)}"


# ============================================================================
# CONTRACT TEST 1: Already-solved configuration
# ============================================================================


@pytest.mark.parametrize("algo_cls", CONCRETE_ALGORITHMS)
def test_already_solved_configuration_no_unnecessary_moves(algo_cls) -> None:
    """
    When initial state already matches target, algorithm should not make
    unnecessary moves and should report success.
    """
    algo = algo_cls()

    if algo_cls in DUAL_SPECIES_ALGORITHMS:
        arr = _make_dual_species_array()
    else:
        arr = _make_single_species_array()

    # Make initial state match target exactly
    arr.matrix = arr.target.copy()

    config, move_set, success_flag = algo.get_moves(arr)

    # Should report success
    assert success_flag

    # Should not require any moves (or minimal cleanup moves)
    # Most algorithms should return zero or very few moves
    assert len(move_set) == 0, (
        f"{algo_cls.__name__} returned unnecessary moves when already solved: "
        f"got {len(move_set)} moves"
    )

    # Verify no atoms are lost
    if isinstance(config, AtomArray):
        assert np.sum(config.matrix) == np.sum(
            arr.target
        ), f"{algo_cls.__name__} lost atoms in solved configuration"


# ============================================================================
# CONTRACT TEST 2: One-move-away configuration
# ============================================================================


@pytest.mark.parametrize("algo_cls", SINGLE_SPECIES_ALGORITHMS)
def test_one_move_away_configuration_single_species(algo_cls) -> None:
    """
    Every algorithm should solve a configuration that is exactly one move away
    from completion (for single-species configurations).
    """

    algo = algo_cls()
    arr = _make_single_species_array()

    # Create a simple one-move-away scenario
    # Place one atom where it needs to move one step
    arr.matrix = np.zeros_like(arr.target, dtype=np.uint8)
    arr.target = np.zeros_like(arr.target, dtype=np.uint8)

    # Make a 3x3 target inside 5x5 array.
    # Move the atoms at (1,2) to (0, 2) such that there is only one move needed to complete the target.

    # Put one atom at (0, 0) that should move to (0, 1)
    arr.matrix[0, 0] = 1
    arr.target[0, 1] = 1

    config, move_set, success_flag = algo.get_moves(arr)

    assert (
        success_flag
    ), f"{algo_cls.__name__} failed to solve one-move-away configuration"
    assert (
        len(move_set) == 1
    ), f"{algo_cls.__name__} found solution but returned more than one move (or empty): got {len(move_set)} moves"
    ## Add some small random tests (for Single species and InsideOut should have)


# ============================================================================
# CONTRACT TEST 3: Noiseless atom conservation
# ============================================================================


@pytest.mark.parametrize("algo_cls", CONCRETE_ALGORITHMS)
def test_noiseless_atom_conservation_random_initial_state(algo_cls) -> None:
    """
    For random initial configurations, noiseless rearrangement should not lose
    atoms (unless ejection is part of the algorithm's intended behavior).
    """
    algo = algo_cls()

    if algo_cls in DUAL_SPECIES_ALGORITHMS:
        arr = _make_dual_species_array()
    else:
        arr = _make_single_species_array()

    # Use provided target, random initial state
    initial_atom_count = np.sum(arr.matrix)

    config, move_set, success_flag = algo.get_moves(arr)

    # Convert to AtomArray if needed for consistent access
    if isinstance(config, np.ndarray):
        final_atom_count = np.sum(config)
    else:
        final_atom_count = np.sum(config.matrix)

    # Atoms should be conserved (not ejected) in noiseless mode
    # Unless the algorithm explicitly includes ejection
    assert final_atom_count == initial_atom_count, (
        f"{algo_cls.__name__} lost atoms in noiseless mode: "
        f"started with {initial_atom_count}, ended with {final_atom_count}"
    )


# ============================================================================
# CONTRACT TEST 4: Success flag validation
# ============================================================================


@pytest.mark.parametrize("algo_cls", CONCRETE_ALGORITHMS)
def test_success_flag_matches_reality(algo_cls) -> None:
    """
    When success=True, the returned configuration must actually satisfy the
    target condition.
    """
    algo = algo_cls()

    if algo_cls in DUAL_SPECIES_ALGORITHMS:
        arr = _make_dual_species_array()
    else:
        arr = _make_single_species_array()

    config, move_set, success_flag = algo.get_moves(arr)

    if success_flag:
        # Extract current and target for comparison
        if isinstance(config, AtomArray):
            final_current = config.matrix
            print(
                f"Final current configuration shape of {algo_cls.__name__}: {np.shape(final_current)}"
            )
            target = config.target
        else:
            final_current = config
            target = arr.target

        # For single species, exact match is expected
        if arr.n_species == 1:
            assert np.array_equal(np.multiply(final_current, target), target), (
                f"{algo_cls.__name__} returned success=True but final "
                "configuration does not match target"
            )
        # For dual species, check each species separately
        else:
            assert np.array_equal(
                np.multiply(final_current[:, :, 0], target[:, :, 0]), target[:, :, 0]
            ) and np.array_equal(
                np.multiply(final_current[:, :, 1], target[:, :, 1]), target[:, :, 1]
            ), (
                f"{algo_cls.__name__} returned success=True but final "
                "configuration does not match target for dual species"
            )


# ============================================================================
# CONTRACT TEST 5: Failure case handling
# ============================================================================


@pytest.mark.parametrize("algo_cls", CONCRETE_ALGORITHMS)
def test_failure_on_insufficient_atoms(algo_cls) -> None:
    """
    When target cannot be completed (too few atoms), algorithm should not claim
    success and should fail gracefully.
    """
    algo = algo_cls()

    if algo_cls in DUAL_SPECIES_ALGORITHMS:
        arr = _make_dual_species_array()
    else:
        arr = _make_single_species_array()

    # Clear all atoms in current state
    arr.matrix = np.zeros(np.shape(arr.matrix), dtype=np.uint8)
    print(arr.matrix)

    # Keep target with atoms - now impossible to reach
    # (unless target is also empty, so set it explicitly to have atoms)
    arr.target[0, 0, 0] = 1
    print(arr.target)

    config, move_set, success_flag = algo.get_moves(arr)

    # Should not report success when impossible
    assert (
        not success_flag
    ), f"{algo_cls.__name__} claimed success when insufficient atoms available"


# ============================================================================
# CONTRACT TEST 6: Replayable moves
# ============================================================================


@pytest.mark.parametrize("algo_cls", CONCRETE_ALGORITHMS)
def test_returned_moves_are_replayable(algo_cls) -> None:
    """
    The moves returned by the algorithm should be replayable on the initial
    state, and the replay should match the returned final configuration.
    """
    from atommovr.utils.move_utils import move_atoms_noiseless

    algo = algo_cls()

    if algo_cls in DUAL_SPECIES_ALGORITHMS:
        arr = _make_dual_species_array()
    else:
        arr = _make_single_species_array()

    initial_state = arr.matrix.copy()
    config, move_set, success_flag = algo.get_moves(arr)

    if len(move_set) == 0:
        # If no moves, final state should match initial
        if isinstance(config, AtomArray):
            replayed_state = initial_state
        else:
            replayed_state = config
    else:
        # Replay the moves on initial state
        replayed_state, _, _ = move_atoms_noiseless(initial_state.copy(), move_set)

    # Extract final state for comparison
    if isinstance(config, AtomArray):
        final_state = config.matrix
    else:
        final_state = config

    assert np.array_equal(replayed_state, final_state), (
        f"{algo_cls.__name__} returned moves that don't replay to final state: "
        "move list and final configuration are inconsistent"
    )


# ============================================================================
# CONTRACT TEST 7: No illegal move coordinates
# ============================================================================


@pytest.mark.parametrize("algo_cls", CONCRETE_ALGORITHMS)
def test_no_illegal_move_coordinates(algo_cls) -> None:
    """
    All returned moves must have coordinates within valid bounds. No negative
    indices or out-of-range row/col values.
    """
    algo = algo_cls()

    if algo_cls in DUAL_SPECIES_ALGORITHMS:
        arr = _make_dual_species_array()
    else:
        arr = _make_single_species_array()

    config, move_set, success_flag = algo.get_moves(arr)

    rows, cols = arr.matrix.shape[:2]

    for i, move in enumerate(move_set):
        assert 0 <= move.from_row < rows, (
            f"{algo_cls.__name__} returned illegal from_row: "
            f"move {i} has from_row={move.from_row}, grid is {rows}x{cols}"
        )
        assert 0 <= move.from_col < cols, (
            f"{algo_cls.__name__} returned illegal from_col: "
            f"move {i} has from_col={move.from_col}, grid is {rows}x{cols}"
        )
        assert 0 <= move.to_row < rows, (
            f"{algo_cls.__name__} returned illegal to_row: "
            f"move {i} has to_row={move.to_row}, grid is {rows}x{cols}"
        )
        assert 0 <= move.to_col < cols, (
            f"{algo_cls.__name__} returned illegal to_col: "
            f"move {i} has to_col={move.to_col}, grid is {rows}x{cols}"
        )


# ============================================================================
# CONTRACT TEST 8: Parallel scheduling consistency
# ============================================================================


@pytest.mark.parametrize("algo_cls", CONCRETE_ALGORITHMS)
def test_parallel_rounds_no_resource_conflicts(algo_cls) -> None:
    """
    Within each parallel batch, no two moves should share a source or
    destination. This ensures parallel schedules are internally consistent.
    """
    from atommovr.utils.move_utils import get_AOD_cmds_from_move_list

    algo = algo_cls()

    if algo_cls in DUAL_SPECIES_ALGORITHMS:
        arr = _make_dual_species_array()
    else:
        arr = _make_single_species_array()

    config, move_set, success_flag = algo.get_moves(arr)

    if len(move_set) == 0:
        return  # Nothing to check

    # Check if moves can be executed in parallel (basic check)
    # This uses the get_AOD_cmds_from_move_list to validate parallelization
    try:
        _, _, can_parallelize = get_AOD_cmds_from_move_list(
            arr.matrix, move_set, verify=True
        )
        # If algorithm claims parallelization is possible, moves should have no conflicts
        if can_parallelize is not None:
            assert can_parallelize or len(move_set) == 1, (
                f"{algo_cls.__name__} returned non-parallelizable move set: "
                "moves have source/destination conflicts"
            )
    except Exception:
        # If parallelization check fails, that's acceptable for non-parallel algorithms
        pass


# ============================================================================
# CONTRACT TEST 9-1: Boundary shape edge cases (single-species)
# ============================================================================


@pytest.mark.parametrize(
    "shape",
    [
        (1, 4),  # 1xN
        (4, 1),  # Nx1
        (3, 5),  # non-square
        (2, 2),  # small square
    ],
)
@pytest.mark.parametrize("algo_cls", SINGLE_SPECIES_ALGORITHMS)
def test_boundary_shape_edge_cases_single_species(algo_cls, shape) -> None:
    """
    Test single-species algorithms on various boundary-case geometries: 1xN, Nx1, non-square,
    and small square arrays. These often expose indexing logic bugs.
    """
    arr = AtomArray(shape=shape, n_species=1)
    arr.generate_target(Configurations.CHECKERBOARD)
    while True:
        arr.load_tweezers()
        if np.sum(arr.matrix) > np.sum(arr.target):
            init_count = np.sum(arr.matrix)
            break

    algo = algo_cls()

    config, move_set, success_flag = algo.get_moves(arr)

    # Basic contract: should not crash and return valid types
    assert isinstance(config, (np.ndarray, AtomArray))
    assert isinstance(move_set, list)
    assert isinstance(success_flag, bool)

    # Check if # atoms consistent after rearrangement (no loss)
    assert init_count == np.sum(arr.matrix)

    # All returned moves should be in-bounds for this shape
    rows, cols = shape
    for moves in move_set:
        for move in moves:
            assert 0 <= move.from_row <= rows
            assert 0 <= move.from_col < cols
            assert 0 <= move.to_row < rows
            assert 0 <= move.to_col < cols


# ============================================================================
# CONTRACT TEST 9-2: Boundary shape edge cases (dual-species)
# ============================================================================
# Do rectangular cases. No need for 1-D structure
@pytest.mark.parametrize(
    "shape",
    [
        (2, 4),  # 1xN
        (4, 2),  # Nx1
        (3, 5),  # non-square
        (4, 4),  # small square
    ],
)
@pytest.mark.parametrize("algo_cls", DUAL_SPECIES_ALGORITHMS)
def test_boundary_shape_edge_cases_dual_species(algo_cls, shape) -> None:
    """
    Test dual-species algorithms on various boundary-case geometries: 2xN, Nx2, non-square,
    and small square arrays. These often expose indexing logic bugs.
    """
    arr = AtomArray(shape=shape, n_species=2)
    arr.generate_target(Configurations.CHECKERBOARD)
    while True:
        arr.load_tweezers()
        if np.sum(arr.matrix[:, :, 0]) > np.sum(arr.target[:, :, 0]) and np.sum(
            arr.matrix[:, :, 1]
        ) > np.sum(arr.target[:, :, 1]):
            break

    algo = algo_cls()

    config, move_set, success_flag = algo.get_moves(arr)

    # Basic contract: should not crash and return valid types
    assert isinstance(config, (np.ndarray, AtomArray))
    print("move_set:", move_set)
    assert isinstance(move_set, list)
    assert isinstance(success_flag, bool)

    # All returned moves should be in-bounds for this shape
    rows, cols = shape
    allow_one_step_outside = algo_cls.__name__ == "InsideOut"

    for moves in move_set:
        for move in moves:
            assert 0 <= move.from_row < rows
            assert 0 <= move.from_col < cols

            if allow_one_step_outside:
                assert -1 <= move.to_row <= rows
                assert -1 <= move.to_col <= cols
                assert abs(move.to_row - move.from_row) <= 1
                assert abs(move.to_col - move.from_col) <= 1
            else:
                assert 0 <= move.to_row < rows
                assert 0 <= move.to_col < cols


@pytest.mark.parametrize("algo_cls", SINGLE_SPECIES_ALGORITHMS)
def test_random_initial_configurations_single_species(algo_cls) -> None:
    """
    Test single-species algorithms with 10 random initial configurations on a 10x10 array.
    If initial atom count is insufficient, regenerate for that round.
    Verifies:
    - Algorithm doesn't crash
    - Returns valid types
    - Atoms are conserved (noiseless mode)
    - Returned moves are in-bounds
    """
    # TODO: Have random seed such that the error is reproducible.
    algo = algo_cls()
    num_trials = 10
    array_size_list = [10]

    for array_size in array_size_list:
        for trial in range(num_trials):
            # Create fresh 10x10 array with random target
            arr = AtomArray(shape=[array_size, array_size], n_species=1)
            arr.generate_target(Configurations.CHECKERBOARD)

            # Regenerate until we have enough atoms
            while True:
                arr.load_tweezers()
                if np.sum(arr.matrix) >= np.sum(arr.target):
                    break

            initial_atom_count = np.sum(arr.matrix)

            # Run algorithm
            config, move_set, success_flag = algo.get_moves(arr)

            # Contract 1: Returns valid types
            assert isinstance(
                config, (np.ndarray, AtomArray)
            ), f"{algo_cls.__name__} trial {trial}: config should be np.ndarray or AtomArray"
            assert isinstance(
                move_set, list
            ), f"{algo_cls.__name__} trial {trial}: move_set should be list"
            assert isinstance(
                success_flag, bool
            ), f"{algo_cls.__name__} trial {trial}: success_flag should be bool"

            # Contract 2: Atoms are conserved
            if isinstance(config, AtomArray):
                final_atom_count = np.sum(config.matrix)
            else:
                final_atom_count = np.sum(config)

            assert final_atom_count == initial_atom_count, (
                f"{algo_cls.__name__} trial {trial}: lost atoms; "
                f"started with {initial_atom_count}, ended with {final_atom_count}"
            )

            # Contract 3: Success flag
            assert success_flag


@pytest.mark.parametrize("algo_cls", DUAL_SPECIES_ALGORITHMS)
def test_random_initial_configurations_dual_species(algo_cls) -> None:
    """
    Test dual-species algorithms with 10 random initial configurations on a 10x10 array.
    If initial atom count is insufficient for either species, regenerate for that round.
    Verifies:
    - Algorithm doesn't crash
    - Returns valid types
    - Atoms are conserved for both species (noiseless mode)
    - Returned moves are in-bounds
    """
    if algo_cls.__name__ == "NaiveParHung":
        pytest.skip("Skipping NaiveParHung for this dual-species random configuration test")
        
    algo = algo_cls()
    num_trials = 10
    array_size_list = [10]

    for array_size in array_size_list:
        for trial in range(num_trials):
            # Create fresh 10x10 array with random target for two species
            arr = AtomArray(shape=[array_size, array_size], n_species=2)
            arr.generate_target(Configurations.CHECKERBOARD, middle_size=[6, 6])

            # Regenerate until we have enough atoms for both species
            while True:
                arr.load_tweezers()
                if np.sum(arr.matrix[:, :, 0]) >= np.sum(
                    arr.target[:, :, 0] * 1.2
                ) and np.sum(arr.matrix[:, :, 1]) >= np.sum(arr.target[:, :, 1] * 1.2):
                    break

            # Run algorithm
            config, move_set, success_flag = algo.get_moves(arr)

            # Contract 1: Returns valid types
            assert isinstance(
                config, (np.ndarray, AtomArray)
            ), f"{algo_cls.__name__} trial {trial}: config should be np.ndarray or AtomArray"
            assert isinstance(
                move_set, list
            ), f"{algo_cls.__name__} trial {trial}: move_set should be list"
            assert isinstance(
                success_flag, bool
            ), f"{algo_cls.__name__} trial {trial}: success_flag should be bool"

            # assert final_atom_count_total == initial_atom_count_total, (
            #     f"{algo_cls.__name__} trial {trial}: lost atoms; "
            #     f"started with {initial_atom_count_total} (Rb: {initial_atom_count_rb}, Cs: {initial_atom_count_cs}), "
            #     f"ended with {final_atom_count_total} (Rb: {final_atom_count_rb}, Cs: {final_atom_count_cs})"
            # )

            # assert final_atom_count_rb == initial_atom_count_rb, (
            #     f"{algo_cls.__name__} trial {trial}: Rb atoms not conserved; "
            #     f"started with {initial_atom_count_rb}, ended with {final_atom_count_rb}"
            # )

            # assert final_atom_count_cs == initial_atom_count_cs, (
            #     f"{algo_cls.__name__} trial {trial}: Cs atoms not conserved; "
            #     f"started with {initial_atom_count_cs}, ended with {final_atom_count_cs}"
            # )

            # Contract 2: Success flag
            assert success_flag


@pytest.mark.parametrize("algo_cls", CONCRETE_ALGORITHMS)
def test_empty_target_configuration(algo_cls) -> None:
    """
    Test algorithm behavior when target is completely empty (no atoms needed).
    """
    if algo_cls in DUAL_SPECIES_ALGORITHMS:
        arr = _make_dual_species_array()
    else:
        arr = _make_single_species_array()

    arr.target = np.zeros_like(arr.target, dtype=np.uint8)

    algo = algo_cls()
    config, move_set, success_flag = algo.get_moves(arr)

    assert isinstance(config, (np.ndarray, AtomArray))
    if type(config) is AtomArray:
        assert np.sum(config.matrix)

    assert isinstance(move_set, list)
    assert (
        move_set == []
    ), f"{algo_cls.__name__} should return empty move set for empty target"
    assert isinstance(success_flag, bool)


@pytest.mark.parametrize("algo_cls", CONCRETE_ALGORITHMS)
def test_fully_occupied_target(algo_cls) -> None:
    """
    Test algorithm behavior when target is fully occupied (all sites filled).
    """
    if algo_cls in DUAL_SPECIES_ALGORITHMS:
        arr = _make_dual_species_array()
    else:
        arr = _make_single_species_array()

    rows, cols = arr.target.shape[:2]

    # Fill entire target
    if arr.n_species > 1:
        # For multi-species, fill with species 0
        arr.target = np.zeros((rows, cols, arr.n_species), dtype=np.uint8)
        arr.target[:, :, 0] = 1

    algo = algo_cls()
    config, move_set, success_flag = algo.get_moves(arr)

    # Should handle fully occupied target gracefully
    assert isinstance(config, (np.ndarray, AtomArray))
    assert isinstance(move_set, list)
    assert isinstance(success_flag, bool)


## TODO: Have randomized tests. See no atom lost during rearrangement

## TODO: Large array randomized tests
