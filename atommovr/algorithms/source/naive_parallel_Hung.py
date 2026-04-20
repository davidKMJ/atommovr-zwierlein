import copy
import numpy as np
from collections import deque
from typing import Callable
from scipy.optimize import linear_sum_assignment

from atommovr.utils.Move import Move
from atommovr.utils.AtomArray import AtomArray
from atommovr.algorithms.source.inside_out_utils import (
    BFSResult,
    collect_non_conflicting_moves,
    regroup_parallel_moves,
    same_species_ok,
    process_chain_moves_new,
    generate_decomposed_move_list,
)
from atommovr.algorithms.source.inside_out import (
    check_atom_enough,
    rearrangement_complete,
)
from atommovr.algorithms.source.Hungarian_works import generate_cost_matrix

"""
Utils for pairing atoms and target vacancies.
"""


def naive_par_Hung(
    rbcs_arrays: AtomArray, do_ejection: bool = False, round_lim: int = 30
):
    # Initialize the variables
    complete_flag = False
    move_list = []
    arrays = copy.deepcopy(rbcs_arrays)
    round_count = 0

    print(np.sum(arrays.matrix[:, :, 0]) >= np.sum(arrays.target_Rb))
    print(np.sum(arrays.matrix[:, :, 1]) >= np.sum(arrays.target_Cs))
    if not check_atom_enough(rbcs_arrays):  # == False: linting error
        return rbcs_arrays, [], False

    # Rearranging Rb arrays layer
    while (not complete_flag) and (round_count < round_lim):
        # Here we use deepcopy to ensure that we are not modifying the original arrays
        Rb_arrays = copy.deepcopy(arrays.matrix[:, :, 0])
        Rb_target = copy.deepcopy(arrays.target[:, :, 0])
        Cs_arrays = copy.deepcopy(arrays.matrix[:, :, 1])
        Cs_target = copy.deepcopy(arrays.target[:, :, 1])

        # 1. Generate the assignments
        Rb_assign = generate_assignments_naive_par(
            Rb_arrays, Cs_arrays, Rb_target, Cs_target
        )
        used_coord = [pair[0] for pair in Rb_assign] + [pair[1] for pair in Rb_assign]

        # 1.1. Avoid same target at reservoir
        Cs_assign = generate_assignments_naive_par(
            Cs_arrays, Rb_arrays, Cs_target, Rb_target, used_coord
        )
        prepared_assignments = Rb_assign + Cs_assign

        # 2. Generate paths according to the assignments
        N_independent_path = generate_path_naive_par(arrays, prepared_assignments)

        # 3. Transform the N_independent_path into a list of moves
        arrays, naive_par_move_list = transform_paths_into_moves_naive_par(
            arrays, N_independent_path, 1
        )
        move_list.extend(naive_par_move_list)

        round_count += 1

    if rearrangement_complete(arrays):
        complete_flag = True

    return arrays, move_list, complete_flag


def define_current_and_target_naive_par(
    matrix, other_matrix, target_config, other_target_config, relax: bool = False
):
    smallest_l = find_smallest_l(matrix, target_config)

    n_rows, n_cols = matrix.shape

    redundant_area = [
        (row, col)
        for row in range(n_rows)
        for col in range(n_cols)
        if target_config[row][col] == 0
        if other_target_config[row][col] == 0
    ]

    # Graceful fallback for locally infeasible subproblems
    if smallest_l is None:
        return [], [], redundant_area

    row_center = n_rows / 2
    col_center = n_cols / 2
    row_delta = n_rows % 2
    col_delta = n_cols % 2

    row_left = max(0, int(row_center - smallest_l + row_delta))
    row_right = min(n_rows, int(row_center + smallest_l))
    col_left = max(0, int(col_center - smallest_l + col_delta))
    col_right = min(n_cols, int(col_center + smallest_l))

    current_positions = [
        (row, col)
        for row in range(row_left, row_right)
        for col in range(col_left, col_right)
        if matrix[row][col] == 1
        if target_config[row][col] == 0
        if other_matrix[row][col] == 0
    ]

    target_positions = [
        (row, col)
        for row in range(n_rows)
        for col in range(n_cols)
        if matrix[row][col] == 0
        if target_config[row][col] == 1
        if other_matrix[row][col] == 0
    ]

    if len(current_positions) > 0 and len(target_positions) == 0:
        row_left2 = max(0, row_left - 2)
        row_right2 = min(n_rows, row_right + 2)
        col_left2 = max(0, col_left - 2)
        col_right2 = min(n_cols, col_right + 2)

        reservoir = [
            (row, col)
            for row in range(row_left2, row_right2)
            for col in range(col_left2, col_right2)
            if matrix[row][col] == 0
            if target_config[row][col] == 0
            if other_matrix[row][col] == 0
            if other_target_config[row][col] == 0
        ]

        if len(reservoir) == 0:
            reservoir = [
                (row, col)
                for row in range(row_left, row_right)
                for col in range(col_left, col_right)
                if matrix[row][col] == 0
                if target_config[row][col] == 0
                if other_matrix[row][col] == 0
            ]

        return current_positions, reservoir, redundant_area

    return current_positions, target_positions, redundant_area


def find_smallest_l(matrix, target_config):
    n_rows, n_cols = matrix.shape

    row_center = n_rows / 2
    col_center = n_cols / 2
    row_delta = n_rows % 2
    col_delta = n_cols % 2

    total_atoms = int(np.sum(matrix))
    total_targets = int(np.sum(target_config))

    # Local subproblem may be infeasible even if the global instance is feasible.
    # Do not raise here; let the caller handle it gracefully.
    if total_atoms < total_targets:
        return None

    max_l = max(n_rows, n_cols)

    for smallest_l in range(1, max_l + 1):
        row_left = max(0, int(row_center - smallest_l + row_delta))
        row_right = min(n_rows, int(row_center + smallest_l))
        col_left = max(0, int(col_center - smallest_l + col_delta))
        col_right = min(n_cols, int(col_center + smallest_l))

        if np.sum(matrix[row_left:row_right, col_left:col_right]) >= total_targets:
            return smallest_l

    return None


def generate_assignments_naive_par(
    matrix, other_matrix, target_config, other_target_config, used_coord: list = None
):
    # Define target positions for the center square in a matrix.
    current_positions, target_positions, redundant_area = (
        define_current_and_target_naive_par(
            matrix, other_matrix, target_config, other_target_config
        )
    )
    # print("current", current_positions)
    # print("target", target_positions)

    # If there are no empty targets or sources inside source area, relax target condition
    if len(target_positions) == 0 or len(current_positions) == 0:
        current_positions, target_positions, redundant_area = (
            define_current_and_target_naive_par(
                matrix, other_matrix, target_config, other_target_config, relax=True
            )
        )

        if len(current_positions) == 0 or len(target_positions) == 0:
            return []

    # If used_coord is provided, filter out the target positions that are already occupied
    if used_coord is not None:
        current_positions = [pos for pos in current_positions if pos not in used_coord]
        target_positions = [pos for pos in target_positions if pos not in used_coord]

    # Generate the cost matrix using the current atom positions and the target positions
    cost_matrix = generate_cost_matrix(current_positions, target_positions)

    # row_ind and col_ind are arrays of indices indicating the optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Pair up row_ind and col_ind and sort by col_ind
    paired_indices = sorted(zip(row_ind, col_ind, strict=True), key=lambda x: x[1])

    if paired_indices:
        # Unzip the sorted pairs if paired_indices is not empty
        sorted_row_ind, sorted_col_ind = zip(*paired_indices, strict=True)
    else:
        # Assign default values if paired_indices is empty
        sorted_row_ind, sorted_col_ind = [], []

    prepared_assignments = [
        (current_positions[i], target_positions[j])
        for i, j in zip(sorted_row_ind, sorted_col_ind, strict=True)
    ]
    prepared_assignments_new = copy.deepcopy(prepared_assignments)

    # Filter the assignments
    for start, end in prepared_assignments:
        if start in redundant_area and end in redundant_area:
            prepared_assignments_new.remove((start, end))
    # print("prepared assignments after remove", prepared_assignments_new)
    return prepared_assignments_new


def generate_path_naive_par(
    arrays: AtomArray, prepared_assignments: list[tuple, tuple]
) -> list:
    """
    For each (start, end), run BFS to find a path. If BFS fails or finds different-species occupant, log to type_2_pair. Return a list of (move_list, category).
    """
    move_list_for_assigns = []
    op_arrays = copy.deepcopy(arrays)

    for start, end in prepared_assignments:
        bfs_res = bfs_find_path_naive_par(
            op_arrays.matrix, start, end, same_species_ok(op_arrays)
        )
        if bfs_res.end_reached:
            single_path = process_chain_moves_new(bfs_res)
            move_list_for_assigns = generate_decomposed_move_list(
                op_arrays, single_path, move_list_for_assigns
            )

    return move_list_for_assigns


def neighbors_8_naive_par(
    r: int, c: int, n_rows: int, n_cols: int
) -> list[tuple[int, int]]:
    neighbors = []
    directions = [
        (dr, dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if (dr, dc) != (0, 0)
    ]
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < n_rows and 0 <= nc < n_cols:
            neighbors.append((nr, nc))
    return neighbors


def bfs_find_path_naive_par(
    matrix: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    handle_obstacle_filter: Callable[[tuple[int, int], tuple[int, int]], bool],
) -> BFSResult:

    n_rows, n_cols = matrix.shape[:2]

    visited = set([start])
    same_obstacle_list = []
    diff_obstacle_list = []
    queue = deque(
        [(start[0], start[1], [start], same_obstacle_list, diff_obstacle_list)]
    )

    while queue:
        row, col, path_so_far, same_obstacle_list, diff_obstacle_list = queue.popleft()

        if (row, col) == end:
            if len(diff_obstacle_list) == 0:
                return BFSResult(
                    path_so_far, True, same_obstacle_list, diff_obstacle_list, 1
                )
            else:
                return BFSResult(
                    path_so_far, True, same_obstacle_list, diff_obstacle_list, 2
                )

        for new_r, new_c in neighbors_8_naive_par(row, col, n_rows, n_cols):
            if (new_r, new_c) not in visited:
                pass_flag, homo_obs, hetero_obs = handle_obstacle_filter(
                    start, (new_r, new_c)
                )
                if pass_flag:
                    visited.add((new_r, new_c))
                    if homo_obs:
                        queue.append(
                            (
                                new_r,
                                new_c,
                                path_so_far + [(new_r, new_c)],
                                same_obstacle_list + [homo_obs],
                                diff_obstacle_list,
                            )
                        )
                    elif hetero_obs:
                        queue.append(
                            (
                                new_r,
                                new_c,
                                path_so_far + [(new_r, new_c)],
                                same_obstacle_list,
                                diff_obstacle_list + [hetero_obs],
                            )
                        )
                    else:
                        queue.append(
                            (
                                new_r,
                                new_c,
                                path_so_far + [(new_r, new_c)],
                                same_obstacle_list,
                                diff_obstacle_list,
                            )
                        )

    return BFSResult(path_so_far, False, same_obstacle_list, diff_obstacle_list, 2)


def transform_paths_into_moves_naive_par(
    arrays: AtomArray, all_paths: list[list[Move]], max_rounds: int = 1
) -> tuple[AtomArray, list[list[Move]]]:
    """
    Execute up to one move from each path in 'paths' per round, avoiding collisions. Collisions occur if two moves share a 'to' or 'from' coordinate.
    Returns: (arrays, parallel_moves)
    parallel_moves: list of rounds, each round is a list of Move objects that were executed simultaneously.
    """
    parallel_moves = []
    round_count = 0

    """If max_rounds set 1, we could dynamically search newest path every time."""
    while round_count < max_rounds:
        # 1) Gather the candidate move from each path, if available
        move_candidates = []
        for path in all_paths:
            if len(path) > 0:  # path is not empty
                move_candidates.append(path[0])  # next move in this path

        # 2) Identify non-conflicting moves among 'candidates'
        moves_in_scan = collect_non_conflicting_moves(move_candidates, arrays)

        # 3) Parallelize moves in this round
        if len(moves_in_scan) > 0:
            matrix = arrays.matrix[:, :, 0] + arrays.matrix[:, :, 1]
            moves_in_scan = regroup_parallel_moves(matrix, moves_in_scan)
            # 2.1.3 Implement the moves
            parallel_moves.extend(moves_in_scan)
            for moves in moves_in_scan:
                _ = arrays.move_atoms(moves)

        # 4) Remove them from each path
        for move in moves_in_scan:
            for path in all_paths:
                if path and path[0] == move:
                    path.pop(0)  # remove this move from that path
                    break
        round_count += 1

    return arrays, parallel_moves
