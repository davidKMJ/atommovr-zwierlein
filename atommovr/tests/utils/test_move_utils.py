import copy
import numpy as np
import pytest

from atommovr.utils.Move import Move
from atommovr.utils import move_utils as mu


def _clone_move(move: Move) -> Move:
    """
    Clone a Move object while preserving the mutable failure annotations used by
    move application.
    """
    cloned = Move(move.from_row, move.from_col, move.to_row, move.to_col)
    cloned.fail_flag = move.fail_flag
    cloned.fail_event = move.fail_event
    return cloned


def _clone_moves(moves: list[Move]) -> list[Move]:
    """
    Clone a list of Move objects for side-effect-safe regression testing.
    """
    return [_clone_move(move) for move in moves]


def _has_legacy_midpoint_crossing(moves: list[Move]) -> bool:
    """
    Return whether two or more moves share the same midpoint, matching the
    legacy crossed-move detector's grouping criterion.
    """
    seen: set[tuple[float, float]] = set()
    for move in moves:
        midpoint = (move.midx, move.midy)
        if midpoint in seen:
            return True
        seen.add(midpoint)
    return False


class TestDETECT_DESTRUCTIVE_AOD_CMD_MASK:
    def test_detects_pattern_2_1(self) -> None:
        """
        The command-mask helper should mark both tones in a [2, 1] pattern.
        """
        arr = np.array([0, 2, 1, 0], dtype=np.int8)
        mask = mu.detect_destructive_aod_cmd_mask(arr)
        expected = np.array([False, True, True, False], dtype=np.bool_)
        assert np.array_equal(mask, expected)

    def test_detects_pattern_1_3(self) -> None:
        """
        The command-mask helper should mark both tones in a [1, 3] pattern.
        """
        arr = np.array([0, 1, 3, 0], dtype=np.int8)
        mask = mu.detect_destructive_aod_cmd_mask(arr)
        expected = np.array([False, True, True, False], dtype=np.bool_)
        assert np.array_equal(mask, expected)

    def test_detects_pattern_2_3(self) -> None:
        """
        The command-mask helper should mark both tones in a [2, 3] pattern.
        """
        arr = np.array([0, 2, 3, 0], dtype=np.int8)
        mask = mu.detect_destructive_aod_cmd_mask(arr)
        expected = np.array([False, True, True, False], dtype=np.bool_)
        assert np.array_equal(mask, expected)

    def test_has_no_false_positive_on_safe_pattern(self) -> None:
        """
        The command-mask helper should leave safe patterns unmarked.
        """
        arr = np.array([0, 2, 0, 1, 0, 3, 0], dtype=np.int8)
        mask = mu.detect_destructive_aod_cmd_mask(arr)
        expected = np.zeros_like(arr, dtype=np.bool_)
        assert np.array_equal(mask, expected)


class TestBUILD_DESTRUCTIVE_SUPPORT_MASK:
    def test_matches_user_example(self) -> None:
        """
        Collided vertical tones should only kill sites supported by those
        collided vertical tones and by active horizontal tones.
        """
        h_cmds = np.array([0, 1, 3, 0, 1], dtype=np.int8)
        v_cmds = np.array([1, 1, 2, 0, 0], dtype=np.int8)

        mask = mu.build_destructive_support_mask(h_cmds, v_cmds)

        expected = np.zeros((5, 5), dtype=np.bool_)
        expected[0:3, 1:3] = True

        assert np.array_equal(mask, expected)

    def test_horizontal_collision_only_kills_supported_sites(self) -> None:
        """
        Collided horizontal tones should only kill sites supported by those
        collided horizontal tones and by active vertical tones.
        """
        v_cmds = np.array([0, 2, 1, 0], dtype=np.int8)
        h_cmds = np.array([1, 1, 0, 1], dtype=np.int8)

        mask = mu.build_destructive_support_mask(h_cmds, v_cmds)

        expected = np.zeros((4, 4), dtype=np.bool_)
        expected[1:3, 0] = True
        expected[1:3, 1] = True
        expected[1:3, 3] = True

        assert np.array_equal(mask, expected)


class TestMOVE_ATOMS_NOISELESS:
    def test_preserves_2d_shape(self) -> None:
        """
        The deterministic planning helper should accept a 2D occupancy matrix
        and return a 2D occupancy matrix.
        """
        matrix = np.zeros((4, 4), dtype=np.uint8)
        matrix[0, 0] = 1

        out = mu.move_atoms_noiseless(matrix, [Move(0, 0, 0, 1)])

        assert out.shape == matrix.shape
        assert out.ndim == 2
        assert out.dtype == matrix.dtype
        assert out[0, 1] == 1
        assert out[0, 0] == 0

    def test_preserves_3d_shape(self) -> None:
        """
        The deterministic planning helper should accept a 3D single-species
        occupancy matrix and return the same shape.
        """
        matrix = np.zeros((4, 4, 1), dtype=np.uint8)
        matrix[0, 0, 0] = 1

        out = mu.move_atoms_noiseless(matrix, [Move(0, 0, 0, 1)])

        assert out.shape == matrix.shape
        assert out.ndim == 3
        assert out.dtype == matrix.dtype
        assert out[0, 1, 0] == 1
        assert out[0, 0, 0] == 0

    def test_vertical_collision_kills_only_supported_rectangle(self) -> None:
        """
        A collided vertical-tone pair should only eject atoms supported by those
        vertical tones and by active horizontal tones.
        """

        matrix = np.zeros([6, 5, 1], dtype=np.uint8)
        matrix[0, :, 0] = [0, 1, 1, 0, 1]
        matrix[1, :, 0] = [1, 1, 1, 0, 1]
        matrix[2, :, 0] = [1, 1, 1, 0, 1]
        matrix[3, :, 0] = [1, 0, 0, 0, 0]
        matrix[4, :, 0] = [0, 0, 0, 0, 1]
        matrix[5, :, 0] = [0, 0, 0, 0, 0]

        h_cmds = np.array([0, 1, 3, 0, 1], dtype=np.uint8)
        v_cmds = np.array([1, 1, 2, 0, 0, 0], dtype=np.uint8)

        moves = mu.get_move_list_from_AOD_cmds(h_cmds, v_cmds)

        out = mu.move_atoms_noiseless(matrix, _clone_moves(moves))

        expected = np.zeros([6, 5, 1], dtype=np.uint8)
        expected[0, :, 0] = [0, 0, 0, 0, 1]
        expected[1, :, 0] = [1, 0, 0, 0, 1]
        expected[2, :, 0] = [1, 0, 0, 0, 0]
        expected[3, :, 0] = [1, 0, 0, 0, 1]
        expected[4, :, 0] = [0, 0, 0, 0, 1]
        expected[5, :, 0] = [0, 0, 0, 0, 0]

        assert np.array_equal(out, expected)

    def test_safe_noiseless_batch_matches_regular_move_atoms(self) -> None:
        """
        On a non-colliding noiseless batch, the pared-down helper should match
        the regular move utility's state update.
        """
        matrix = np.zeros((4, 4, 1), dtype=np.uint8)
        matrix[0, 0, 0] = 1
        matrix[2, 2, 0] = 1

        moves = [
            Move(0, 0, 0, 1),
            Move(2, 2, 3, 2),
        ]

        ref_matrix, _ = mu.move_atoms(matrix.copy(), _clone_moves(moves))
        new_matrix = mu.move_atoms_noiseless(matrix.copy(), _clone_moves(moves))

        assert np.array_equal(new_matrix, ref_matrix)

    def test_safe_random_batches_match_regular_move_atoms(self) -> None:
        """
        On random batches with no destructive AOD support mask and no legacy
        midpoint-crossing ambiguity, the new helper should agree with the legacy
        move utility.
        """
        rng = np.random.default_rng(0)

        for side in [4, 5, 8]:
            for _ in range(100):
                matrix = (rng.random((side, side, 1)) < 0.25).astype(
                    np.uint8, copy=False
                )

                n_moves = int(rng.integers(1, 6))
                moves: list[Move] = []
                for _ in range(n_moves):
                    from_row = int(rng.integers(0, side))
                    from_col = int(rng.integers(0, side))
                    d_row = int(rng.integers(-1, 2))
                    d_col = int(rng.integers(-1, 2))

                    if d_row == 0 and d_col == 0:
                        d_row = 1

                    moves.append(
                        Move(
                            from_row,
                            from_col,
                            from_row + d_row,
                            from_col + d_col,
                        )
                    )

                support_mask, ok = mu.find_destructive_support_mask_from_moves(
                    matrix,
                    moves,
                )
                if support_mask.any() or not ok or _has_legacy_midpoint_crossing(moves):
                    continue

                try:
                    ref_matrix, _ = mu.move_atoms(matrix.copy(), _clone_moves(moves))
                except Exception:
                    continue

                new_matrix = mu.move_atoms_noiseless(matrix.copy(), _clone_moves(moves))
                assert np.array_equal(new_matrix, ref_matrix)


class TestMOVE_ATOMS:
    def test_matches_original_on_simple_legal_moves(self) -> None:
        """
        Behavioral regression test: fast move application must match the current
        implementation on a simple non-conflicting batch.
        """
        matrix = np.zeros((4, 4, 1), dtype=np.uint8)
        matrix[0, 0, 0] = 1
        matrix[2, 2, 0] = 1

        moves = [
            Move(0, 0, 0, 1),
            Move(2, 2, 3, 2),
        ]

        ref_moves = _clone_moves(moves)
        new_moves = _clone_moves(moves)

        ref_matrix, ref_meta = mu.move_atoms(matrix.copy(), ref_moves)
        new_matrix, new_meta = mu.move_atoms_fast(matrix.copy(), new_moves)

        assert np.array_equal(new_matrix, ref_matrix)
        assert new_meta == ref_meta
        assert [m.fail_flag for m in new_moves] == [m.fail_flag for m in ref_moves]

    def test_matches_original_on_illegal_and_eject_moves(self) -> None:
        """
        Behavioral regression test: fast move application must preserve the
        current handling of occupied destinations and ejection moves.
        """
        matrix = np.zeros((4, 4, 1), dtype=np.uint8)
        matrix[1, 1, 0] = 1
        matrix[1, 2, 0] = 1
        matrix[3, 3, 0] = 1

        moves = [
            Move(1, 1, 1, 2),
            Move(3, 3, 4, 3),
        ]

        ref_moves = _clone_moves(moves)
        new_moves = _clone_moves(moves)

        ref_matrix, ref_meta = mu.move_atoms(matrix.copy(), ref_moves)
        new_matrix, new_meta = mu.move_atoms_fast(matrix.copy(), new_moves)

        assert np.array_equal(new_matrix, ref_matrix)
        assert new_meta == ref_meta
        assert [m.fail_flag for m in new_moves] == [m.fail_flag for m in ref_moves]

    def test_matches_original_on_crossed_moves(self) -> None:
        """
        Behavioral regression test: fast move application must preserve the
        current crossed-tweezer resolution behavior.
        """
        matrix = np.zeros((3, 3, 1), dtype=np.uint8)
        matrix[0, 0, 0] = 1
        matrix[0, 1, 0] = 1

        moves = [
            Move(0, 0, 0, 1),
            Move(0, 1, 0, 0),
        ]

        ref_moves = _clone_moves(moves)
        new_moves = _clone_moves(moves)

        ref_matrix, ref_meta = mu.move_atoms(matrix.copy(), ref_moves)
        new_matrix, new_meta = mu.move_atoms_fast(matrix.copy(), new_moves)

        assert np.array_equal(new_matrix, ref_matrix)
        assert new_meta == ref_meta
        assert [m.fail_flag for m in new_moves] == [m.fail_flag for m in ref_moves]

    def test_matches_original_on_random_small_batches(self) -> None:
        """
        Behavioral regression test: fast move application must match the current
        implementation on representative small random single-species cases.

        If the legacy implementation raises on a random case, the fast version must
        raise the same exception type instead of silently diverging.
        """
        rng = np.random.default_rng(0)

        for side in [4, 5, 8]:
            for _ in range(100):
                matrix = (rng.random((side, side, 1)) < 0.25).astype(
                    np.uint8, copy=False
                )

                n_moves = int(rng.integers(1, 8))
                moves: list[Move] = []
                for _ in range(n_moves):
                    from_row = int(rng.integers(0, side))
                    from_col = int(rng.integers(0, side))
                    d_row = int(rng.integers(-1, 2))
                    d_col = int(rng.integers(-1, 2))

                    if d_row == 0 and d_col == 0:
                        d_row = 1

                    to_row = from_row + d_row
                    to_col = from_col + d_col
                    moves.append(Move(from_row, from_col, to_row, to_col))

                ref_moves = _clone_moves(moves)
                new_moves = _clone_moves(moves)

                try:
                    ref_matrix, ref_meta = mu.move_atoms(matrix.copy(), ref_moves)
                except Exception as ref_exc:
                    with pytest.raises(type(ref_exc)) as new_exc:
                        mu.move_atoms_fast(matrix.copy(), new_moves)

                    assert str(new_exc.value) == str(ref_exc)
                    assert [m.fail_flag for m in new_moves] == [
                        m.fail_flag for m in ref_moves
                    ]
                    continue

                new_matrix, new_meta = mu.move_atoms_fast(matrix.copy(), new_moves)

                assert np.array_equal(new_matrix, ref_matrix)
                assert new_meta == ref_meta
                assert [m.fail_flag for m in new_moves] == [
                    m.fail_flag for m in ref_moves
                ]


class TestAllocEventMask:
    def test_alloc_event_mask_shape_and_dtype(self) -> None:
        m = mu.alloc_event_mask(5)
        assert m.shape == (5,)
        assert m.dtype == np.uint64
        assert np.all(m == 0)

    def test_alloc_event_mask_custom_dtype(self) -> None:
        m = mu.alloc_event_mask(3, dtype=np.uint32)
        assert m.dtype == np.uint32


class TestGetMoveListFromAODCmds:
    def test_no_moves_for_all_zero(self) -> None:
        moves = mu.get_move_list_from_AOD_cmds([0, 0], [0, 0])
        assert moves == []

    def test_no_moves_when_one_axis_is_all_zero(self) -> None:
        moves = mu.get_move_list_from_AOD_cmds([1, 0], [0, 0, 0])
        assert moves == []

    def test_recognizes_static_static_pairs(self) -> None:
        # row=0 col=0 both 1 => no move
        moves = mu.get_move_list_from_AOD_cmds([0, 1, 0], [0, 0, 0, 0, 1])
        assert len(moves) == 1
        mv = moves[0]
        assert (mv.from_row, mv.from_col, mv.to_row, mv.to_col) == (4, 1, 4, 1)

    def test_generates_move_for_single_axis_motion(self) -> None:
        # vertical hold, horizontal moves +1
        moves = mu.get_move_list_from_AOD_cmds([2, 0, 0], [1])
        assert len(moves) == 1
        mv = moves[0]
        assert (mv.from_row, mv.from_col, mv.to_row, mv.to_col) == (0, 0, 0, 1)

    def test_generates_move_for_dual_axis_motion(self) -> None:
        # vertical moves -1, horizontal moves +1
        moves = mu.get_move_list_from_AOD_cmds([0, 2, 0, 0], [0, 0, 3])
        assert len(moves) == 1
        mv = moves[0]
        assert (mv.from_row, mv.from_col, mv.to_row, mv.to_col) == (2, 1, 1, 2)


class TestGetAODCmdsFromMoveList:
    def test_infers_axis_cmds_for_consistent_moves(self) -> None:
        mat = np.zeros((3, 4), dtype=np.int8)
        moves = [Move(1, 2, 1, 3)]  # horizontal +1, vertical hold
        h, v, ok = mu.get_AOD_cmds_from_move_list(mat, moves)
        assert ok is True
        assert h.dtype == np.int8 and v.dtype == np.int8
        assert h.tolist() == [0, 0, 2, 0]
        assert v.tolist() == [0, 1, 0]

    def test_detects_conflicting_cmds_on_same_source(self) -> None:
        mat = np.zeros((3, 4), dtype=np.int8)
        # Same source row=1 but conflicting vertical cmd (hold vs move)
        moves = [Move(1, 0, 1, 1), Move(1, 2, 2, 2)]
        _, _, ok = mu.get_AOD_cmds_from_move_list(mat, moves)
        assert ok is False


class TestMoveAtoms:
    def test_unsigned_input_is_recast_back_and_internal_is_signed(
        self, monkeypatch
    ) -> None:
        init = np.zeros((2, 2, 1), dtype=np.uint8)
        init[0, 0] = 1
        moves = [Move(0, 0, 0, 1)]

        seen_signed = {"ok": False}
        real_apply = mu._apply_moves

        def _wrapped_apply(
            init_matrix, matrix_copy, move_set, duplicate_move_inds, look_for_flag=False
        ):
            # This is the key assertion: internal working copy must be signed.
            assert np.issubdtype(matrix_copy.dtype, np.signedinteger)
            seen_signed["ok"] = True
            return real_apply(
                init_matrix,
                matrix_copy,
                move_set,
                duplicate_move_inds,
                look_for_flag=look_for_flag,
            )

        monkeypatch.setattr(mu, "_apply_moves", _wrapped_apply)

        out, [failed, flags] = mu.move_atoms(
            init_matrix=init,
            moves=moves,
            look_for_flag=False,
        )

        assert seen_signed["ok"] is True
        assert out.dtype == np.uint8  # recast back to original
        assert failed == []
        assert flags == []
        assert out[0, 0] == 0 and out[0, 1] == 1

    def test_signed_input_keeps_dtype(self) -> None:
        init = np.zeros((2, 2, 1), dtype=np.int16)
        init[0, 0] = 1
        moves = [Move(0, 0, 0, 1)]

        out, _ = mu.move_atoms(
            init_matrix=init,
            moves=moves,
            look_for_flag=False,
        )
        assert out.dtype == np.int16


class TestGetDuplicateValsFromList:
    def test_finds_duplicates(self) -> None:
        moves = [Move(0, 0, 0, 1), Move(0, 1, 0, 0), Move(1, 1, 1, 2)]
        midpoints = []
        for move in moves:
            midpoints.append((move.midx, move.midy))
        d = mu._get_duplicate_vals_from_list(midpoints)
        assert d == [(0.5, 0.0)]

    def test_noop(self) -> None:
        moves = []
        midpoints = []
        for move in moves:
            midpoints.append((move.midx, move.midy))
        d = mu._get_duplicate_vals_from_list(midpoints)
        assert d == []

    def test_no_duplicates(self) -> None:
        moves = [Move(0, 1, 0, 0), Move(1, 1, 1, 2)]
        midpoints = []
        for move in moves:
            midpoints.append((move.midx, move.midy))
        d = mu._get_duplicate_vals_from_list(midpoints)
        assert d == []


class TestFindAndResolveCrossedMoves:
    def test_marks_all_but_first_as_crossed_static(self) -> None:
        moves = [Move(1, 0, 0, 0), Move(0, 0, 1, 0)]
        matrix = np.zeros((2, 2), dtype=np.uint8)
        matrix[0, 0] = np.uint8(1)
        mat_out, dup = mu._find_and_resolve_crossed_moves(moves, matrix)
        assert dup == [0, 1]
        assert np.array_equal(np.zeros((2, 2), dtype=np.uint8), mat_out)

        assert int(moves[0].fail_flag) == 3
        assert int(moves[1].fail_flag) == 0


class TestApplyMoves:
    def test_legal_move(self) -> None:
        init = np.array([[1, 0], [0, 0]], dtype=float)
        out = copy.deepcopy(init)
        m = Move(0, 0, 0, 1)
        m.failure_flag = 0
        result_out, failed, fl = mu._apply_moves(init, out, [m])
        assert result_out[0, 0] == 0
        assert result_out[0, 1] == 1
        assert failed == []

    def test_no_atom_to_move(self) -> None:
        init = np.array([[0, 0], [0, 0]], dtype=float)
        out = copy.deepcopy(init)
        m = Move(0, 0, 0, 1)
        result_out, failed, fl = mu._apply_moves(init, out, [m])
        assert np.array_equal(result_out, np.zeros((2, 2)))
