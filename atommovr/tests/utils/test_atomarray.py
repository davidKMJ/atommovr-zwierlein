import pytest
import numpy as np
from numpy.typing import NDArray
import importlib

from atommovr.utils.AtomArray import AtomArray
from atommovr.utils.Move import Move
from atommovr.utils.move_utils import MoveType, MultiOccupancyFlag
from atommovr.utils.core import Configurations
from atommovr.utils.failure_policy import FailureEvent, FailureFlag
from atommovr.tests.support.doubles import TimingSpyErrorModel, BoomErrorModel
from atommovr.tests.support.helpers import boom

AtomArray_mod = importlib.import_module("atommovr.utils.AtomArray")


class TestAtomArrayInitBasic:
    def test_default_initialization(self):
        array = AtomArray()
        assert array.shape == [10, 10]
        assert array.n_species == 1
        assert array.matrix.shape == (10, 10, 1)

    def test_custom_shape(self):
        array = AtomArray(shape=[5, 8])
        assert array.shape == [5, 8]
        assert array.matrix.shape == (5, 8, 1)

    def test_dual_species(self):
        array = AtomArray(shape=[5, 5], n_species=2)
        assert array.n_species == 2
        assert array.matrix.shape == (5, 5, 2)

    def test_invalid_n_species_raises(self):
        with pytest.raises(ValueError):
            AtomArray(n_species=3)

    def test_load_tweezers_single_species_binary(self):
        array = AtomArray(shape=[4, 4], n_species=1)
        array.load_tweezers()
        assert array.matrix.shape == (4, 4, 1)
        assert np.all((array.matrix == 0) | (array.matrix == 1))

    def test_load_tweezers_dual_species_no_overlap(self):
        array = AtomArray(shape=[3, 3], n_species=2)
        array.load_tweezers()
        for i in range(3):
            for j in range(3):
                assert not (array.matrix[i, j, 0] == 1 and array.matrix[i, j, 1] == 1)


class TestMoveAtomsFastPath:
    def test_no_crossed_tones_skips_expensive_cross_bookkeeping(
        self, monkeypatch
    ) -> None:
        """
        Performance regression tripwire:

        When the precheck `_has_colliding_tones(...)` is False, `move_atoms` should not
        call `_find_colliding_tones` nor `collision_eligibility_from_tones`.

        This matters because `move_atoms` is called ~O(1e5-1e6) times in long simulations.
        """
        aa = AtomArray(shape=[2, 3], n_species=1)
        aa.matrix[:, :, 0] = np.uint8(0)
        aa.matrix[0, 0, 0] = np.uint8(1)

        # Ensure we go through the pipeline with a valid, parallelizable move.
        moves = [Move(0, 0, 0, 1)]

        monkeypatch.setattr(AtomArray_mod, "_has_colliding_tones", lambda v, h: False)
        monkeypatch.setattr(AtomArray_mod, "_find_colliding_tones", boom)
        monkeypatch.setattr(AtomArray_mod, "collision_eligibility_from_tones", boom)

        (failed, flags), _t = aa.move_atoms(moves)
        assert failed == []
        assert isinstance(flags, list)

    def test_crossed_tones_true_calls_cross_info_once(self, monkeypatch) -> None:
        """
        Basic control: if the precheck says we have colliding tones, we should call the
        collision-info routine (and not crash due to missing locals).
        """
        aa = AtomArray(shape=[2, 3], n_species=1)
        aa.matrix[:, :, 0] = np.uint8(0)
        aa.matrix[0, 0, 0] = np.uint8(1)

        moves = [Move(0, 0, 0, 1)]

        monkeypatch.setattr(AtomArray_mod, "_has_colliding_tones", lambda v, h: True)

        calls = {"n": 0}

        def _fake_cross_info(v, h):
            calls["n"] += 1
            # Return “no victims” outputs with correct dtypes/shapes.
            return (
                np.zeros(0, dtype=np.intp),
                np.zeros(0, dtype=np.intp),
                np.zeros(0, dtype=np.intp),
                np.zeros(0, dtype=np.intp),
            )

        monkeypatch.setattr(AtomArray_mod, "_find_colliding_tones", _fake_cross_info)

        # crossed_eligibility_from_tones should be called but with empty crossed arrays.
        def _fake_eligibility(**kwargs):
            n = len(kwargs["move_set"])
            return np.zeros(n, dtype=np.bool_), np.zeros(n, dtype=np.bool_)

        monkeypatch.setattr(
            AtomArray_mod, "collision_eligibility_from_tones", _fake_eligibility
        )

        aa.move_atoms(moves)
        assert calls["n"] == 1


class TestMoveAtomsInputValidation:
    def test_non_parallelizable_move_list_raises_valueerror(
        self, one_atom_array_3x3
    ) -> None:
        """
        Document the contract: `move_atoms` rejects non-parallelizable move sets.

        This is important because timing analysis assumes a consistent AOD command encoding.
        """
        aa = one_atom_array_3x3
        aa.matrix[0, 1, 0] = np.uint8(1)

        # Two moves share the same source row but different columns => should violate parallelizability
        # under your AOD model (depending on your get_AOD_cmds_from_move_list implementation).
        moves = [Move(0, 0, 0, 1), Move(0, 1, 1, 2)]

        with pytest.raises(ValueError, match="Non-parallelizable"):
            aa.move_atoms(moves)

    def test_empty_move_list_is_noop(self) -> None:
        """
        Regression guard:

        `move_atoms([])` should be a no-op (no time, no failures) rather than raising
        `IndexError` from the `move_list[0]` probe.
        """
        aa = AtomArray(shape=[2, 2], n_species=1)
        before = aa.matrix.copy()

        # Expected behavior (recommended): no-op.
        (failed, flags), move_time = aa.move_atoms([])

        assert failed == []
        assert isinstance(flags, list)
        assert move_time == 0
        assert np.array_equal(aa.matrix, before)


class TestLoadTweezers:
    def test_load_tweezers_dual_species_resolves_double_occupancy(
        self, monkeypatch
    ) -> None:
        """
        Deterministic test for the dual-species overlap-resolution logic in `load_tweezers`.

        We force `random_loading` to return all-ones for both species so every site overlaps,
        then force `random.randint` to always remove species 0. After loading, species 0
        should be all zeros, species 1 all ones.
        """
        aa = AtomArray(shape=[2, 3], n_species=2)

        def _fake_random_loading(shape, probability):
            return np.ones((shape[0], shape[1]), dtype=np.uint8)

        monkeypatch.setattr(AtomArray_mod, "random_loading", _fake_random_loading)
        monkeypatch.setattr(AtomArray_mod.random, "randint", lambda a, b: 0)

        aa.load_tweezers()

        assert np.all(aa.matrix[:, :, 0] == np.uint8(0))
        assert np.all(aa.matrix[:, :, 1] == np.uint8(1))
        # and last_loaded_config is set
        assert hasattr(aa, "last_loaded_config")
        assert aa.last_loaded_config.shape == aa.matrix.shape


class TestGenerateTarget:

    @pytest.mark.parametrize(
        "pattern",
        [
            Configurations.CHECKERBOARD,
            Configurations.MIDDLE_FILL,
            Configurations.RANDOM,
            Configurations.ZEBRA_HORIZONTAL,
            Configurations.ZEBRA_VERTICAL,
            Configurations.Left_Sweep,
        ],
    )
    def test_generate_target_single_species_pattern_shape_and_values(
        self, pattern
    ) -> None:
        """
        Basic target-generation invariant: correct shape and boolean occupancy for all patterns.
        """
        aa = AtomArray(shape=[4, 5], n_species=1)
        aa.generate_target(pattern=pattern)

        assert aa.target.shape == (4, 5, 1)
        assert aa.target.dtype == np.uint8
        assert np.all((aa.target == np.uint8(0)) | (aa.target == np.uint8(1)))

    @pytest.mark.parametrize(
        "pattern",
        [
            Configurations.CHECKERBOARD,
            Configurations.SEPARATE,
            Configurations.ZEBRA_HORIZONTAL,
            Configurations.ZEBRA_VERTICAL,
        ],
    )
    def test_generate_target_dual_species_pattern_shape_and_values(
        self, pattern
    ) -> None:
        """
        Basic target-generation invariant: correct shape and boolean occupancy for all patterns.
        """
        aa = AtomArray(shape=[5, 4], n_species=2)
        aa.generate_target(pattern=pattern)

        assert aa.target.shape == (5, 4, 2)
        assert aa.target.dtype == np.uint8
        assert np.all((aa.target == np.uint8(0)) | (aa.target == np.uint8(1)))

    def test_generate_target_invalid_species_raises(self) -> None:
        """
        Defensive contract: only n_species in {1,2} is supported.
        """
        aa = AtomArray(shape=[2, 2], n_species=1)
        aa.n_species = 3  # simulate a bad mutation
        with pytest.raises(ValueError, match="only supports single and dual species"):
            aa.generate_target(pattern=Configurations.CHECKERBOARD)


class TestImageHelpers:
    def test_image_invalid_plotted_species_raises_dual(self, monkeypatch) -> None:
        """
        `image(plotted_species=...)` should validate the requested species label for dual-species arrays.
        """
        aa = AtomArray(shape=[2, 2], n_species=2)

        # prevent actual plotting
        monkeypatch.setattr(
            AtomArray_mod, "dual_species_image", lambda *args, **kwargs: None
        )

        with pytest.raises(
            ValueError, match="Invalid entry for parameter 'plotted_species'"
        ):
            aa.image(plotted_species="definitely-not-a-species")

    def test_plot_target_config_calls_expected_backend(self, monkeypatch) -> None:
        """
        Non-rendering test: ensure the correct plotting backend is selected.
        """
        calls = {"single": 0, "dual": 0}

        monkeypatch.setattr(
            AtomArray_mod,
            "single_species_image",
            lambda *args, **kwargs: calls.__setitem__("single", calls["single"] + 1),
        )
        monkeypatch.setattr(
            AtomArray_mod,
            "dual_species_image",
            lambda *args, **kwargs: calls.__setitem__("dual", calls["dual"] + 1),
        )

        aa1 = AtomArray(shape=[2, 2], n_species=1)
        aa1.plot_target_config()
        assert calls["single"] == 1
        assert calls["dual"] == 0

        aa2 = AtomArray(shape=[2, 2], n_species=2)
        aa2.plot_target_config()
        assert calls["dual"] == 1


class TestFlagsStructure:
    def test_move_atoms_flags_not_nested_list(self) -> None:
        """
        API legibility test:

        `move_atoms` should return a flat list of flags, not a list that contains
        an embedded list (e.g. `flags.append(multi_occupancy_flags)`).

        If this fails currently, consider changing:
            flags.append(multi_occupancy_flags)
        to:
            flags.extend(multi_occupancy_flags)
        and only extend when non-empty.
        """
        aa = AtomArray(shape=[2, 3], n_species=1)
        aa.matrix[:, :, 0] = np.uint8(0)
        aa.matrix[0, 0, 0] = np.uint8(1)

        (failed, flags), _t = aa.move_atoms([Move(0, 0, 0, 1)])

        # We don't require any specific flags here, only that the container is flat.
        assert all(not isinstance(x, list) for x in flags)


class TestMoveAtomsContracts:
    def test_move_atoms_raises_on_negative_occupancy_before_detectors_or_error_model(
        self, monkeypatch
    ) -> None:
        """
        The physical-state guard should fire before any downstream timing/collision/error
        machinery runs.

        This is a seam/invariant test: an invalid input state should not partially execute
        transport logic before raising.
        """
        aa = AtomArray(shape=[2, 2], n_species=1)
        aa.matrix = aa.matrix.astype(np.int8, copy=True)
        aa.matrix[0, 0, 0] = np.int8(-1)

        move = Move(0, 0, 0, 1)

        def _boom(*args, **kwargs):
            raise AssertionError(
                "Downstream detector/error machinery should not run on invalid input state."
            )

        monkeypatch.setattr(AtomArray_mod, "get_AOD_cmds_from_move_list", _boom)

        aa.error_model = BoomErrorModel()

        with pytest.raises(ValueError, match="negative occupancy"):
            aa.move_atoms([move])

    # NOTE: disabled this error; instead made move_atoms fill in the incomplete rectilinear AOD grid.
    # def test_move_atoms_incomplete_parallel_grid_raises(
    #     self, monkeypatch, one_atom_array_3x3
    # ) -> None:
    #     """
    #     `move_atoms` should reject an incomplete rectilinear AOD grid.

    #     If the active horizontal/vertical tones define a 2x2 grid of tweezer intersections,
    #     then the move list must contain all 4 row/col pairings so downstream error modeling
    #     sees the full parallel command set, including stationary tweezers.
    #     """
    #     aa = one_atom_array_3x3

    #     incomplete_moves = [
    #         Move(1, 0, 1, 1),
    #         Move(2, 1, 3, 1),
    #         Move(2, 0, 3, 1),
    #     ]

    #     curr_horiz = np.array([2, 1, 0], dtype=np.int8)
    #     curr_vert = np.array([0, 1, 2], dtype=np.int8)

    #     monkeypatch.setattr(
    #         AtomArray_mod,
    #         "get_AOD_cmds_from_move_list",
    #         lambda matrix, move_list: (curr_horiz, curr_vert, True),
    #     )

    #     with pytest.raises(ValueError, match="Move list is not complete"):
    #         aa.move_atoms(incomplete_moves)


class TestMoveAtomsSeams:
    def test_evaluate_moves_sums_per_round_move_times_exactly(
        self, monkeypatch
    ) -> None:
        """
        `evaluate_moves` is an orchestrator over repeated `move_atoms` calls.

        This seam test pins down the contract that total time is the exact sum of the
        per-round times returned by `move_atoms`, while the move counters still reflect
        the original move-set structure.
        """
        aa = AtomArray(shape=[2, 3], n_species=1)

        round_times = {
            1: 1.25,
            2: 2.5,
            3: 3.74,
        }
        seen_prev_next: list[tuple[int, int, int]] = []

        def _fake_move_atoms(move_list, prev_move_list=None, next_move_list=None):
            prev_len = len(prev_move_list) if prev_move_list is not None else -1
            next_len = len(next_move_list) if next_move_list is not None else -1
            curr_len = len(move_list)
            seen_prev_next.append((prev_len, curr_len, next_len))
            return [[], []], round_times[curr_len]

        monkeypatch.setattr(aa, "move_atoms", _fake_move_atoms)

        move_set = [
            [Move(0, 0, 0, 1), Move(1, 0, 1, 1)],
            [Move(0, 1, 0, 2)],
            [Move(2, 4, 3, 5), Move(9, 3, 9, 2), Move(1, 1, 1, 1)],
        ]

        total_time, [n_parallel, n_non_parallel] = aa.evaluate_moves(move_set)

        assert total_time == pytest.approx(7.49)
        assert n_parallel == 3
        assert n_non_parallel == 6
        assert seen_prev_next == [(0, 2, 1), (2, 1, 3), (1, 3, 0)]

    @pytest.mark.parametrize(
        "error_event",
        [
            [True, True, False, True],
            [True, False, True, False],
            [False, False, False, False],
            [True, False, False, True],
            [True, True, True, True],
        ],
    )
    def test_time_accounting_is_exact_sum_of_enabled_phases_and_travel(
        self, monkeypatch, error_event
    ) -> None:
        """
        `move_atoms` should add exactly the enabled phase costs plus geometric travel time,
        and pass that exact total into `error_model.get_atom_loss`.
        """
        error_model = TimingSpyErrorModel()
        aa = AtomArray(shape=[1, 3], n_species=1, error_model=error_model)
        aa.matrix[:, :, 0] = np.uint8(0)
        aa.matrix[0, 0, 0] = np.uint8(1)

        aa.params.spacing = 2.0
        aa.params.AOD_speed = 4.0

        moves = [
            Move(0, 0, 0, 1),
            Move(0, 2, 0, 2),
        ]  # distance = 2 -> travel time = 2 / 4 = 0.5

        monkeypatch.setattr(AtomArray_mod, "_has_colliding_tones", lambda v, h: False)
        monkeypatch.setattr(
            AtomArray_mod,
            "_detect_pickup_and_accel_masks",
            lambda *args, **kwargs: (
                np.array([error_event[0], False], dtype=np.bool_),
                np.array([error_event[1], False], dtype=np.bool_),
            ),
        )
        monkeypatch.setattr(
            AtomArray_mod,
            "_detect_decel_and_putdown_masks",
            lambda *args, **kwargs: (
                np.array([error_event[2], False], dtype=np.bool_),
                np.array([error_event[3], False], dtype=np.bool_),
            ),
        )

        (_, _), move_time = aa.move_atoms(moves)

        expected = (
            error_model.pickup_time * int(error_event[0])
            + error_model.accel_time * int(error_event[1])
            + error_model.decel_time * int(error_event[2])
            + error_model.putdown_time * int(error_event[3])
            + 0.5
        )
        assert move_time == pytest.approx(expected)
        assert error_model.loss_times == [pytest.approx(expected)]

    def test_timing_detectors_receive_prev_curr_next_cmds_and_source_arrays(
        self, monkeypatch
    ) -> None:
        """
        `move_atoms` should pass the correct previous/current/next AOD commands and
        source row/col arrays into the timing-mask detectors.
        """
        error_model = TimingSpyErrorModel()
        aa = AtomArray(shape=[2, 3], n_species=1, error_model=error_model)
        aa.matrix[:, :, 0] = np.uint8(0)
        aa.matrix[0, 0, 0] = np.uint8(1)

        prev_moves = [Move(0, 0, 0, 1)]
        curr_moves = [Move(0, 0, 1, 0)]
        next_moves = [Move(1, 0, 1, 1)]

        prev_h = np.array([11, 0], dtype=np.int8)
        prev_v = np.array([14, 0, 0], dtype=np.int8)
        curr_h = np.array([21, 0], dtype=np.int8)
        curr_v = np.array([0, 24, 0], dtype=np.int8)
        next_h = np.array([0, 32], dtype=np.int8)
        next_v = np.array([0, 34, 0], dtype=np.int8)

        def _fake_get_aod_cmds(
            matrix: NDArray, move_list: list[Move]
        ) -> tuple[NDArray[np.int8], NDArray[np.int8], bool]:
            if move_list is curr_moves:
                return curr_h, curr_v, True
            if move_list is prev_moves:
                return prev_h, prev_v, True
            if move_list is next_moves:
                return next_h, next_v, True
            raise AssertionError("Unexpected move list")

        def _fake_pickup_accel(
            ph: NDArray[np.int8],
            pv: NDArray[np.int8],
            ch: NDArray[np.int8],
            cv: NDArray[np.int8],
            move_list: list[Move],
            source_cols: NDArray[np.int_],
            source_rows: NDArray[np.int_],
        ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
            assert move_list is curr_moves
            assert np.array_equal(ph, prev_h)
            assert np.array_equal(pv, prev_v)
            assert np.array_equal(ch, curr_h)
            assert np.array_equal(cv, curr_v)
            assert np.array_equal(source_rows, np.array([0], dtype=np.int_))
            assert np.array_equal(source_cols, np.array([0], dtype=np.int_))
            return np.array([False], dtype=np.bool_), np.array([False], dtype=np.bool_)

        def _fake_decel_putdown(
            ch: NDArray[np.int8],
            cv: NDArray[np.int8],
            nh: NDArray[np.int8],
            nv: NDArray[np.int8],
            move_list: list[Move],
            source_cols: NDArray[np.int_],
            source_rows: NDArray[np.int_],
        ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
            assert move_list is curr_moves
            assert np.array_equal(ch, curr_h)
            assert np.array_equal(cv, curr_v)
            assert np.array_equal(nh, next_h)
            assert np.array_equal(nv, next_v)
            assert np.array_equal(source_rows, np.array([0], dtype=np.int_))
            assert np.array_equal(source_cols, np.array([0], dtype=np.int_))
            return np.array([False], dtype=np.bool_), np.array([False], dtype=np.bool_)

        monkeypatch.setattr(
            AtomArray_mod, "get_AOD_cmds_from_move_list", _fake_get_aod_cmds
        )
        monkeypatch.setattr(AtomArray_mod, "_has_colliding_tones", lambda v, h: False)
        monkeypatch.setattr(
            AtomArray_mod, "_detect_pickup_and_accel_masks", _fake_pickup_accel
        )
        monkeypatch.setattr(
            AtomArray_mod, "_detect_decel_and_putdown_masks", _fake_decel_putdown
        )

        aa.move_atoms(curr_moves, prev_move_list=prev_moves, next_move_list=next_moves)

    def test_phase_error_methods_are_not_called_when_masks_are_all_false(
        self, monkeypatch
    ) -> None:
        """
        Phase-specific error hooks should not be called when all timing masks are false.
        In that case, only geometric travel time should contribute.
        """
        error_model = TimingSpyErrorModel()
        aa = AtomArray(shape=[1, 2], n_species=1, error_model=error_model)
        aa.matrix[:, :, 0] = np.uint8(0)
        aa.matrix[0, 0, 0] = np.uint8(1)

        monkeypatch.setattr(AtomArray_mod, "_has_colliding_tones", lambda v, h: False)
        monkeypatch.setattr(
            AtomArray_mod,
            "_detect_pickup_and_accel_masks",
            lambda *args, **kwargs: (
                np.array([False], dtype=np.bool_),
                np.array([False], dtype=np.bool_),
            ),
        )
        monkeypatch.setattr(
            AtomArray_mod,
            "_detect_decel_and_putdown_masks",
            lambda *args, **kwargs: (
                np.array([False], dtype=np.bool_),
                np.array([False], dtype=np.bool_),
            ),
        )

        (_, _), move_time = aa.move_atoms([Move(0, 0, 0, 1)])

        expected_travel = aa.params.spacing / aa.params.AOD_speed
        assert move_time == pytest.approx(expected_travel)
        assert error_model.calls == []

    def test_resolved_primary_events_are_written_back_before_apply(
        self, monkeypatch
    ) -> None:
        """
        `move_atoms` should resolve primary events, write them onto the `Move` objects,
        and only then hand the move list to `_apply_moves`.
        """
        error_model = TimingSpyErrorModel()
        aa = AtomArray(shape=[1, 2], n_species=1, error_model=error_model)
        aa.matrix[:, :, 0] = np.uint8(0)
        aa.matrix[0, 0, 0] = np.uint8(1)

        move = Move(0, 0, 0, 1)

        monkeypatch.setattr(AtomArray_mod, "_has_colliding_tones", lambda v, h: False)
        monkeypatch.setattr(
            AtomArray_mod,
            "_detect_pickup_and_accel_masks",
            lambda *args, **kwargs: (
                np.array([True], dtype=np.bool_),
                np.array([False], dtype=np.bool_),
            ),
        )
        monkeypatch.setattr(
            AtomArray_mod,
            "_detect_decel_and_putdown_masks",
            lambda *args, **kwargs: (
                np.array([False], dtype=np.bool_),
                np.array([False], dtype=np.bool_),
            ),
        )

        monkeypatch.setattr(AtomArray_mod, "suppress_inplace", lambda event_mask: None)
        monkeypatch.setattr(
            AtomArray_mod,
            "resolve_primary_events",
            lambda event_mask: np.array(
                [int(FailureEvent.PICKUP_FAIL)], dtype=np.int32
            ),
        )

        def _fake_apply_moves(move_list: list[Move]) -> tuple[list[int], list[int]]:
            assert move_list[0].fail_event == FailureEvent.PICKUP_FAIL
            return [0], [move_list[0].fail_flag]

        monkeypatch.setattr(aa, "_apply_moves", _fake_apply_moves)

        (failed, _flags), _ = aa.move_atoms([move])

        assert failed == [0]


class TestMoveAtomsResults:
    def test_move_atoms_raises_on_negative_occupancy(self) -> None:
        """
        Document the physical-state guard: the occupancy representation must not
        contain negative values. This is mainly a regression tripwire for accidental
        dtype changes (e.g., if `matrix` ever becomes signed).
        """
        aa = AtomArray(shape=[2, 2], n_species=1)

        # Force a signed dtype to simulate a bug/regression where negatives become possible.
        aa.matrix = aa.matrix.astype(np.int8, copy=True)
        aa.matrix[0, 0, 0] = np.int8(-1)

        m = Move(from_row=0, from_col=0, to_row=0, to_col=1)

        with pytest.raises(ValueError, match="negative occupancy"):
            aa.move_atoms([m])

    def test_single_move(self):

        # 1. Single species
        array = AtomArray(shape=[3, 4])
        array.matrix = np.array(
            [[0, 1, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.uint8
        ).reshape(3, 4, 1)
        # checking that moves get row/col assignments correct
        _ = array.move_atoms(move_list=[Move(0, 1, 1, 1)])
        assert np.array_equal(
            array.matrix,
            np.array(
                [[0, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 0]], dtype=np.uint8
            ).reshape(3, 4, 1),
        )

    def test_chain_move(self):
        array = AtomArray(shape=[3, 4])
        array.matrix = np.array(
            [[0, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 0]], dtype=np.uint8
        ).reshape(3, 4, 1)
        # checking that moves of sites next to one another work properly
        _ = array.move_atoms(move_list=[Move(1, 0, 1, 1), Move(1, 1, 1, 2)])
        assert np.array_equal(
            array.matrix,
            np.array([[0, 0, 1, 0], [0, 1, 1, 0], [0, 0, 1, 0]]).reshape(3, 4, 1),
        )

    def test_chain_move_with_absent_atom(self):
        array = AtomArray(shape=[3, 4])
        array.matrix = np.array(
            [[0, 0, 1, 0], [0, 1, 1, 0], [0, 0, 1, 0]], dtype=np.uint8
        ).reshape(3, 4, 1)
        # checking that atoms are not used in two moves
        _ = array.move_atoms(move_list=[Move(2, 2, 2, 1), Move(2, 1, 2, 0)])
        assert np.array_equal(
            array.matrix,
            np.array([[0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0]]).reshape(3, 4, 1),
        )

    def test_dual_occupancy_expels_atoms(self):
        array = AtomArray(shape=[3, 4])
        array.matrix = np.array(
            [[0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0]], dtype=np.uint8
        ).reshape(3, 4, 1)
        [failed_moves, flags], _ = array.move_atoms(move_list=[Move(1, 2, 0, 2)])
        assert len(failed_moves) == 0
        assert MultiOccupancyFlag.MULTI_ATOM_OCCUPANCY in flags
        assert np.array_equal(
            array.matrix,
            np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]]).reshape(3, 4, 1),
        )

    def test_static_crossed_tweezer_expels_atoms(self):
        array = AtomArray(shape=[3, 4])
        array.matrix = np.array(
            [[0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0]], dtype=np.uint8
        ).reshape(3, 4, 1)
        # checking that collisions expel both atoms
        inevitable_collision_moves = [Move(1, 2, 0, 2), Move(0, 2, 0, 2)]
        [failed_moves, flags], _ = array.move_atoms(
            move_list=inevitable_collision_moves
        )
        assert (
            inevitable_collision_moves[0].fail_event == FailureEvent.COLLISION_AVOIDABLE
        )
        assert (
            inevitable_collision_moves[1].fail_event
            == FailureEvent.COLLISION_INEVITABLE
        )
        assert failed_moves == [0, 1]
        assert FailureFlag.LOSS in flags
        assert np.array_equal(
            array.matrix,
            np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]]).reshape(3, 4, 1),
        )

    @pytest.mark.xfail(
        reason="Collision logic in aod_timing is incomplete",
        strict=True,
    )
    def test_crossed_moving_tweezers_expel_atoms(self):
        array = AtomArray(shape=[3, 4])
        array.matrix = np.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]], dtype=np.uint8
        ).reshape(3, 4, 1)
        # checking that crossed tweezers expel atoms
        crossed_moves = [Move(1, 1, 2, 0), Move(2, 1, 1, 0)]
        [failed_moves, flags], _ = array.move_atoms(move_list=crossed_moves)
        assert failed_moves == [0, 1]
        assert crossed_moves[0].fail_event == FailureEvent.COLLISION_AVOIDABLE
        assert crossed_moves[1].fail_event == FailureEvent.COLLISION_AVOIDABLE
        assert FailureFlag.LOSS in flags
        assert np.array_equal(
            array.matrix,
            np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]).reshape(3, 4, 1),
        )

    # 2. Dual species
    def test_move_atoms_raises_on_negative_occupancy_dual_species(self) -> None:
        """
        Document the physical-state guard: the occupancy representation must not
        contain negative values. This is mainly a regression tripwire for accidental
        dtype changes (e.g., if `matrix` ever becomes signed).
        """
        aa = AtomArray(shape=[2, 2], n_species=2)

        # Force a signed dtype to simulate a bug/regression where negatives become possible.
        aa.matrix = aa.matrix.astype(np.int8, copy=True)
        aa.matrix[0, 0, 0] = np.int8(-1)

        m = Move(from_row=0, from_col=0, to_row=0, to_col=1)

        with pytest.raises(ValueError, match="negative occupancy"):
            aa.move_atoms([m])

    def test_single_move_dual_species(self):

        # 1. Single species
        array = AtomArray(shape=[3, 4], n_species=2)
        array.matrix[:, :, 0] = np.array(
            [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]], dtype=np.uint8
        )
        array.matrix[:, :, 1] = np.array(
            [[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
        )
        # checking that moves get row/col assignments correct
        _ = array.move_atoms(move_list=[Move(0, 1, 1, 1)])
        expected_mat = np.zeros([3, 4, 2])
        expected_mat[:, :, 0] = np.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.uint8
        )
        expected_mat[:, :, 1] = np.array(
            [[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
        )
        assert np.array_equal(array.matrix, expected_mat)

    def test_chain_move_dual_species(self):
        array = AtomArray(shape=[3, 4], n_species=2)
        array.matrix[:, :, 0] = np.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.uint8
        )
        array.matrix[:, :, 1] = np.array(
            [[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
        )
        # checking that moves of sites next to one another work properly
        _ = array.move_atoms(move_list=[Move(1, 0, 1, 1), Move(1, 1, 1, 2)])
        expected_mat = np.zeros([3, 4, 2])
        expected_mat[:, :, 0] = np.array(
            [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]], dtype=np.uint8
        )
        expected_mat[:, :, 1] = np.array(
            [[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
        )
        assert np.array_equal(array.matrix, expected_mat)

    def test_chain_move_with_absent_atom_dual_species(self):
        array = AtomArray(shape=[3, 4], n_species=2)
        array.matrix[:, :, 0] = np.array(
            [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]], dtype=np.uint8
        )
        array.matrix[:, :, 1] = np.array(
            [[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
        )
        # checking that atoms are not used in two moves
        _ = array.move_atoms(move_list=[Move(2, 2, 2, 1), Move(2, 1, 2, 0)])
        expected_mat = np.zeros([3, 4, 2])
        expected_mat[:, :, 0] = np.array(
            [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=np.uint8
        )
        expected_mat[:, :, 1] = np.array(
            [[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
        )
        assert np.array_equal(array.matrix, expected_mat)

    def test_dual_occupancy_expels_atoms_dual_species(self):
        array = AtomArray(shape=[3, 4], n_species=2)
        array.matrix[:, :, 0] = np.array(
            [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=np.uint8
        )
        array.matrix[:, :, 1] = np.array(
            [[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
        )
        # checking that collisions expel both atoms
        [failed_moves, flags], _ = array.move_atoms(move_list=[Move(1, 2, 0, 2)])
        expected_mat = np.zeros([3, 4, 2])
        expected_mat[:, :, 0] = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]], dtype=np.uint8
        )
        expected_mat[:, :, 1] = np.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
        )
        assert np.array_equal(array.matrix, expected_mat)
        assert failed_moves == []
        assert MultiOccupancyFlag.MULTI_ATOM_OCCUPANCY in flags

    def test_static_crossed_tweezer_expels_atoms_dual_species(self):
        array = AtomArray(shape=[3, 4], n_species=2)
        array.matrix[:, :, 0] = np.array(
            [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=np.uint8
        )
        array.matrix[:, :, 1] = np.array(
            [[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
        )
        # checking that collisions expel both atoms
        inevitable_collision_moves = [Move(1, 2, 0, 2), Move(0, 2, 0, 2)]
        [failed_moves, flags], _ = array.move_atoms(
            move_list=inevitable_collision_moves
        )
        assert failed_moves == [0, 1]
        assert (
            inevitable_collision_moves[0].fail_event == FailureEvent.COLLISION_AVOIDABLE
        )
        assert (
            inevitable_collision_moves[1].fail_event
            == FailureEvent.COLLISION_INEVITABLE
        )
        assert FailureFlag.LOSS in flags
        expected_mat = np.zeros([3, 4, 2])
        expected_mat[:, :, 0] = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]], dtype=np.uint8
        )
        expected_mat[:, :, 1] = np.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
        )
        assert np.array_equal(array.matrix, expected_mat)

    @pytest.mark.xfail(
        reason="Incomplete grid count bug in move_atoms",
        strict=True,
    )
    def test_crossed_moving_tweezers_expel_atoms_dual_species(self):
        array = AtomArray(shape=[3, 4], n_species=2)
        array.matrix[:, :, 0] = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]], dtype=np.uint8
        )
        array.matrix[:, :, 1] = np.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
        )
        # checking that crossed tweezers expel atoms
        crossed_moves = [Move(1, 1, 2, 0), Move(2, 1, 1, 0)]
        _ = array.move_atoms(move_list=crossed_moves)
        assert crossed_moves[0].fail_event == FailureEvent.COLLISION_AVOIDABLE
        assert crossed_moves[1].fail_event == FailureEvent.COLLISION_AVOIDABLE
        expected_mat = np.zeros([3, 4, 2])
        expected_mat[:, :, 0] = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
        )
        expected_mat[:, :, 1] = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
        )
        assert np.array_equal(array.matrix, expected_mat)

    def test_REGRESSION_crunch_moves(self) -> None:
        array = AtomArray(shape=[14, 14])
        array.matrix[:, :, 0] = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1],
                [1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        crunch_moves = [
            Move(8, 0, 8, 1),
            Move(8, 1, 8, 2),
            Move(8, 2, 8, 3),
            Move(8, 5, 8, 4),
            Move(8, 6, 8, 5),
            Move(8, 7, 8, 6),
            Move(8, 8, 8, 7),
            Move(8, 9, 8, 8),
            Move(8, 10, 8, 9),
            Move(8, 11, 8, 10),
            Move(8, 12, 8, 11),
            Move(8, 13, 8, 12),
        ]
        _ = array.move_atoms(move_list=crunch_moves)
        expected_mat = np.zeros([14, 14, 1])
        expected_mat[:, :, 0] = np.array(
            [
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    0,
                ],
                [
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    0,
                ],
                [
                    1,
                    0,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    0,
                    1,
                    0,
                ],
                [
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                ],
                [
                    0,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                ],
                [
                    0,
                    0,
                    1,
                    1,
                    0,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ],
            dtype=np.uint8,
        )
        assert np.array_equal(array.matrix, expected_mat)

    def test_REGRESSION_crunch_moves_single_row(self) -> None:
        array = AtomArray(shape=[1, 14])
        array.matrix[:, :, 0] = np.array(
            [[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1]], dtype=np.uint8
        )
        expected_mat = np.zeros([1, 14, 1])
        expected_mat[:, :, 0] = np.array(
            [[0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0]], dtype=np.uint8
        )
        crunch_moves = [
            Move(0, 0, 0, 1),
            Move(0, 1, 0, 2),
            Move(0, 2, 0, 3),
            Move(0, 5, 0, 4),
            Move(0, 6, 0, 5),
            Move(0, 7, 0, 6),
            Move(0, 8, 0, 7),
            Move(0, 9, 0, 8),
            Move(0, 10, 0, 9),
            Move(0, 11, 0, 10),
            Move(0, 12, 0, 11),
            Move(0, 13, 0, 12),
        ]
        _ = array.move_atoms(move_list=crunch_moves)

        assert np.array_equal(array.matrix, expected_mat)


class TestEvaluateMoves:
    def test_empty_move_list(self):
        array = AtomArray(shape=[3, 4])
        array.matrix = np.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]], dtype=np.uint8
        ).reshape(3, 4, 1)

        moves = []
        time, [n_parallel, n_non_parallel] = array.evaluate_moves(move_set=moves)
        assert time == 0
        assert n_parallel == 0
        assert n_non_parallel == 0
        assert np.array_equal(
            array.matrix,
            np.array(
                [[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]], dtype=np.uint8
            ).reshape(3, 4, 1),
        )

    def test_parallel_move_and_solitary_move(self):
        array = AtomArray(shape=[3, 4])
        array.matrix = np.array(
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]], dtype=np.uint8
        ).reshape(3, 4, 1)

        moves = [[Move(1, 1, 1, 2), Move(2, 1, 2, 2)], [Move(1, 2, 1, 3)]]
        time, [n_parallel, n_non_parallel] = array.evaluate_moves(move_set=moves)
        assert time != 0
        assert n_parallel == 2
        assert n_non_parallel == 3
        assert np.array_equal(
            array.matrix,
            np.array(
                [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.uint8
            ).reshape(3, 4, 1),
        )

    def test_crossed_move_and_solitary_move(self):
        array = AtomArray(shape=[3, 4])
        array.matrix = np.array(
            [[0, 0, 0, 1], [0, 1, 0, 0], [0, 1, 0, 0]], dtype=np.uint8
        ).reshape(3, 4, 1)

        moves = [[Move(1, 1, 2, 2), Move(2, 1, 1, 2)], [Move(0, 3, 1, 3)]]
        time, [n_parallel, n_non_parallel] = array.evaluate_moves(move_set=moves)
        assert time != 0
        assert n_parallel == 2
        assert n_non_parallel == 3
        assert np.array_equal(
            array.matrix,
            np.array(
                [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=np.uint8
            ).reshape(3, 4, 1),
        )


class TestEjectDualOccupiedSitesInplace:
    def test_post_apply_cleanup_preserves_matrix_dtype_single_species(self) -> None:
        """
        Multi-occupancy cleanup should not accidentally upcast the array dtype.

        In this package, `matrix` dtype discipline matters because downstream mask
        logic and occupancy checks assume a compact integer representation.
        """
        aa = AtomArray(shape=[2, 2], n_species=1)
        assert aa.matrix.dtype == np.uint8

        aa.matrix[1, 1, 0] = np.uint8(2)  # multi-occupancy

        m = Move(from_row=0, from_col=0, to_row=1, to_col=1)
        m.movetype = MoveType.LEGAL_MOVE
        m.set_failure_event(int(FailureEvent.SUCCESS))
        move_list = [m]

        source_rows = np.asarray([mv.from_row for mv in move_list], dtype=np.int_)
        source_cols = np.asarray([mv.from_col for mv in move_list], dtype=np.int_)

        flags = aa._eject_dual_occupied_sites_inplace(
            move_list, source_rows, source_cols
        )

        assert aa.matrix.dtype == np.uint8
        assert aa.matrix[1, 1, 0] == np.uint8(0)
        assert flags == [MultiOccupancyFlag.MULTI_ATOM_OCCUPANCY]

    def test_post_apply_cleanup_enforces_values_in_0_1_single_species(
        self, empty_3x3_atomarray
    ) -> None:
        """
        Invariant test: after multi-occupancy ejection, the occupancy representation
        should be physical again (values only in {0,1} for single species).
        """
        aa = empty_3x3_atomarray
        aa.matrix[0, 0, 0] = np.uint8(2)  # inject non-physical occupancy

        m = Move(from_row=1, from_col=1, to_row=0, to_col=0)
        m.movetype = MoveType.LEGAL_MOVE
        m.set_failure_event(int(FailureEvent.SUCCESS))
        move_list = [m]

        source_rows = np.asarray([mv.from_row for mv in move_list], dtype=np.int_)
        source_cols = np.asarray([mv.from_col for mv in move_list], dtype=np.int_)

        aa._eject_dual_occupied_sites_inplace(move_list, source_rows, source_cols)

        assert np.max(aa.matrix[:, :, 0]) <= np.uint8(1)
        assert np.all(
            (aa.matrix[:, :, 0] == np.uint8(0)) | (aa.matrix[:, :, 0] == np.uint8(1))
        )

    def test_post_apply_cleanup_enforces_values_in_0_1_dual_species_total(self) -> None:
        """
        Invariant test (dual species): after cleanup, each tweezer should have
        total occupancy <= 1 across species.
        """
        aa = AtomArray(shape=[2, 2], n_species=2)
        assert aa.matrix.dtype == np.uint8

        # Inject a multi-occupied tweezer: (1,1) has both species present.
        aa.matrix[1, 1, 0] = np.uint8(1)
        aa.matrix[1, 1, 1] = np.uint8(1)

        m = Move(from_row=0, from_col=0, to_row=1, to_col=1)
        m.movetype = MoveType.LEGAL_MOVE
        m.set_failure_event(int(FailureEvent.SUCCESS))
        move_list = [m]

        source_rows = np.asarray([mv.from_row for mv in move_list], dtype=np.int_)
        source_cols = np.asarray([mv.from_col for mv in move_list], dtype=np.int_)

        aa._eject_dual_occupied_sites_inplace(move_list, source_rows, source_cols)

        total = (aa.matrix[:, :, 0] + aa.matrix[:, :, 1]).astype(np.uint8, copy=False)
        assert np.max(total) <= np.uint8(1)
        assert np.all((total == np.uint8(0)) | (total == np.uint8(1)))

    def _mk_move(
        self,
        from_row: int,
        from_col: int,
        to_row: int,
        to_col: int,
        movetype: MoveType,
        fail_event: FailureEvent,
    ) -> Move:
        """
        Abbreviated helper.

        Creates a `Move` with explicit `movetype` and failure event, so tests can
        target post-apply multi-occupancy tagging rules without depending on the
        move-classification pipeline.
        """
        m = Move(from_row=from_row, from_col=from_col, to_row=to_row, to_col=to_col)
        m.movetype = movetype
        m.set_failure_event(int(fail_event))
        return m

    def test_empty_move_list_noop(self) -> None:
        """
        The post-apply multi-occupancy cleanup is a batch operation; an empty batch
        should not mutate state or emit flags.
        """
        aa = AtomArray(shape=[2, 2], n_species=1)
        move_list: list[Move] = []
        source_rows = np.asarray([], dtype=np.int_)
        source_cols = np.asarray([], dtype=np.int_)

        flags = aa._eject_dual_occupied_sites_inplace(
            move_list, source_rows, source_cols
        )

        assert flags == []
        assert np.all(aa.matrix == np.uint8(0))

    def test_no_multi_occupancy_emits_no_flags_and_does_not_mutate(self) -> None:
        """
        If the post-apply array state already satisfies the occupancy invariant,
        this helper should be a no-op (no ejection, no tagging).
        """
        aa = AtomArray(shape=[2, 2], n_species=1)
        aa.matrix[0, 0, 0] = np.uint8(1)

        m0 = self._mk_move(0, 0, 0, 1, MoveType.LEGAL_MOVE, FailureEvent.SUCCESS)
        move_list = [m0]
        source_rows = np.asarray([m.from_row for m in move_list], dtype=np.int_)
        source_cols = np.asarray([m.from_col for m in move_list], dtype=np.int_)

        before = aa.matrix.copy()
        flags = aa._eject_dual_occupied_sites_inplace(
            move_list, source_rows, source_cols
        )

        assert flags == []
        assert np.array_equal(aa.matrix, before)
        assert not hasattr(m0, "multi_occupancy_flag")

    def test_single_species_tags_moves_that_target_multi_occupied_tweezer(self) -> None:
        """
        Destination-based tagging: any move whose destination is a multi-occupied
        tweezer should be tagged, regardless of that move's own failure mode.
        """
        aa = AtomArray(shape=[2, 2], n_species=1)
        aa.matrix[1, 1, 0] = np.uint8(2)  # multi-occupancy

        m_hit = self._mk_move(0, 0, 1, 1, MoveType.LEGAL_MOVE, FailureEvent.SUCCESS)
        m_miss = self._mk_move(0, 1, 1, 0, MoveType.LEGAL_MOVE, FailureEvent.SUCCESS)
        move_list = [m_hit, m_miss]
        source_rows = np.asarray([m.from_row for m in move_list], dtype=np.int_)
        source_cols = np.asarray([m.from_col for m in move_list], dtype=np.int_)

        flags = aa._eject_dual_occupied_sites_inplace(
            move_list, source_rows, source_cols
        )

        assert aa.matrix[1, 1, 0] == np.uint8(0)
        assert flags == [MultiOccupancyFlag.MULTI_ATOM_OCCUPANCY]
        assert m_hit.multi_occupancy_flag == MultiOccupancyFlag.MULTI_ATOM_OCCUPANCY
        assert not hasattr(m_miss, "multi_occupancy_flag")

    def test_single_species_tags_source_move_only_if_atom_left_behind(self) -> None:
        """
        Source-based tagging is conservative: a move is tagged for its source only
        when it plausibly failed while leaving an atom behind (e.g., pickup fail).
        """
        aa = AtomArray(shape=[2, 2], n_species=1)
        aa.matrix[0, 0, 0] = np.uint8(2)  # multi-occupancy at source/destination

        # Move A: failed pickup -> atom plausibly remains at source.
        m_left_behind = self._mk_move(
            0, 0, 0, 1, MoveType.LEGAL_MOVE, FailureEvent.PICKUP_FAIL
        )
        # Move B: targets the collided tweezer (dest tagging).
        m_into_collision = self._mk_move(
            1, 1, 0, 0, MoveType.LEGAL_MOVE, FailureEvent.SUCCESS
        )

        move_list = [m_left_behind, m_into_collision]
        source_rows = np.asarray([m.from_row for m in move_list], dtype=np.int_)
        source_cols = np.asarray([m.from_col for m in move_list], dtype=np.int_)

        flags = aa._eject_dual_occupied_sites_inplace(
            move_list, source_rows, source_cols
        )

        assert aa.matrix[0, 0, 0] == np.uint8(0)
        assert sorted(flags) == sorted([MultiOccupancyFlag.MULTI_ATOM_OCCUPANCY] * 2)
        assert (
            m_left_behind.multi_occupancy_flag
            == MultiOccupancyFlag.MULTI_ATOM_OCCUPANCY
        )
        assert (
            m_into_collision.multi_occupancy_flag
            == MultiOccupancyFlag.MULTI_ATOM_OCCUPANCY
        )

    def test_single_species_no_atom_moves_are_not_tagged_via_source_rule(self) -> None:
        """
        A `NO_ATOM` move is excluded from "left atom behind" source tagging, since it
        represents a bookkeeping/eligibility condition rather than a failed transport
        that left a physical atom at the source.
        """
        aa = AtomArray(shape=[2, 2], n_species=1)
        aa.matrix[0, 0, 0] = np.uint8(2)

        m_no_atom = self._mk_move(0, 0, 0, 1, MoveType.LEGAL_MOVE, FailureEvent.NO_ATOM)
        m_into_collision = self._mk_move(
            1, 1, 0, 0, MoveType.LEGAL_MOVE, FailureEvent.SUCCESS
        )

        move_list = [m_no_atom, m_into_collision]
        source_rows = np.asarray([m.from_row for m in move_list], dtype=np.int_)
        source_cols = np.asarray([m.from_col for m in move_list], dtype=np.int_)

        flags = aa._eject_dual_occupied_sites_inplace(
            move_list, source_rows, source_cols
        )

        assert aa.matrix[0, 0, 0] == np.uint8(0)
        assert flags == [MultiOccupancyFlag.MULTI_ATOM_OCCUPANCY]
        assert not hasattr(m_no_atom, "multi_occupancy_flag")
        assert (
            m_into_collision.multi_occupancy_flag
            == MultiOccupancyFlag.MULTI_ATOM_OCCUPANCY
        )

    def test_eject_moves_are_not_tagged_via_source_rule(self) -> None:
        """
        Explicit ejection moves are excluded from "left atom behind" source tagging,
        since they are not contributors to multi-occupancy in the intended diagnostic
        sense (they are a cleanup operation themselves).
        """
        aa = AtomArray(shape=[2, 2], n_species=1)
        aa.matrix[0, 0, 0] = np.uint8(2)

        m_eject = self._mk_move(
            0, 0, 0, 1, MoveType.EJECT_MOVE, FailureEvent.PICKUP_FAIL
        )
        m_into_collision = self._mk_move(
            1, 1, 0, 0, MoveType.LEGAL_MOVE, FailureEvent.SUCCESS
        )

        move_list = [m_eject, m_into_collision]
        source_rows = np.asarray([m.from_row for m in move_list], dtype=np.int_)
        source_cols = np.asarray([m.from_col for m in move_list], dtype=np.int_)

        flags = aa._eject_dual_occupied_sites_inplace(
            move_list, source_rows, source_cols
        )

        assert aa.matrix[0, 0, 0] == np.uint8(0)
        assert flags == [MultiOccupancyFlag.MULTI_ATOM_OCCUPANCY]
        assert not hasattr(m_eject, "multi_occupancy_flag")
        assert (
            m_into_collision.multi_occupancy_flag
            == MultiOccupancyFlag.MULTI_ATOM_OCCUPANCY
        )

    def test_dual_species_multi_occupancy_detects_sum_over_species_and_ejects(
        self,
    ) -> None:
        """
        For dual-species arrays, a tweezer is multi-occupied if total occupancy across
        species exceeds one (e.g., (1,1) at the same tweezer).
        """
        aa = AtomArray(shape=[2, 2], n_species=2)
        aa.matrix[1, 1, 0] = np.uint8(1)
        aa.matrix[1, 1, 1] = np.uint8(1)  # total=2 -> multi-occupied

        m_hit = self._mk_move(0, 0, 1, 1, MoveType.LEGAL_MOVE, FailureEvent.SUCCESS)
        move_list = [m_hit]
        source_rows = np.asarray([m.from_row for m in move_list], dtype=np.int_)
        source_cols = np.asarray([m.from_col for m in move_list], dtype=np.int_)

        flags = aa._eject_dual_occupied_sites_inplace(
            move_list, source_rows, source_cols
        )

        assert np.all(aa.matrix[1, 1, :] == np.uint8(0))
        assert flags == [MultiOccupancyFlag.MULTI_ATOM_OCCUPANCY]
        assert m_hit.multi_occupancy_flag == MultiOccupancyFlag.MULTI_ATOM_OCCUPANCY


def _make_atomarray_single(matrix2d_or3d: NDArray) -> AtomArray:
    aa = AtomArray.__new__(AtomArray)
    aa.n_species = 1
    aa.shape = matrix2d_or3d.shape[:2]  # IMPORTANT: rows, cols only
    aa.matrix = np.array(matrix2d_or3d, copy=True)
    return aa


def _make_atomarray_dual(matrix3d: NDArray) -> AtomArray:
    aa = AtomArray.__new__(AtomArray)
    aa.n_species = 2
    aa.shape = matrix3d.shape[:2]  # IMPORTANT: rows, cols only
    aa.matrix = np.array(matrix3d, copy=True)
    return aa


class TestHelpers:
    def test_atomarray_test_helper_preserves_matrix_after_shape_assignment(
        self,
    ) -> None:
        mat = np.array([[[1], [0]]], dtype=np.uint8)
        aa = _make_atomarray_single(mat)
        assert np.array_equal(aa.matrix, mat)

    def test_atomarray_test_helper_preserves_dual_matrix_after_shape_assignment(
        self,
    ) -> None:
        mat = np.zeros((1, 3, 2), dtype=np.uint8)
        mat[0, 1, 0] = 1
        aa = _make_atomarray_single(mat)
        assert np.array_equal(aa.matrix, mat)


class TestSetAttr:
    def test_setting_shape_resets_single_species_buffers(self) -> None:
        """
        Document the simulator invariant: changing `AtomArray.shape` is treated as a
        *re-initialization event*.

        In this codebase, `shape` is not “metadata”; it defines the physical lattice.
        So, when contributors resize the lattice, the state/targets are intentionally
        wiped to a consistent all-zero configuration to avoid silently carrying
        incompatible state into a new geometry.
        """
        aa: AtomArray = AtomArray(shape=[2, 3], n_species=1)

        # Populate buffers with nonzero values to ensure the reset is observable.
        aa.matrix[:, :, :] = np.uint8(1)
        aa.target[:, :, :] = np.uint8(1)
        aa.target_Rb[:, :] = np.uint8(1)
        aa.target_Cs[:, :] = np.uint8(1)

        new_shape: list[int] = [4, 5]
        aa.shape = new_shape

        expected_matrix_shape: tuple[int, int, int] = (4, 5, 1)
        expected_target_shape: tuple[int, int, int] = (4, 5, 1)
        expected_2d_shape: tuple[int, int] = (4, 5)

        assert aa.matrix.shape == expected_matrix_shape
        assert aa.target.shape == expected_target_shape
        assert aa.target_Rb.shape == expected_2d_shape
        assert aa.target_Cs.shape == expected_2d_shape

        assert aa.matrix.dtype == np.uint8
        assert aa.target.dtype == np.uint8
        assert aa.target_Rb.dtype == np.uint8
        assert aa.target_Cs.dtype == np.uint8

        assert np.all(aa.matrix == np.uint8(0))
        assert np.all(aa.target == np.uint8(0))
        assert np.all(aa.target_Rb == np.uint8(0))
        assert np.all(aa.target_Cs == np.uint8(0))

    def test_setting_shape_resets_dual_species_buffers(self) -> None:
        """
        Dual-species arrays keep per-species occupancy/targets in the last axis of
        `matrix`/`target`, while `target_Rb`/`target_Cs` are 2D “species-specific”
        convenience targets.

        This test documents that a `shape` change resets *all* of these buffers to
        a clean consistent state, since the lattice itself has changed.
        """
        aa: AtomArray = AtomArray(shape=[2, 3], n_species=2)

        aa.matrix[:, :, :] = np.uint8(1)
        aa.target[:, :, :] = np.uint8(1)
        aa.target_Rb[:, :] = np.uint8(1)
        aa.target_Cs[:, :] = np.uint8(1)

        new_shape: list[int] = [4, 5]
        aa.shape = new_shape

        expected_matrix_shape: tuple[int, int, int] = (4, 5, 2)
        expected_target_shape: tuple[int, int, int] = (4, 5, 2)
        expected_2d_shape: tuple[int, int] = (4, 5)

        assert aa.matrix.shape == expected_matrix_shape
        assert aa.target.shape == expected_target_shape
        assert aa.target_Rb.shape == expected_2d_shape
        assert aa.target_Cs.shape == expected_2d_shape

        assert aa.matrix.dtype == np.uint8
        assert aa.target.dtype == np.uint8
        assert aa.target_Rb.dtype == np.uint8
        assert aa.target_Cs.dtype == np.uint8

        assert np.all(aa.matrix == np.uint8(0))
        assert np.all(aa.target == np.uint8(0))
        assert np.all(aa.target_Rb == np.uint8(0))
        assert np.all(aa.target_Cs == np.uint8(0))


class TestApplyMoves:
    def test_apply_moves_single_species_success_legal_move(self) -> None:
        aa = _make_atomarray_single(np.array([[[1], [0], [0]]], dtype=np.uint8))
        moves = [Move(0, 0, 0, 1)]

        assert aa.matrix.shape == (1, 3, 1)
        assert aa.matrix[0, 0, 0] == 1
        assert moves[0].from_row == 0
        assert moves[0].from_col == 0
        assert moves[0].to_row == 0
        assert moves[0].to_col == 1
        failed_inds, flags = aa._apply_moves_single_species(moves)

        assert failed_inds == []
        assert flags == []
        assert moves[0].movetype == MoveType.LEGAL_MOVE
        assert moves[0].fail_flag == FailureFlag.SUCCESS

        expected = np.array([[[0], [1], [0]]], dtype=np.uint8)
        assert np.array_equal(aa.matrix, expected)

    def test_apply_two_moves_single_species_success_legal_move(self) -> None:
        aa = _make_atomarray_single(np.array([[[1], [1], [0]]], dtype=np.uint8))
        moves = [Move(0, 0, 0, 1), Move(0, 1, 0, 2)]

        failed_inds, flags = aa._apply_moves_single_species(moves)

        assert failed_inds == []
        assert flags == []
        assert moves[0].movetype == MoveType.LEGAL_MOVE
        assert moves[0].fail_flag == FailureFlag.SUCCESS
        assert moves[1].movetype == MoveType.LEGAL_MOVE
        assert moves[1].fail_flag == FailureFlag.SUCCESS

        expected = np.array([[[0], [1], [1]]], dtype=np.uint8)
        assert np.array_equal(aa.matrix, expected)

    def test_apply_moves_single_species_no_pickup_leaves_atom_at_source(self) -> None:
        aa = _make_atomarray_single(np.array([[[1], [0], [0]]], dtype=np.uint8))
        moves = [Move(0, 0, 0, 1)]
        moves[0].set_failure_event(FailureEvent.PICKUP_FAIL)

        failed_inds, flags = aa._apply_moves_single_species(moves)

        assert failed_inds == [0]
        assert flags == [int(FailureFlag.NO_PICKUP)]
        assert moves[0].fail_flag == FailureFlag.NO_PICKUP
        assert moves[0].movetype == MoveType.LEGAL_MOVE  # geometry still legal

        expected = np.array([[[1], [0], [0]]], dtype=np.uint8)
        assert np.array_equal(aa.matrix, expected)

    def test_apply_moves_single_species_occupied_destination_not_vacated_is_illegal(
        self,
    ) -> None:
        aa = _make_atomarray_single(np.array([[[1], [1], [0]]], dtype=np.uint8))
        moves = [Move(0, 0, 0, 1)]  # destination occupied, and no move vacates (0,1)

        aa._apply_moves_single_species(moves)

        assert moves[0].movetype == MoveType.ILLEGAL_MOVE

    def test_apply_two_moves_single_species_no_pickup_collision(self) -> None:
        aa = _make_atomarray_single(np.array([[[1], [1], [0]]], dtype=np.uint8))
        moves = [Move(0, 0, 0, 1), Move(0, 1, 0, 2)]
        moves[1].set_failure_event(FailureEvent.PICKUP_FAIL)

        failed_inds, flags = aa._apply_moves_single_species(moves)

        assert failed_inds == [1]
        assert flags == [int(FailureFlag.NO_PICKUP)]
        assert moves[0].movetype == MoveType.LEGAL_MOVE
        assert moves[0].fail_flag == FailureFlag.SUCCESS
        assert moves[1].movetype == MoveType.LEGAL_MOVE
        assert moves[1].fail_flag == FailureFlag.NO_PICKUP

        expected = np.array([[[0], [2], [0]]], dtype=np.uint8)
        assert np.array_equal(aa.matrix, expected)

    def test_apply_moves_single_species_loss_removes_atom_from_source(self) -> None:
        aa = _make_atomarray_single(np.array([[[1], [0], [0]]], dtype=np.uint8))
        moves = [Move(0, 0, 0, 1)]
        moves[0].set_failure_event(FailureEvent.DECEL_FAIL)

        failed_inds, flags = aa._apply_moves_single_species(moves)

        assert failed_inds == [0]
        assert flags == [int(FailureFlag.LOSS)]
        assert moves[0].fail_flag == FailureFlag.LOSS
        assert moves[0].movetype == MoveType.LEGAL_MOVE

        expected = np.array([[[0], [0], [0]]], dtype=np.uint8)
        assert np.array_equal(aa.matrix, expected)

    def test_apply_moves_single_species_crossed_moving_removes_atom_from_source(
        self,
    ) -> None:
        aa = _make_atomarray_single(np.array([[[1], [0], [0]]], dtype=np.uint8))
        moves = [Move(0, 0, 0, 1), Move(0, 1, 0, 0)]
        moves[0].set_failure_event(FailureEvent.COLLISION_AVOIDABLE)
        moves[1].set_failure_event(FailureEvent.NO_ATOM)

        failed_inds, flags = aa._apply_moves_single_species(moves)

        assert failed_inds == [0, 1]
        assert flags == [int(FailureFlag.LOSS), int(FailureFlag.NO_ATOM)]
        assert moves[0].fail_flag == FailureFlag.LOSS
        assert moves[1].fail_flag == FailureFlag.NO_ATOM

        expected = np.array([[[0], [0], [0]]], dtype=np.uint8)
        assert np.array_equal(aa.matrix, expected)

    def test_apply_moves_single_species_crossed_static_removes_atom_from_source(
        self,
    ) -> None:
        aa = _make_atomarray_single(np.array([[[1], [0], [0]]], dtype=np.uint8))
        moves = [Move(0, 0, 0, 0), Move(0, 1, 0, 0)]
        moves[0].set_failure_event(FailureEvent.COLLISION_INEVITABLE)
        moves[1].set_failure_event(FailureEvent.NO_ATOM)

        failed_inds, flags = aa._apply_moves_single_species(moves)

        assert failed_inds == [0, 1]
        assert flags == [int(FailureFlag.LOSS), int(FailureFlag.NO_ATOM)]
        assert moves[0].fail_flag == FailureFlag.LOSS
        assert moves[1].fail_flag == FailureFlag.NO_ATOM

        expected = np.array([[[0], [0], [0]]], dtype=np.uint8)
        assert np.array_equal(aa.matrix, expected)

    def test_apply_moves_single_species_noop_empty_list(self) -> None:
        aa = _make_atomarray_single(np.zeros((2, 3, 1), dtype=np.uint8))
        before = aa.matrix.copy()

        failed_inds, flags = aa._apply_moves_single_species([])

        assert failed_inds == []
        assert flags == []
        assert np.array_equal(aa.matrix, before)

    def test_apply_moves_single_species_eject_success_removes_source_only(self) -> None:
        aa = _make_atomarray_single(np.array([[[1], [0], [0]]], dtype=np.uint8))
        # destination column 3 is out-of-bounds for shape (1,3)
        moves = [Move(0, 0, 0, -1)]

        failed_inds, flags = aa._apply_moves_single_species(moves)

        assert failed_inds == []
        assert flags == []
        assert moves[0].movetype == MoveType.EJECT_MOVE
        assert moves[0].fail_flag == FailureFlag.SUCCESS

        expected = np.array([[[0], [0], [0]]], dtype=np.uint8)
        assert np.array_equal(aa.matrix, expected)

    def test_apply_moves_dual_species_success_preserves_species_identity(self) -> None:
        mat = np.zeros((1, 3, 2), dtype=np.uint8)
        mat[0, 0, 0] = 1
        aa = _make_atomarray_dual(mat)

        moves = [Move(0, 0, 0, 1)]
        failed_inds, flags = aa._apply_moves_dual_species(moves)

        assert failed_inds == []
        assert flags == []
        assert moves[0].movetype == MoveType.LEGAL_MOVE
        assert moves[0].fail_flag == FailureFlag.SUCCESS

        expected = np.zeros((1, 3, 2), dtype=np.uint8)
        expected[0, 1, 0] = 1
        assert np.array_equal(aa.matrix, expected)

    def test_apply_moves_dual_species_chain_moves_classified_legal(self) -> None:
        mat = np.zeros((1, 3, 2), dtype=np.uint8)
        mat[0, 0, 0] = 1
        mat[0, 1, 1] = 1
        aa = _make_atomarray_dual(mat)

        moves = [Move(0, 0, 0, 1), Move(0, 1, 0, 2)]
        failed_inds, flags = aa._apply_moves_dual_species(moves)

        assert failed_inds == []
        assert flags == []
        assert moves[0].movetype == MoveType.LEGAL_MOVE
        assert moves[1].movetype == MoveType.LEGAL_MOVE

        expected = np.zeros((1, 3, 2), dtype=np.uint8)
        expected[0, 1, 0] = 1
        expected[0, 2, 1] = 1
        assert np.array_equal(aa.matrix, expected)

    def test_apply_moves_dual_species_loss_removes_correct_species(self) -> None:
        mat = np.zeros((1, 3, 2), dtype=np.uint8)
        mat[0, 0, 1] = 1  # species 1 at source
        aa = _make_atomarray_dual(mat)

        moves = [Move(0, 0, 0, 1)]
        moves[0].set_failure_event(FailureEvent.COLLISION_INEVITABLE)

        failed_inds, flags = aa._apply_moves_dual_species(moves)

        assert failed_inds == [0]
        assert flags == [int(FailureFlag.LOSS)]
        assert moves[0].fail_flag == FailureFlag.LOSS

        expected = np.zeros((1, 3, 2), dtype=np.uint8)
        assert np.array_equal(aa.matrix, expected)

    def test_apply_moves_dual_species_noop_empty_list(self) -> None:
        aa = _make_atomarray_dual(np.zeros((2, 3, 2), dtype=np.uint8))
        before = aa.matrix.copy()

        failed_inds, flags = aa._apply_moves_dual_species([])

        assert failed_inds == []
        assert flags == []
        assert np.array_equal(aa.matrix, before)
