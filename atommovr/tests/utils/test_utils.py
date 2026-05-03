# Tests for utility functions (in progress)

import numpy as np

import pytest

from atommovr.utils.Move import Move
from atommovr.utils.move_utils import (
    get_move_list_from_AOD_cmds,
    get_AOD_cmds_from_move_list,
    MoveType,
)


class TestAODToMoveListAndViceVersa:
    def test_move_list_AOD_conversion(self):
        move = [Move(3, 5, 4, 6)]
        horiz_AOD_cmds, vert_AOD_cmds, parallel_success_flag = (
            get_AOD_cmds_from_move_list(np.zeros([10, 10, 1]), move)
        )
        move_list = get_move_list_from_AOD_cmds(horiz_AOD_cmds, vert_AOD_cmds)
        assert move[0].to_col == move_list[0].to_col
        assert move[0].to_row == move_list[0].to_row
        assert move[0].from_col == move_list[0].from_col
        assert move[0].from_row == move_list[0].from_row

    def test_REGRESSION_move_list_aod_roundtrip_preserves_row_col_order(self) -> None:
        move_seq = [Move(1, 4, 2, 3)]  # (from_row, from_col, to_row, to_col)

        horiz_AOD_cmds, vert_AOD_cmds, parallel_success_flag = (
            get_AOD_cmds_from_move_list(np.zeros((10, 10, 1), dtype=np.uint8), move_seq)
        )
        assert parallel_success_flag is True
        move_list = get_move_list_from_AOD_cmds(horiz_AOD_cmds, vert_AOD_cmds)
        assert len(move_list) == 1
        assert move_list[0] == move_seq[0]  # uses Move.__eq__

    def test_REGRESSION_move_list_aod_roundtrip_non_square_matrix(self) -> None:
        move_seq = [Move(2, 7, 3, 6)]
        matrix = np.zeros((6, 12, 1), dtype=np.uint8)  # non-square

        horiz_AOD_cmds, vert_AOD_cmds, parallel_success_flag = (
            get_AOD_cmds_from_move_list(matrix, move_seq)
        )
        assert parallel_success_flag is True

        roundtrip = get_move_list_from_AOD_cmds(horiz_AOD_cmds, vert_AOD_cmds)
        assert len(roundtrip) == 1
        assert roundtrip[0] == move_seq[0]


class TestMove:
    def test_move_and_repr(self):
        move = Move(0, 2, 1, 3)
        assert move._move_str() == move.__repr__()
        assert move.__repr__() == "(0, 2) -> (1, 3)"
        assert move.from_row == 0
        assert move.from_col == 2
        assert move.to_row == 1
        assert move.to_col == 3

    def test_basic_initialization(self):
        move = Move(1, 3, 0, 4)
        assert move.from_row == 1
        assert move.from_col == 3
        assert move.to_row == 0
        assert move.to_col == 4

    def test_delta_calculation(self):
        move = Move(3, 1, 4, 0)
        assert move.dx == -1
        assert move.dy == 1

    def test_negative_delta(self):
        move = Move(1, 1, 0, 1)
        assert move.dx == 0
        assert move.dy == -1

    def test_distance_calculation(self):
        move = Move(0, 0, 1, 1)
        assert move.distance == np.sqrt(2)

    def test_midpoint_calculation(self):
        move = Move(1, 0, 2, -1)
        assert move.midx == -0.5
        assert move.midy == 1.5

    def test_equality(self):
        move1 = Move(1, 3, 2, 4)
        move2 = Move(1, 3, 2, 4)
        move3 = Move(1, 3, 2, 3)
        assert move1 == move2
        assert move1 != move3
        assert move1 != (1, 2, 3, 4)

    def test_zero_distance_move(self):
        move = Move(2, 3, 2, 3)
        assert move.distance == 0
        assert move.dx == 0
        assert move.dy == 0

    @pytest.mark.parametrize(['from_row', 'from_col', 'to_row', 'to_col'],
                             [[1, 0, 3, 1],
                              [3, 2, 2, 0]])
    def test_move_raises_on_greater_than_one_displacement(self, from_row, from_col, to_row, to_col):
        with pytest.raises(ValueError):
            _ = Move(from_row, from_col, to_row, to_col)

class TestMoveType:
    def test_enum_values(self):
        assert MoveType.ILLEGAL_MOVE == 0
        assert MoveType.LEGAL_MOVE == 1
        assert MoveType.EJECT_MOVE == 2
        assert MoveType.NO_ATOM_TO_MOVE == 3

    def test_enum_count(self):
        assert len(MoveType) == 4
