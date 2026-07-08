"""
Authors: Nikhil Harle.

Description:
A `Move` class for streamlined processing/integration with algorithms and benchmarking
(see docstrings for more details).
"""

import numpy as np
from typing import Tuple

from atommovr.utils.failure_policy import (
    FailureEvent,
    FailureFlag,
    FAILURE_EVENT_TO_FLAG,
)


class Move:
    """
    Represents an operation to transfer an atom from one tweezer site to another.

    Parameters
    ----------
    from_row : int
        Source row index.
    from_col : int
        Source column index.
    to_row : int
        Destination row index.
    to_col : int
        Destination column index.
    fail_event : FailureEvent, optional
        Initial move outcome event.

    Attributes
    ----------
    from_row : int
        Source row index.
    from_col : int
        Source column index.
    to_row : int
        Destination row index.
    to_col : int
        Destination column index.
    dx : int
        Column displacement.
    dy : int
        Row displacement.
    distance : float
        Euclidean move distance in units of spacing.
    midx : float
        Midpoint column coordinate.
    midy : float
        Midpoint row coordinate.
    fail_event : FailureEvent
        Move-level failure event.
    fail_flag : FailureFlag
        Coarser failure classification derived from ``fail_event``.
    """

    def __init__(
        self,
        from_row: int,
        from_col: int,
        to_row: int,
        to_col: int,
        fail_event: FailureEvent = FailureEvent.SUCCESS,
    ) -> None:
        self.from_row = from_row
        self.from_col = from_col
        self.to_row = to_row
        self.to_col = to_col

        self.dx = to_col - from_col
        self.dy = to_row - from_row

        self.distance = self._get_distance()
        self.midx, self.midy = self._get_move_midpoint()

        self.fail_event = fail_event
        self._update_fail_flag()

    def __repr__(self) -> str:
        return self._move_str()

    def __eq__(self, other) -> bool:
        if isinstance(other, Move):
            return (
                self.from_row == other.from_row
                and self.from_col == other.from_col
                and self.to_row == other.to_row
                and self.to_col == other.to_col
            )
        return False

    def set_failure_event(self, fail_event: int) -> None:
        """
        Set the move failure event and refresh the corresponding failure flag.

        Parameters
        ----------
        fail_event : int
            Enumeration member of the ``FailureEvent`` class.

        Returns
        -------
        None
            Updates ``fail_event`` and ``fail_flag`` in place.

        Raises
        ------
        ValueError
            If ``fail_event`` is not a recognized failure event.
        """
        self.fail_event = fail_event
        self._update_fail_flag()

    def _update_fail_flag(self) -> None:
        """
        Helper function for `.assign_failure_event()`.
        Automatically updates failure flag based on failure event.

        Mapping: `FailureEvent` -> `FailureFlag`
        - SUCCESS -> SUCCESS
        - PICKUP_FAIL -> NO_PICKUP (atom stays at source)
        - NO_ATOM -> NO_ATOM (no atom to move)
        - PUTDOWN_FAIL, CROSSED_STATIC, CROSSED_MOVING, ACCEL_FAIL, DECEL_FAIL, TRANSPORT_FAIL -> LOSS (atom is lost)
        """
        if self.fail_event not in FAILURE_EVENT_TO_FLAG:
            raise ValueError(f"Invalid fail_event: {self.fail_event}")

        self.fail_flag = FAILURE_EVENT_TO_FLAG[self.fail_event]

    def is_successful(self) -> bool:
        """Check if move succeeded."""
        return self.fail_flag == FailureFlag.SUCCESS

    def atom_was_lost(self) -> bool:
        """Check if atom was lost during this move."""
        return self.fail_flag == FailureFlag.LOSS

    def atom_stayed_at_source(self) -> bool:
        """Check if atom remained at source position."""
        return self.fail_flag == FailureFlag.NO_PICKUP

    def _get_distance(self) -> float:
        """Compute the Euclidean move distance."""
        return np.sqrt((self.dx) ** 2 + (self.dy) ** 2)

    def _move_str(self) -> str:
        """Format the move as a human-readable source-to-destination string."""
        return f"({self.from_row}, {self.from_col}) -> ({self.to_row}, {self.to_col})"

    def _get_move_midpoint(self) -> Tuple[float, float]:
        """Compute the geometric midpoint of the move."""
        self.midx = self.from_col + self.dx / 2
        self.midy = self.from_row + self.dy / 2
        return self.midx, self.midy
