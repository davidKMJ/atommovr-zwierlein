# Core object for simulating noise and loss processes

import numpy as np

from atommovr.utils.failure_policy import FailureBit
from atommovr.utils.error_utils import (
    apply_bernoulli_event_inplace,
    set_event_bit_inplace,
)


class ErrorModel:
    """
    Base class for move-related and background-loss error processes.

    `ErrorModel` defines the interface used by `AtomArray.move_atoms()` to:
    1) tag per-move failure events in a vectorized event bitmask, and
    2) apply array-wide atom loss between move rounds (e.g. vacuum-limited lifetime).

    Design intent
    -------------
    - Subclass this class (or a built-in child class) to implement custom error models.
    - Preserve method signatures and return types so models remain compatible with
    `AtomArray.move_atoms()` and the event-resolution pipeline.
    - Child classes should call `super().__init__(...)` so required attributes
    (especially `self.rng`) are initialized.

    Integration points
    ------------------
    `AtomArray.move_atoms()` uses an `ErrorModel` instance to:
    - apply stochastic move-related failures via `apply_*_errors_mask(...)`
    - tag deterministic crossing events via `apply_crossed_*_mask(...)`
    - apply suppression / primary-event resolution (via failure policy utilities)
    - write primary failure events back to `Move` objects
    - apply background atom loss via `get_atom_loss(...)`

    Required attributes (used by the move pipeline)
    -----------------------------------------------
    Implementations are expected to expose the following attributes:
    - `rng` : numpy.random.Generator
        Random number generator used for deterministic / reproducible sampling.
    - `pickup_time`, `putdown_time`, `accel_time`, `decel_time` : float
        Time costs (s) used in move-time accounting.
    - `pickup_fail_rate`, `putdown_fail_rate`, `accel_fail_rate`, `decel_fail_rate` : float
        Bernoulli failure probabilities in [0, 1] for move-related processes.
    - `lifetime` : float
        Characteristic lifetime (s) for background atom-loss processes.

    Method contract (shape / dtype expectations)
    --------------------------------------------
    Vectorized mask methods mutate `event_mask` in-place and must not return anything:
    - `apply_pickup_errors_mask(event_mask, eligible) -> None`
    - `apply_putdown_errors_mask(event_mask, eligible) -> None`
    - `apply_accel_errors_mask(event_mask, eligible) -> None`
    - `apply_decel_errors_mask(event_mask, eligible) -> None`
    - `apply_inevitable_collision_mask(event_mask, eligible) -> None`
    - `apply_avoidable_collision_mask(event_mask, eligible) -> None`

    Expected inputs:
    - `event_mask`: `np.ndarray` of shape `(N,)`, dtype `np.uint64`
        Per-move bitmask storing one or more `FailureBit` values.
    - `eligible`: `np.ndarray` of shape `(N,)`, dtype `bool`
        Boolean mask selecting which moves are affected by the process.

    Background loss method:
    - `get_atom_loss(state, evolution_time, n_species=1) -> tuple[np.ndarray, bool]`

    Expected behavior:
    - Returns `(new_state, loss_flag)` where:
    - `new_state` is a NumPy array with same shape as `state`
    - `loss_flag` is `True` iff any atom was lost
    - `n_species` is currently expected to be 1 or 2.

    Default behavior
    ----------------
    This base class provides generic Bernoulli implementations for the move-related
    mask methods and deterministic crossed-event tagging. The default `get_atom_loss`
    implementation is a no-op (returns the input state unchanged with `loss_flag=False`).

    Notes for custom subclasses
    ---------------------------
    - To disable a process, override the corresponding method with a no-op.
    - To extend a standard process, call `super().method_name(...)` and then add
    custom logic.
    - If you override `__init__`, call `super().__init__(...)`.
    """

    def __init__(
        self,
        putdown_time: float = 0,
        pickup_time: float = 0,
        accel_time: float = 0,
        decel_time: float = 0,
        pickup_fail_rate: float = 0,
        putdown_fail_rate: float = 0,
        accel_fail_rate: float = 0,
        decel_fail_rate: float = 0,
        lifetime: float = np.inf,
        seed: int | None = None,
    ):
        self.name = "Generic ErrorModel object"
        self.rng = np.random.default_rng(seed)

        self.putdown_time = putdown_time  # s
        self.pickup_time = pickup_time  # s
        self.accel_time = accel_time  # s
        self.decel_time = decel_time  # s

        self.pickup_fail_rate = pickup_fail_rate
        self.putdown_fail_rate = putdown_fail_rate
        self.accel_fail_rate = accel_fail_rate
        self.decel_fail_rate = decel_fail_rate

        self.lifetime = lifetime  # s

    def __repr__(self) -> str:
        return self.name

    def apply_pickup_errors_mask(
        self, event_mask: np.ndarray, eligible: np.ndarray
    ) -> None:
        """
        Apply pickup failures to eligible moves by setting FailureBit.PICKUP_FAIL.
        Updates `event_mask` in-place.
        """
        apply_bernoulli_event_inplace(
            event_mask=event_mask,
            eligible=eligible,
            p_fail=float(self.pickup_fail_rate),
            bit=FailureBit.PICKUP_FAIL,
            rng=self.rng,
        )

    def apply_putdown_errors_mask(
        self, event_mask: np.ndarray, eligible: np.ndarray
    ) -> None:
        """
        Apply putdown failures to eligible moves by setting FailureBit.PUTDOWN_FAIL.
        Updates `event_mask` in-place.
        """
        apply_bernoulli_event_inplace(
            event_mask=event_mask,
            eligible=eligible,
            p_fail=float(self.putdown_fail_rate),
            bit=FailureBit.PUTDOWN_FAIL,
            rng=self.rng,
        )

    def apply_accel_errors_mask(
        self, event_mask: np.ndarray, eligible: np.ndarray
    ) -> None:
        """
        Apply acceleration failures to eligible moves by setting FailureBit.ACCEL_FAIL.
        Updates `event_mask` in-place.
        """
        apply_bernoulli_event_inplace(
            event_mask=event_mask,
            eligible=eligible,
            p_fail=float(self.accel_fail_rate),
            bit=FailureBit.ACCEL_FAIL,
            rng=self.rng,
        )

    def apply_decel_errors_mask(
        self, event_mask: np.ndarray, eligible: np.ndarray
    ) -> None:
        """
        Apply deceleration failures to eligible moves by setting FailureBit.DECEL_FAIL.
        Updates `event_mask` in-place.
        """
        apply_bernoulli_event_inplace(
            event_mask=event_mask,
            eligible=eligible,
            p_fail=float(self.decel_fail_rate),
            bit=FailureBit.DECEL_FAIL,
            rng=self.rng,
        )

    def apply_inevitable_collision_mask(
        self, event_mask: np.ndarray, eligible: np.ndarray
    ) -> None:
        """
        Deterministically tag victims of inevitable collisions (always lost in policy).

        Updates `event_mask` in-place by setting FailureBit.COLLISION_INEVITABLE.
        """
        set_event_bit_inplace(
            event_mask=event_mask,
            eligible=eligible,
            bit=FailureBit.COLLISION_INEVITABLE,
        )

    def apply_avoidable_collision_mask(
        self, event_mask: np.ndarray, eligible: np.ndarray
    ) -> None:
        """
        Deterministically tag victims of avoidable collisions.

        Policy will later suppress this bit if PICKUP_FAIL occurred for that move.
        Updates `event_mask` in-place by setting FailureBit.COLLISION_AVOIDABLE.
        """
        set_event_bit_inplace(
            event_mask=event_mask,
            eligible=eligible,
            bit=FailureBit.COLLISION_AVOIDABLE,
        )

    def get_atom_loss(
        self, state: np.ndarray, evolution_time: float, n_species: int = 1
    ) -> tuple[np.ndarray, bool]:
        """
        Simulates a general loss process and returns the modified state of
        the array.

        :param state: the current state of the array
        :type state: np.ndarray
        :param evolution_time: the time it took to execute the last time of moves, or the time since the error syndrome was last simulated.
        :type evolution_time: float
        :param n_species: the number of atomic species in the array.
        :type n_species: int, optional (default = 1)
        """
        loss_flag = False
        return state, loss_flag
