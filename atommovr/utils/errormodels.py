# Collection of ErrorModel objects representing various loss processes

import numpy as np

from atommovr.utils.core import atom_loss, atom_loss_dual
from atommovr.utils.ErrorModel import ErrorModel


class ZeroNoise(ErrorModel):
    """
    Simulates errorless rearrangement (assumes perfect tweezers
    and an infinitely long vacuum-limited lifetime).
    """

    def __init__(
        self,
        putdown_time: float = 0,
        pickup_time: float = 0,
        accel_time: float = 0,
        decel_time: float = 0,
        seed: int | None = None,
    ):
        super().__init__(
            putdown_time=putdown_time,
            pickup_time=pickup_time,
            accel_time=accel_time,
            decel_time=decel_time,
            pickup_fail_rate=0.0,
            putdown_fail_rate=0.0,
            accel_fail_rate=0.0,
            decel_fail_rate=0.0,
            lifetime=np.inf,
            seed=seed,
        )
        self.name = "ZeroNoise"

    def __repr__(self) -> str:
        return self.name

    def apply_accel_errors_mask(
        self, event_mask: np.ndarray, eligible: np.ndarray
    ) -> None:
        pass

    def apply_decel_errors_mask(
        self, event_mask: np.ndarray, eligible: np.ndarray
    ) -> None:
        pass

    def apply_pickup_errors_mask(
        self, event_mask: np.ndarray, eligible: np.ndarray
    ) -> None:
        pass

    def apply_putdown_errors_mask(
        self, event_mask: np.ndarray, eligible: np.ndarray
    ) -> None:
        pass

    def get_atom_loss(
        self, state: np.ndarray, evolution_time: float, n_species: int = 1
    ) -> tuple[np.ndarray, bool]:
        r"""
        Given the current state of the atom array, applies any general loss process
        over the period $\Delta t$ = evolution_time.

        For this error model, it just returns the same state.

        ## Parameters
        state : np.ndarray
            the current state of the atom array.
        evolution_time : float
            the time over which we calculate the loss process (usually the time
            for a single move or set of parallel moves).
        - n_species : int, optional (default = 1)
            the number of atomic species (single, dual).

        ## Returns
        new_state : np.ndarray
            the state after the loss process.
        loss_flag : bool
            1 if any atom loss occurred, 0 if not.
        """
        if n_species not in [1, 2]:
            raise ValueError(f'Parameter "n_species" must be 1 or 2, not {n_species}')
        loss_flag = False
        new_state = state.copy()
        return new_state, loss_flag


class UniformVacuumTweezerError(ErrorModel):
    """
    Considers atom loss due to imperfect vacuum
    (i.e. collisions with background gas particles)
    and uniform tweezer failure rates.

    ## Parameters
     - `pickup_fail_rate` (optional): float, between 0 and 1.
     Probability that an atom to be moved will be not picked up by the moving tweezer
    (in this case, the atom is not lost but just stays in its original spot). Default
    value is 0.01 (1%).
     - `putdown_fail_rate` (optional): float, between 0 and 1.
    Probability that an atom to be moved will be picked up by the moving tweezer, but
    will be subsequently lost in the transfer to the new tweezer. Default value is
    0.01 (1%).
     - `lifetime` (optional): float.
    Vacuum limited lifetime of an individual atom (assumed to be uniform for all atoms),
    in seconds. Default value is 30.
    """

    def __init__(
        self,
        putdown_time: float = 0,
        pickup_time: float = 0,
        accel_time: float = 0,
        decel_time: float = 0,
        pickup_fail_rate: float = 0.01,
        putdown_fail_rate: float = 0.01,
        accel_fail_rate: float = 0,
        decel_fail_rate: float = 0,
        lifetime: float = 30,
        seed: int | None = None,
    ):
        super().__init__(
            putdown_time=putdown_time,
            pickup_time=pickup_time,
            accel_time=accel_time,
            decel_time=decel_time,
            pickup_fail_rate=pickup_fail_rate,
            putdown_fail_rate=putdown_fail_rate,
            accel_fail_rate=accel_fail_rate,
            decel_fail_rate=decel_fail_rate,
            lifetime=lifetime,
            seed=seed,
        )
        self.name = "UniformVacuumTweezerError"

    def __repr__(self) -> str:
        return self.name

    def get_atom_loss(
        self, state: np.ndarray, evolution_time: float, n_species: int = 1
    ) -> tuple[np.ndarray, bool]:
        r"""
        Given the current state of the atom array, applies any general loss process
        over the period $\Delta t$ = evolution_time.

        For this error model, we consider uniform loss from background gas particles
        knocking atoms out of their traps.

        ## Parameters
        - state (np.ndarray). The current state of the atom array.
        - evolution_time (float). The time over which we calculate the loss process
        (usually the time for a single move).
        - n_species (int, must be 1 or 2). The number of atomic species (single, dual).

        ## Returns
        - new_state (np.ndarray). The state after the loss process.
        - loss_flag (bool). 1 if any atom loss occurred, 0 if not.
        """
        if n_species not in [1, 2]:
            raise ValueError(f'Parameter "n_species" must be 1 or 2, not {n_species}')
        evolution_time = evolution_time
        if n_species == 1:
            new_state, loss_flag = atom_loss(
                state, evolution_time, self.lifetime, self.rng
            )
        elif n_species == 2:
            new_state, loss_flag = atom_loss_dual(
                state, evolution_time, self.lifetime, self.rng
            )
        else:
            raise ValueError(
                f"Parameter 'n_species' must be either 1 or 2, not {n_species}."
            )
        return new_state, loss_flag
