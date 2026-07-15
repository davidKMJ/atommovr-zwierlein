import numpy as np
import pytest

from atommovr.algorithms.source.ejection import ejection


class TestEjectionMethodParameter:
    """`ejection`'s `method` kwarg is a forward-compatible hook: only
    `"sublattice"` is implemented today, and passing anything else must raise
    `NotImplementedError` rather than silently falling through.
    """

    def test_unsupported_method_raises_not_implemented(self) -> None:
        matrix = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        target = np.array([[0, 1], [0, 0]], dtype=np.uint8)

        with pytest.raises(NotImplementedError):
            ejection(matrix, target, method="other")

    def test_default_method_is_sublattice_and_runs(self) -> None:
        matrix = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        target = np.array([[0, 1], [0, 0]], dtype=np.uint8)

        move_list, final_config = ejection(
            matrix, target, final_size=[0, 1, 0, 1], method="sublattice"
        )
        assert isinstance(move_list, list)
