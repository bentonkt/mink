"""Constraints task implementation."""

from __future__ import annotations
from typing import Optional

import numpy as np
import numpy.typing as npt
import mujoco

from ..configuration import Configuration
from .task import Task
from .exceptions import InvalidConstraint


class EqualityConstraintTask(Task):
    """Equality constraint task.

    Attributes:
        equality_name: Name of the equality constraint to regulate. If not provided,
            the task will regulate all equality constraints in the model.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        constraint_cost: npt.ArrayLike,
        equality_name: Optional[str] = None,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        super().__init__(cost=np.zeros(1), gain=gain, lm_damping=lm_damping)
        self._constraint_cost = constraint_cost
        self._mask = None
        self._n_constraints = None

        if equality_name is not None:
            eq_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_name
            )
            if eq_id == -1:
                raise InvalidConstraint(
                    f"Equality constraint '{equality_name}' not found"
                )
            self._eq_id = eq_id
        else:
            self._eq_id = None

    def _update_constraint_info(self, configuration: Configuration) -> None:
        data = configuration.data
        if self._eq_id is None:
            self._mask = data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        else:
            self._mask = data.efc_id == self._eq_id
        self._n_constraints = np.sum(self._mask)
        if self._n_constraints > 0:
            self.cost = np.full(self._n_constraints, self._constraint_cost)

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        """Compute the equality constraint task error.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Equality constraint task error vector :math:`e(q)`.
        """
        self._update_constraint_info(configuration)
        return configuration.data.efc_pos[self._mask]

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        """Compute the task Jacobian at a given configuration.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Equality constraint task jacobian :math:`J(q)`.
        """
        self._update_constraint_info(configuration)
        data = configuration.data
        J_efc = data.efc_J.reshape((data.nefc, configuration.model.nv))
        return J_efc[self._mask]
