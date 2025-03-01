"""Constraints task implementation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import mujoco

from ..configuration import Configuration
from .task import Task


class ConstraintsTask(Task):
    def __init__(
        self,
        constraint_cost: npt.ArrayLike,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        super().__init__(cost=np.zeros(1), gain=gain, lm_damping=lm_damping)
        self._constraint_cost = constraint_cost
        self._mask = None
        self._n_constraints = None

    def _update_constraint_info(self, configuration: Configuration) -> None:
        data = configuration.data
        self._mask = data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        self._n_constraints = np.sum(self._mask)
        if self._n_constraints > 0:
            self.cost = np.full(self._n_constraints, self._constraint_cost)

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        if self._mask is None:
            self._update_constraint_info(configuration)
        data = configuration.data
        return data.efc_pos[self._mask]

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        if self._mask is None:
            self._update_constraint_info(configuration)
        data = configuration.data
        J_efc = data.efc_J.reshape((data.nefc, configuration.model.nv))
        return J_efc[self._mask]