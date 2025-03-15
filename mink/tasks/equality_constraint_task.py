"""Equality constraint task implementation."""

from __future__ import annotations

import logging
from typing import Optional

import mujoco
import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from .exceptions import InvalidConstraint, TaskDefinitionError
from .task import Task


def get_constraint_dim(constraint: mujoco.mjtEq) -> int:
    """Return the dimension of an equality constraint in the efc* arrays."""
    return {
        mujoco.mjtEq.mjEQ_CONNECT.value: 3,
        mujoco.mjtEq.mjEQ_WELD.value: 6,
        mujoco.mjtEq.mjEQ_JOINT.value: 1,
        mujoco.mjtEq.mjEQ_TENDON.value: 1,
    }[constraint]


def get_cost_dim(eq_types: np.ndarray) -> int:
    """Get the total size of the cost vector for a set of equality constraints."""
    dim: int = 0
    for eq_type in eq_types:
        dim += get_constraint_dim(eq_type)
    return dim


def get_equality_constraint_indices(
    data: mujoco.MjData, eq_ids: np.ndarray
) -> np.ndarray:
    """Get indices of a given equality constraint if it is active."""
    return np.isin(data.efc_id, eq_ids)


class EqualityConstraintTask(Task):
    """Equality constraint task.

    Attributes:
        equality_name: Name of the equality constraint to regulate. If not provided,
            the task will regulate all equality constraints in the model.
        cost: Cost vector for the equality constraint task. Either a scalar, in which
            case the same cost is applied to all constraints, or a vector of shape
            `(neq,)`, where `neq` is the number of equality constraints in the model.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        cost: npt.ArrayLike,
        equality_name: Optional[str] = None,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        self._mask = None  # Active equality constraint mask.
        self._neq_total = model.neq  # Total number of equality constraints.
        self._neq_active = 0  # Number of active equality constraints.

        eq_ids = []
        if equality_name is not None:
            eq_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_name
            )
            if eq_id == -1:
                raise InvalidConstraint(
                    f"Equality constraint '{equality_name}' not found"
                )
            eq_ids.append(eq_id)
        else:
            eq_ids = list(range(model.neq))
            logging.warning("Regulating %d equality constraints", len(eq_ids))
        if len(eq_ids) == 0:
            raise TaskDefinitionError(
                f"{self.__class__.__name__} no equality constraints found in this "
                "model."
            )
        self._eq_ids = np.array(eq_ids)
        self._eq_types = model.eq_type[self._eq_ids]

        cost_dim = get_cost_dim(self._eq_types)
        super().__init__(cost=np.zeros((cost_dim,)), gain=gain, lm_damping=lm_damping)
        self.set_cost(cost)

    def set_cost(self, cost: npt.ArrayLike) -> None:
        cost = np.atleast_1d(cost)
        if cost.ndim != 1 or cost.shape[0] not in (1, self._neq_total):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} cost must be a vector of shape (1,) "
                f"or ({self._neq_total},). Got {cost.shape}."
            )
        if not np.all(cost >= 0.0):
            raise TaskDefinitionError(f"{self.__class__.__name__} cost must be >= 0")

        # For scalar cost, broadcast to all elements.
        if cost.shape[0] == 1:
            self.cost[:] = cost[0]
            return

        # For vector cost, repeat each element according to its constraint dimension.
        repeats = [get_constraint_dim(eq_type) for eq_type in self._eq_types]
        self.cost[:] = np.repeat(cost, repeats)

    def _update_constraint_info(self, configuration: Configuration) -> None:
        self._mask = get_equality_constraint_indices(configuration.data, self._eq_ids)
        self._neq_active = np.sum(self._mask)

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
