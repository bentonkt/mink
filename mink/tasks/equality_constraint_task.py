"""Equality constraint task implementation."""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import mujoco
import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from .exceptions import InvalidConstraint, TaskDefinitionError
from .task import Task


def _get_constraint_dim(constraint: mujoco.mjtEq) -> int:
    """Return the dimension of an equality constraint in the efc* arrays."""
    return {
        mujoco.mjtEq.mjEQ_CONNECT.value: 3,
        mujoco.mjtEq.mjEQ_WELD.value: 6,
        mujoco.mjtEq.mjEQ_JOINT.value: 1,
        mujoco.mjtEq.mjEQ_TENDON.value: 1,
    }[constraint]


def _get_cost_dim(eq_types: np.ndarray) -> int:
    """Get the total size of the cost vector for a set of equality constraints."""
    dim: int = 0
    for eq_type in eq_types:
        dim += _get_constraint_dim(eq_type)
    return dim


def _get_equality_constraint_indices(
    data: mujoco.MjData, eq_ids: np.ndarray
) -> np.ndarray:
    """Get indices of a given equality constraint if it is active."""
    return np.isin(data.efc_id, eq_ids)


class EqualityConstraintTask(Task):
    """Regulate equality constraints in a model.

    Equality constraints are useful, among other things, for modeling "loop joints"
    such as four-bar linkages. In MuJoCo, there are several types of equality
    constraints, including:

    * ``mjEQ_CONNECT``: Connect two bodies at a point (ball joint).
    * ``mjEQ_WELD``: Fix relative pose of two bodies.
    * ``mjEQ_JOINT``: couple the values of two scalar joints
    * ``mjEQ_TENDON``: couple the values of two tendons

    This task can regulate all equality constraints in a model or a specific subset
    identified by name or ID.

    Attributes:
        equalities: ID or name of the equality constraints to regulate. If not provided,
            the task will regulate all equality constraints in the model.
        cost: Cost vector for the equality constraint task. Either a scalar, in which
            case the same cost is applied to all constraints, or a vector of shape
            ``(neq,)``, where ``neq`` is the number of equality constraints in the
            model.

    Raises:
        InvalidConstraint: If a specified equality constraint name or ID is not found.
        TaskDefinitionError: If no equality constraints are found or if cost parameters
            have invalid shape or values.

    Example:

    .. code-block:: python

        # Regulate all equality constraints with the same cost.
        eq_task = EqualityConstraintTask(model, cost=1.0)

        # Regulate specific equality constraints with different costs.
        eq_task = EqualityConstraintTask(
            model,
            cost=[1.0, 0.5],
            equalities=["connect_right", "connect_left"]
        )
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        cost: npt.ArrayLike,
        equalities: Optional[Sequence[int | str]] = None,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        self._mask: np.ndarray | None = None  # Active equality constraint mask.
        self._neq_active: int | None = None  # Number of active equality constraints.

        eq_ids: list[int] = []
        if equalities is not None:
            for eq_id_or_name in equalities:
                eq_id: int
                if isinstance(eq_id_or_name, str):
                    eq_id = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_id_or_name
                    )
                    if eq_id == -1:
                        raise InvalidConstraint(
                            f"Equality constraint '{eq_id_or_name}' not found."
                        )
                else:
                    eq_id = eq_id_or_name
                    if eq_id < 0 or eq_id >= model.neq:
                        raise InvalidConstraint(
                            f"Equality constraint index {eq_id} out of range."
                            f"Must be in range [0, {model.neq})."
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
        self._neq_total = len(self._eq_ids)  # Total number of equality constraints.

        cost_dim = _get_cost_dim(self._eq_types)
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
        repeats = [_get_constraint_dim(eq_type) for eq_type in self._eq_types]
        self.cost[:] = np.repeat(cost, repeats)

    def _update_constraint_info(self, configuration: Configuration) -> None:
        self._mask = _get_equality_constraint_indices(configuration.data, self._eq_ids)
        self._neq_active = np.sum(self._mask)

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        """Compute the equality constraint task error.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Equality constraint task error vector :math:`e(q)`. The shape of the
            error vector is ``(neq_active * constraint_dim,)``, where ``neq_active``
            is the number of active equality constraints, and ``constraint_dim``
            depends on the type of equality constraint.
        """
        self._update_constraint_info(configuration)
        return configuration.data.efc_pos[self._mask]

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        """Compute the task Jacobian at a given configuration.

        Args:
            configuration: Robot configuration :math:`q`.

        Returns:
            Equality constraint task jacobian :math:`J(q)`. The shape of the Jacobian
            is ``(neq_active * constraint_dim, nv)``, where ``neq_active`` is the
            number of active equality constraints, ``constraint_dim`` depends on the
            type of equality constraint, and ``nv`` is the dimension of the tangent
            space.
        """
        self._update_constraint_info(configuration)
        data = configuration.data
        J_efc = data.efc_J.reshape((data.nefc, configuration.model.nv))
        return J_efc[self._mask]
