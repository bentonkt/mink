"""Damping task implementation."""

from __future__ import annotations

import mujoco
import numpy as np
import numpy.typing as npt

from mink.tasks.task import Objective

from ..configuration import Configuration
from ..utils import get_freejoint_dims
from .exceptions import TaskDefinitionError
from .task import Task


class DampingTask(Task):
    r"""Minimize joint velocities.

    This damping task serves as a regularizer that minimizes the L2 norm of the joint
    velocities. This biases the solution toward the current configuration. A higher
    damping cost discourages motion and brings the robot to a stop if no other tasks
    are active.

    This task contributes the following quadratic penalty to the QP objective:

    .. math::
        \sum_i \lambda_i^2 \dot{q}_i^2,

    which acts as a weighted L2 regularization on joint velocities. The weight term
    :math:`\lambda_i` can be a scalar or a vector of shape ``(model.nv)``.

    Floating-base coordinates are not affected by this task.
    """

    def __init__(self, model: mujoco.MjModel, cost: npt.ArrayLike):
        super().__init__(
            cost=np.zeros((model.nv,)),
            gain=0.0,
            lm_damping=0.0,
        )

        self._v_ids: np.ndarray | None
        _, v_ids_or_none = get_freejoint_dims(model)
        if v_ids_or_none:
            self._v_ids = np.asarray(v_ids_or_none)
        else:
            self._v_ids = None

        self.k = model.nv
        self.nq = model.nq
        self.set_cost(cost)

    def set_cost(self, cost: npt.ArrayLike) -> None:
        cost = np.atleast_1d(cost)
        if cost.ndim != 1 or cost.shape[0] not in (1, self.k):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} cost must be a vector of shape (1,) "
                f"(aka identical cost for all dofs) or ({self.k},). Got {cost.shape}"
            )
        if not np.all(cost >= 0.0):
            raise TaskDefinitionError(f"{self.__class__.__name__} cost should be >= 0")
        self.cost[: self.k] = cost

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        raise NotImplementedError

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        raise NotImplementedError

    def compute_qp_objective(self, configuration: Configuration) -> Objective:
        jac = -np.eye(configuration.nv)
        if self._v_ids is not None:
            jac[:, self._v_ids] = 0.0

        weight = np.diag(self.cost)
        weighted_jacobian = weight @ jac

        H = weighted_jacobian.T @ weighted_jacobian
        c = np.zeros(configuration.nv)

        return Objective(H, c)
