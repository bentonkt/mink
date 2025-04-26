"""Damping task implementation."""

from __future__ import annotations

import mujoco
import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from .posture_task import PostureTask


class DampingTask(PostureTask):
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

    Example:

    .. code-block:: python

        # Uniform damping across all degrees of freedom.
        damping_task = DampingTask(model, cost=1.0)

        # Custom damping.
        cost = np.zeros(model.nv)
        cost[:3] = 1.0  # High damping for the first 3 joints.
        cost[3:] = 0.1  # Lower damping for the remaining joints.
        damping_task = DampingTask(model, cost)
    """

    def __init__(self, model: mujoco.MjModel, cost: npt.ArrayLike):
        super().__init__(model, cost, 0.0, 0.0)

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        return np.zeros(configuration.nv)
