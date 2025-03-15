"""Tests for equality_constraint_task.py."""

import numpy as np
from absl.testing import absltest
from robot_descriptions.loaders.mujoco import load_robot_description

from mink import Configuration
from mink.tasks import EqualityConstraintTask, TaskDefinitionError


class TestEqualityConstraintTask(absltest.TestCase):
    """Test consistency of the equality constraint task."""

    def test_no_equality_constraint_throws(self):
        model = load_robot_description("ur5e_mj_description")
        with self.assertRaises(TaskDefinitionError) as cm:
            EqualityConstraintTask(model=model, cost=1.0)
        expected_error_message = (
            "EqualityConstraintTask no equality constraints found in this model."
        )
        self.assertEqual(str(cm.exception), expected_error_message)

    def test_wrong_cost_dim_throws(self):
        model = load_robot_description("cassie_mj_description")
        # Cassie has 4 equality constraints of type connect. The cost should
        # either be a scalar or a vector of shape (4,).
        with self.assertRaises(TaskDefinitionError) as cm:
            EqualityConstraintTask(model=model, cost=(1, 2))
        expected_error_message = (
            "EqualityConstraintTask cost must be a vector of shape (1,) "
            "or (4,). Got (2,)."
        )
        self.assertEqual(str(cm.exception), expected_error_message)

    def test_cost_correctly_broadcast(self):
        model = load_robot_description("cassie_mj_description")
        # Each connect constraint has dimension 3, so the cost dimension should be 12.
        task = EqualityConstraintTask(model=model, cost=1.0)
        np.testing.assert_array_equal(task.cost, np.full((12,), 1.0))
        task = EqualityConstraintTask(model=model, cost=[1, 2, 3, 4])
        expected_cost = np.asarray([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        np.testing.assert_array_equal(task.cost, expected_cost)

    def test_cost_throws_if_negative(self):
        model = load_robot_description("cassie_mj_description")
        with self.assertRaises(TaskDefinitionError) as cm:
            EqualityConstraintTask(model=model, cost=[-1, 2, 3, 4])
        expected_error_message = "EqualityConstraintTask cost must be >= 0"
        self.assertEqual(str(cm.exception), expected_error_message)

    def test_zero_error_when_constraint_is_satisfied(self):
        model = load_robot_description("cassie_mj_description")
        task = EqualityConstraintTask(model=model, cost=1.0)
        configuration = Configuration(model)
        configuration.update(model.qpos0)
        error = task.compute_error(configuration)
        np.testing.assert_array_almost_equal(error, np.zeros_like(task.cost), decimal=8)

    def test_zero_cost_same_as_disabling_task(self):
        model = load_robot_description("cassie_mj_description")
        task = EqualityConstraintTask(model=model, cost=0.0)
        configuration = Configuration(model)
        configuration.update_from_keyframe("home")
        objective = task.compute_qp_objective(configuration)
        x = np.random.random(configuration.nv)
        cost = objective.value(x)
        self.assertAlmostEqual(cost, 0.0)


if __name__ == "__main__":
    absltest.main()
