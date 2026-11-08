"""Independent CPU checks of the solver's mathematical formulas.

These tests validate the range table and scalar RDP formulas, not CUDA execution.
In particular, they do not assert monotonicity of the full count-rescaled solver.
Run with: python -m unittest discover -s tests -p test_rdp.py
"""

import math
from pathlib import Path
import unittest

import numpy as np
from scipy.optimize import minimize_scalar

from reference import range_table_error, read_model


class RangeTableTests(unittest.TestCase):
    def test_dense_survival_table_error_all_isotopes(self):
        _, isotopes, bins = read_model(Path(__file__).resolve().parents[1] / "prc.cu")
        self.assertEqual(set(isotopes), {"11C", "13N", "15O", "18F", "64Cu", "68Ga", "82Rb", "124I"})
        for name, branches in isotopes.items():
            with self.subTest(isotope=name):
                self.assertLessEqual(range_table_error(branches, bins), 2e-5)


def penalty(x, y, gamma, epsilon=0.0):
    return (x - y) ** 2 / (x + y + gamma * abs(x - y) + epsilon)


def derivatives(x, y, gamma, epsilon=0.0):
    """Analytic first and second derivatives with respect to the first voxel."""
    d = x - y
    denominator = x + y + gamma * abs(d) + epsilon
    gradient = d * (x + 3 * y + gamma * abs(d) + 2 * epsilon) / denominator**2
    curvature = 2 * (2 * y + epsilon) ** 2 / denominator**3
    return gradient, curvature


def surrogate_step(old, em, gradient, curvature, beta):
    """Nonnegative optimizer of -em*log(x)+x+beta*(G*dx+C*dx^2/2)."""
    a = beta * curvature
    b = 1 + beta * (gradient - curvature * old)
    if a == 0:
        return em / b
    root = math.sqrt(b * b + 4 * a * em)
    if b < 0:
        return (root - b) / (2 * a)
    return 2 * em / (b + root) if em > 0 else 0.0


class RdpMathematicsTests(unittest.TestCase):
    def test_derivatives_against_finite_differences(self):
        # The deliberately non-negligible epsilon catches missing derivative terms.
        for gamma in (0.0, 2.0, 10.0, 50.0):
            for epsilon in (0.0, 0.17):
                for x, y in ((0.2, 1.3), (4.0, 0.7), (2.0, 2.0)):
                    h = 1e-5
                    actual_gradient, actual_curvature = derivatives(x, y, gamma, epsilon)
                    finite_gradient = (penalty(x + h, y, gamma, epsilon)
                                       - penalty(x - h, y, gamma, epsilon)) / (2 * h)
                    # Differentiate the first derivative to avoid cancellation in psi.
                    finite_curvature = (derivatives(x + h, y, gamma, epsilon)[0]
                                        - derivatives(x - h, y, gamma, epsilon)[0]) / (2 * h)
                    self.assertAlmostEqual(actual_gradient, finite_gradient, delta=2e-7)
                    self.assertAlmostEqual(actual_curvature, finite_curvature,
                                           delta=max(1e-8, abs(actual_curvature) * 4e-4))

    def test_root_matches_independent_scalar_minimization(self):
        rng = np.random.default_rng(42)
        for _ in range(80):
            old, em, curvature, beta = np.exp(rng.uniform(-2, 2, 4))
            gradient = rng.uniform(-5, 5)
            actual = surrogate_step(old, em, gradient, curvature, beta)

            def objective(x):
                delta = x - old
                return -em * math.log(x) + x + beta * (gradient * delta + curvature * delta**2 / 2)

            optimized = minimize_scalar(objective, bounds=(1e-12, 2 * actual + 1),
                                        method="bounded", options={"xatol": 1e-11})
            self.assertTrue(optimized.success)
            self.assertAlmostEqual(actual, optimized.x, delta=max(1e-7, actual * 1e-6))

    def test_zero_em_can_have_a_positive_prior_update(self):
        # For this case the objective is 2*x^2 - 2*x: its minimum is x=0.5.
        # An unconditional `if em == 0: return 0` is incorrect.
        actual = surrogate_step(old=0, em=0, gradient=-3, curvature=4, beta=1)
        self.assertEqual(actual, 0.5)
        self.assertEqual(surrogate_step(old=0, em=0, gradient=1, curvature=4, beta=1), 0)

    def test_rl_multiplicative_zero_locking(self):
        # A supported zero is absorbing for RL, although a positive initialization
        # can recover it. This is why initialization must not copy all data zeros.
        a = np.array([[0.8, 0.2], [0.2, 0.8]])
        data = np.array([0.0, 2.0])
        start = data.copy()
        updated = start * (a.T @ (data / (a @ start)))
        self.assertEqual(updated[0], 0)
        start[0] = 1e-3
        updated = start * (a.T @ (data / (a @ start)))
        self.assertGreater(updated[0], 0)

    def test_curvature_is_not_a_global_majorizer_for_all_parameters(self):
        # Reproducible counterexample to the previous universal convergence claim.
        # Diagonal curvature at one iterate need not bound the pair penalty globally.
        x, y = 100.0, 100.0
        new_x, new_y = 0.1, 140.0
        gamma = 0.0
        d, denominator = x - y, x + y
        gx, hxx = derivatives(x, y, gamma)
        gy, hyy = derivatives(y, x, gamma)
        half_quadratic = (2 * (x + y) + gamma * abs(d)) / denominator**2
        cx, cy = 2 * max(half_quadratic, hxx), 2 * max(half_quadratic, hyy)
        quadratic = (penalty(x, y, gamma) + gx * (new_x - x) + gy * (new_y - y)
                     + cx * (new_x - x) ** 2 / 2 + cy * (new_y - y) ** 2 / 2)
        self.assertGreater(penalty(new_x, new_y, gamma) - quadratic, 20)


if __name__ == "__main__":
    unittest.main()
