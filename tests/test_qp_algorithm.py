import numpy as np
import pytest
from scipy.optimize import OptimizeResult, minimize, Bounds
from sklearn.metrics.pairwise import cosine_similarity

from albopt import qp_solver


def test_example_solve_qp_eq():
    G = np.asarray(
        [
            [6, 2, 1],
            [2, 5, 2],
            [1, 2, 4],
        ]
    )
    c = np.asarray([-8, -3, -3])
    A_eq = np.asarray(
        [
            [1, 0, 1],
            [0, 1, 1],
        ]
    )
    b_eq = np.asarray([3, 0])
    x_star, lambdas = qp_solver.solve_qp_equality(G, c, A_eq, b_eq)
    assert np.isclose(x_star, [2, -1, 1]).all()
    assert np.isclose(lambdas, [3, -2]).all()


def test_example_solve_qp_ge():
    G = np.asarray(
        [
            [6, 2, 1],
            [2, 5, 2],
            [1, 2, 4],
        ]
    )
    c = np.asarray([-8, -3, -3])
    A_eq = np.asarray(
        [
            [1, 0, 1],
            [0, 1, 1],
        ]
    )
    b_eq = np.asarray([3, 0])

    A_ge = np.asarray(
        [
            [0, 0, 1],
            [0, 0, 2],
        ]
    )
    b_ge = np.asarray([2, 4])

    x0 = np.array([1, -2, 2])
    x_star = qp_solver.solve_qp(G, x0, c, A_eq, b_eq, A_ge, b_ge)
    assert (np.array(x_star) == [1, -2, 2]).all()
    assert qp_solver.get_value(x_star, G, c) == 3


@pytest.mark.parametrize(
    'R,N',
    [
        (10, 100),
        (90, 100),
        (1, 100),
        (15.5, 300),
    ],
)
def test_real_example(R, N, tol=1e-12):
    # prepare random data matrix
    rng = np.random.RandomState(seed=42)
    X = rng.random((N, N))

    G = cosine_similarity(X).clip(min=0)
    assert (G == G.T).all()
    assert np.isclose(G.diagonal(), 1).all()
    np.fill_diagonal(G, 2.0)  # multiply by 2

    # run reference algorithm for comparison
    def objective(x):
        return (np.dot(x, G) @ x.T) / 2

    def derivative(x):
        return np.dot(x, (G * 2))

    eq_cons = {
        'type': 'eq',
        'fun': lambda x: x.sum() == R,
        'jac': lambda x: np.ones_like(x),
    }

    x0 = np.ones(N, dtype=np.float64) * R / N
    expected_res: OptimizeResult = minimize(
        objective,
        x0,
        method='SLSQP',
        jac=derivative,
        constraints=[eq_cons],
        options={'ftol': tol, 'disp': False},
        bounds=Bounds(0, 1),
    )

    # run our algorithm
    c = np.zeros(N)
    A_eq = np.asarray(
        [
            np.ones(N),
        ]
    )
    b_eq = np.array([R])

    # foreach x_i: 0 <= x_i <= 1
    A_ge = np.concatenate((np.eye(N), -np.eye(N)), axis=0)
    b_ge = np.concatenate((np.zeros(N), -np.ones(N)), axis=0)
    x_star = qp_solver.solve_qp(G, x0, c, A_eq, b_eq, A_ge, b_ge, tol=tol)
    fn_val = qp_solver.get_value(x_star, G, c)

    assert np.isclose(expected_res.fun, fn_val, rtol=1e-4)
    assert np.isclose(expected_res.x, x_star, rtol=1e-4).all()
