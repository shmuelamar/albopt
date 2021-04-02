import numpy as np
from numpy import linalg

DTYPE = 'float64'
NDIGITS = 4
DEBUG = False


def solve_qp(G: np.ndarray, x0, c, A_eq, b_eq, A_ge, b_ge, tol=1e-10):
    """solves quadratic program with linear constraints of the form:

    min x q(x) = xGx/2 + x*c

    s.t.
        A_eq*x = b_eq
        A_ge*x >= b_ge

    :return: the vector x
    """

    N, M_eq, M_ge = _check_input(G, c, A_eq, b_eq, A_ge, b_ge, x0)
    W = np.zeros(M_ge, dtype=bool)
    k = 0

    # 1. find A(x0) and remove lineary dependent contraints
    # 1.1 W[i] is 1 if inequality constraint i is active on A(x)
    W[:] = np.isclose(np.dot(x0, A_ge.T), b_ge, rtol=tol, atol=tol)
    W_lookup = np.arange(len(W), dtype=np.int)
    active_set = A_ge[W]

    # 1.2 remove linear dependent constraints on x0
    ld_inds = _get_linear_dependent_inds(active_set, tol)
    ld_original_ids = W_lookup[ld_inds][W[ld_inds]]
    W[ld_original_ids] = 0

    # 2.0
    xk = x0
    while True:
        if DEBUG:
            print(f'iter {k} |W|={W.sum()}')
        # 2.1 solve reduced QP problem
        c_new = np.dot(G, xk) + c
        # get all equality and active inequality constraints
        active_set = np.concatenate((A_ge[W], A_eq), axis=0)
        pk, lambdas_k = solve_qp_equality(
            G,
            c_new,
            A_eq=active_set,
            b_eq=np.zeros(len(active_set)),
            fast_check=True,
        )

        # print(
        #     f'sub-problem solution: pk={pk.round(NDIGITS)} '
        #     f'λ={lambdas_k.round(NDIGITS)}'
        # )

        # 2.2 if pk != 0
        xk_is_zero = np.isclose(pk, 0.0, rtol=tol, atol=tol).all()
        # print(f'xk is{"" if xk_is_zero else " not"} zero')
        if not xk_is_zero:
            alpha_k = 1.0
            blocking_constraint = None
            # find minimum blocking constraint
            for i, is_active in enumerate(W):
                if is_active:
                    continue

                ai = A_ge[i]
                bi = b_ge[i]

                ai_dot_pk = np.dot(ai, pk)
                if ai_dot_pk < 0:
                    val = (bi - np.dot(ai, xk)) / ai_dot_pk
                    if val < alpha_k:
                        alpha_k = val
                        blocking_constraint = i

            xk = xk + alpha_k * pk
            # print(f'x{k} = {xk.round(NDIGITS)} alpha = {alpha_k}')
            if blocking_constraint is not None:
                W[blocking_constraint] = 1

        # 2.3 - pk = 0
        else:
            try:
                # only interested on inequality lambdas
                j = lambdas_k[: -len(A_eq)].argmin()
                lambda_j = lambdas_k[j]

            # no inequality lambdas - no active ge constraints
            except ValueError:
                j = None
                lambda_j = 0

            if lambda_j < 0:
                constraint_to_remove = W_lookup[W][j]
                W[constraint_to_remove] = 0
                # xk kept the same

            # found a local minimum
            else:
                return xk

        k += 1


def solve_qp_equality(G, c, A_eq, b_eq, fast_check=False):
    """solves quadratic program with linear constraints of the form:

    min x q(x) = xGx/2 + x*c

    s.t.
        A_eq*x = b_eq

    :return: the vector x
    """
    N, M = _check_input(G, c, A_eq, b_eq, fast=fast_check)

    K = np.zeros(shape=(N + M,) * 2, dtype=DTYPE)
    K[0:N, 0:N] = G
    K[N : N + M, 0:N] = A_eq
    K[0:N, N : N + M] = -A_eq.T
    # rest kept zero

    v = np.concatenate((-c, b_eq), axis=0)

    # solve the equation K*(x λ) = v
    sol = linalg.solve(K, v)
    x, lambdas = sol[:N], sol[N:]
    return x, lambdas


def _check_input(G, c, A_eq, b_eq, A_ge=None, b_ge=None, x0=None, fast=False):
    N = G.shape[0]
    M_eq = A_eq.shape[0]

    assert c.shape == (N,)
    assert A_eq.shape == (M_eq, N)
    assert b_eq.shape == (M_eq,)

    assert G.shape == (N, N)

    # avoid this usually, takes longer
    if not fast:
        assert (G == G.T).all(), 'G not symmetric'
        # assert linalg.matrix_rank(A_eq) == M_eq, 'A_eq linear dependent'

        # test matrix is positive definite
        np.linalg.cholesky(G)

    if A_ge is None and b_ge is None and x0 is None:
        return N, M_eq

    M_ge = A_ge.shape[0]
    assert x0.shape == (N,)
    assert A_ge.shape == (M_ge, N)
    assert b_ge.shape == (M_ge,)

    # check point in omega
    assert np.isclose(
        np.dot(A_eq, x0), b_eq
    ).all(), 'not all eq constraints satisfied'
    assert (np.dot(A_ge, x0) >= b_ge).all(), 'not all ge constraints satisfied'

    return N, M_eq, M_ge


def _get_linear_dependent_inds(mat, tol=1e-8):
    ld_inds = []
    rank = 0
    for i in range(len(mat)):
        new_rank = np.linalg.matrix_rank(mat[ld_inds + [i]], tol=tol)
        if rank == new_rank:
            ld_inds.append(i)
        rank = new_rank
    return ld_inds


def get_value(x, G, c):
    return np.dot(np.dot(x, G), x) / 2 + np.dot(x, c)
