import random
from dataclasses import replace
from operator import itemgetter
from typing import Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from albopt import qp_solver
from albopt.models import Image
from albopt.qp_solver import DEBUG
from pocs.matprint import pretty_mat

DTYPE = np.float64


def compute(algo, images: Sequence[Image], R: float) -> Sequence[Image]:
    if algo == 'albopt':
        return min_corr(images, R, g_pow=1, max_c=False)
    if algo == 'albopt_pow2':
        return min_corr(images, R, g_pow=2, max_c=False)
    if algo == 'albopt_pow2_max_c':
        return min_corr(images, R, g_pow=2, max_c=True)
    if algo == 'greedy':
        return greedy_sim(images, R)
    if algo == 'random':
        return random_sim(images, R)
    raise ValueError(f'invalid algorithm {algo}')


def random_sim(images: Sequence[Image], R: float) -> Sequence[Image]:
    fnames = '|'.join([image.name for image in images])
    random.seed(fnames)
    image_indices = list(range(len(images)))

    idx2ratio = {}
    while R > 0:
        idx = random.choice(image_indices)
        image_indices.remove(idx)
        ratio = min(random.random(), R)
        idx2ratio[idx] = ratio
        R -= ratio

    final_images = []
    for idx, image in enumerate(images):
        final_images.append(replace(image, ratio=idx2ratio.get(idx, 0.0)))
    return final_images


def greedy_sim(images: Sequence[Image], R: float) -> Sequence[Image]:
    X = np.stack([image.hidden_vector for image in images])
    G = cosine_similarity(X).astype(DTYPE)

    G = G.clip(min=1e-5, out=G)  # no negative similarity
    G = np.log2(G, out=G)

    idx2ratio = {}
    while R > 0:
        selected_images = list(idx2ratio)
        if not selected_images:
            img_index = G.sum(axis=1).argmin()
            ratio = 1.0
        else:
            min_values = G[:, list(selected_images)].sum(axis=1)
            min_indices = min_values.argsort()
            img_index = [x for x in min_indices if x not in idx2ratio][0]
            ratio = -G[selected_images, img_index].sum() / len(idx2ratio)

        ratio = min(ratio, R)
        idx2ratio[img_index] = ratio
        R -= ratio

    final_images = []
    for idx, image in enumerate(images):
        final_images.append(replace(image, ratio=idx2ratio.get(idx, 0.0)))
    return final_images


def min_corr(
    images: Sequence[Image], R: float, g_pow=2, diag_1=True, max_c=True
) -> Sequence[Image]:
    N = len(images)
    X = np.stack([image.hidden_vector for image in images])
    G = cosine_similarity(X).astype(DTYPE)
    G = G.clip(min=0, out=G)  # no negative similarity

    if DEBUG:
        print(pretty_mat(G[0].round(3)))

    if DEBUG:
        print(pretty_mat(G[0].round(3)))

    G **= g_pow  # try maybe 3?

    # diag == 1, we multiply it
    if diag_1:
        np.fill_diagonal(G, 1)
    else:
        np.fill_diagonal(G, 2)

    if DEBUG and len(G) < 50:
        print(pretty_mat(G.round(3)))

    if max_c:
        c = np.triu(G, 1).max(axis=1)
    else:
        c = np.zeros(N, dtype=DTYPE)

    if DEBUG:
        print(pretty_mat(c.round(2)))

    A_eq = np.asarray([np.ones(N, dtype=DTYPE)])
    b_eq = np.array([R], dtype=DTYPE)

    # foreach x_i: 0 <= x_i <= 1
    A_ge = np.concatenate(
        (np.eye(N, dtype=DTYPE), -np.eye(N, dtype=DTYPE)), axis=0
    )
    b_ge = np.concatenate(
        (np.zeros(N, dtype=DTYPE), -np.ones(N, dtype=DTYPE)), axis=0
    )
    x0 = np.ones(N, dtype=DTYPE) * R / N

    x_star = qp_solver.solve_qp(G, x0, c, A_eq, b_eq, A_ge, b_ge)
    fn_val = qp_solver.get_value(x_star, G, c)

    if DEBUG:
        print('value:', fn_val.round(5))

    sorted_vals = sorted(
        zip(np.arange(len(x_star)), x_star.round(3)),
        key=itemgetter(1),
        reverse=True,
    )
    if DEBUG:
        print('x_star:', dict(sorted_vals))
        print(pretty_mat(x_star.round(4)))
    final_images = []
    for xi, image in zip(x_star, images):
        final_images.append(replace(image, ratio=xi))
    return final_images
