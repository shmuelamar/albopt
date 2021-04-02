import os
import random
import pandas as pd
from dataclasses import dataclass
from glob import glob
from operator import attrgetter
from typing import List

import numpy as np

from albopt import utils, neural_encoder, min_similarity


@dataclass
class Dataset:
    classes: list
    cls2imgs: dict
    name: str

    @classmethod
    def load(cls, name, path, ext='.jpg'):
        cls2imgs = {}
        for fname in glob(f'{path}/**/*{ext}', recursive=True):
            clsid = fname.split('/')[-2]

            if clsid not in cls2imgs:
                cls2imgs[clsid] = []
            cls2imgs[clsid].append(fname)

        print(f'loaded {len(cls2imgs)} classes')
        return cls(
            cls2imgs=cls2imgs,
            classes=sorted(cls2imgs),
            name=name,
        )

    def get_random_images(self, classes_id, size):
        class_id = random.choice(classes_id)
        return class_id, random.sample(self.cls2imgs[class_id], k=size)

    def get_classes_by_size(self, size):
        classes = []
        for cls, imgs in self.cls2imgs.items():
            if len(imgs) >= size:
                classes.append(cls)
        return classes


def sampler(dataset: Dataset, alpha=0.1, sample_size=100, nb_classes=10):
    prob = np.random.dirichlet([alpha] * nb_classes)
    sizes = np.random.multinomial(sample_size - nb_classes, prob) + 1

    seen_clses = set()

    cls2imgs = {}
    for size in sizes:
        possible_classes = sorted(
            set(dataset.get_classes_by_size(size)) - seen_clses
        )
        class_id, images = dataset.get_random_images(possible_classes, size)
        cls2imgs[class_id] = images
        seen_clses.add(class_id)
    return cls2imgs


def evaluate(cls2images, ratio_algorithms: List[callable], r=10):
    img2cls = {
        img: cls for cls, images in cls2images.items() for img in images
    }
    assert len(img2cls) == 100

    images = [
        utils.load_image(img)
        for images in cls2images.values()
        for img in images
    ]
    images = tuple(neural_encoder.add_vectors(images))

    algos_mse = {}
    algos_images = {}
    for alg in ratio_algorithms:
        fn_name = alg.__name__
        images = list(alg(images, R=r))

        cls2weight = {cls: 0 for cls in cls2images}
        for img in images:
            cls2weight[img2cls[img.filename]] += img.ratio

        weights = np.asarray(list(cls2weight.values()))
        mse = ((weights - 1.0) ** 2).sum() / len(weights)

        images.sort(key=attrgetter('ratio'), reverse=True)

        algos_mse[fn_name] = mse.round(4)
        algos_images[fn_name] = images
    return algos_mse, algos_images


def main():
    def albopt(images, R):
        return min_similarity.min_corr(
            images, R, g_pow=1, diag_1=False, max_c=False
        )

    def albopt_pow2(images, R):
        return min_similarity.min_corr(
            images, R, g_pow=2, diag_1=True, max_c=False
        )

    def albopt_pow2_max_c(images, R):
        return min_similarity.min_corr(
            images, R, g_pow=2, diag_1=True, max_c=True
        )

    algos = [
        albopt,
        albopt_pow2,
        albopt_pow2_max_c,
        min_similarity.random_sim,
        min_similarity.greedy_sim,
    ]
    datasets = [
        Dataset.load('people', 'data/lfw-funneled'),
        Dataset.load('places', 'data/places'),
    ]

    results = []
    for dataset in datasets:
        for alpha in [0.1, 0.3, 0.5, 1, 5, 25, 100]:
            for seed in range(100):
                random.seed(seed)
                np.random.seed(seed)
                cls2images = sampler(dataset, alpha=alpha)

                res, images = evaluate(cls2images, algos)
                print(res)
                for algo, mse in res.items():
                    results.append(
                        {
                            'algo': algo,
                            'mse': mse,
                            'alpha': alpha,
                            'dataset': dataset.name,
                        }
                    )

                    if seed == 0:
                        output_name = f'eval/{alpha}_{seed}_{algo}'
                        os.makedirs(
                            os.path.dirname(output_name), exist_ok=True
                        )
                        algo_imgs = [
                            image
                            for image in images[algo]
                            if image.ratio >= 0.1
                        ]
                        collage = utils.get_collage(algo_imgs)
                        collage.save(f'{output_name}_collage.png')

    df = pd.DataFrame(results)
    df.to_csv('results.csv', index=False)


if __name__ == '__main__':
    main()
