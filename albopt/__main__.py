import os
import random

import cbox

from albopt import utils, neural_encoder, min_similarity


@cbox.cmd
def main(
    files_pattern: str,
    r: float,
    output_name: str = 'out',
    min_ratio: float = 0.1,
    algo: str = 'albopt_pow2_max_c',
):
    """Selects unique images from dataset by minimizing similarity between
    images.

    :param files_pattern: file pattern (e.g. images/*.png)
    :param r: how many images or images fractions to take
    :param min_ratio: minimum value of r images to include in output
    :param output_name: the basename for output filename
    :param algo: the name of the weighting algorithm to use to compute w_i.
      one of - albopt, albopt_pow2, albopt_pow2_max_c, greedy, random
    """
    images = list(utils.load_images(files_pattern))
    images = tuple(neural_encoder.add_vectors(images))
    images = list(min_similarity.compute(algo, images, R=r))

    images = [image for image in images if image.ratio >= min_ratio]

    os.makedirs(os.path.dirname(output_name), exist_ok=True)
    random.seed(42)
    random.shuffle(images)

    print('creating gif')
    utils.save_gif(images, f'{output_name}_gif.gif')

    print('creating collage')
    collage = utils.get_collage(images)
    collage.save(f'{output_name}_collage.png')

    os.system(f'tiv -h 160 -w 160 "{output_name}_collage.png"')


if __name__ == '__main__':
    cbox.main(main)
