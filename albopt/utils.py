import os
from glob import glob
from typing import Iterable

import PIL
import imageio
import pygifsicle
import rpack

from albopt.models import Image


def load_images(files_pattern: str) -> Iterable[Image]:
    files = list(filter(os.path.isfile, glob(files_pattern, recursive=True)))
    if not files:
        raise ValueError(f'no files found on {files_pattern}')

    for fname in files:
        yield load_image(fname)


def load_image(fname: str) -> Image:
    img = PIL.Image.open(fname)
    return Image(
        name=os.path.basename(fname),
        filename=fname,
        img=img,
    )


def save_gif(images: Iterable[Image], outfile: str):
    images_to_gif = []
    durations = []

    for image in images:
        img = image.img.copy()
        img.thumbnail((256, 256))
        images_to_gif.append(img)
        durations.append(image.ratio * 5)
    imageio.mimsave(outfile, images_to_gif, format='GIF', duration=durations)
    pygifsicle.optimize(outfile)


def print_images_to_terminal(images: Iterable[Image]):
    for i, image in enumerate(images, start=1):
        print(i, image.name, image.ratio)
        os.system(f'tiv -h 32 -w 32 "{image.filename}"')
        print()


# def print_to_terminal(image: 'PIL.Image'):
#     fname = f'{tempfile.gettempprefix()}'
#     with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as fp:
#         image.save(fp)
#         fname = fp.name
#         fp.flush()
#     os.system(f'tiv -h 32 -w 32 "{fname}"')
#     os.remove(fname)


# FIXME: height control
# FIXME: logs
def get_collage(images: Iterable[Image]) -> 'PIL.Image':
    imgs = []
    rects = []
    max_ratio = max([image.ratio for image in images])

    for image in images:
        scale = int(512 * (image.ratio / max_ratio))
        img = image.img.copy()
        img.thumbnail((scale,) * 2)
        imgs.append(img)

        w, h = img.size
        rects.append((w, h))

    # find collage allocation
    positions = rpack.pack(rects, max_height=2048)

    collage_x = max([(x + w) for (x, y), (w, h) in zip(positions, rects)])
    collage_y = max([(y + h) for (x, y), (w, h) in zip(positions, rects)])

    collage = PIL.Image.new('RGBA', (collage_x, collage_y), (255, 0, 0, 0))

    for (x, y), img, (w, h) in zip(positions, imgs, rects):
        box = (x, y, x + w, y + h)
        collage.paste(img, box)

    return collage
