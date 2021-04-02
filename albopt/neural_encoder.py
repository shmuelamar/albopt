from dataclasses import replace
from itertools import islice
from typing import Iterable

import torch
import torchvision.models as models
from torchvision import transforms

from albopt.models import Image


def add_vectors(images: Iterable[Image], batch_size=32) -> Iterable[Image]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = models.resnet18(pretrained=True)
    model.to(device)

    out_layer = model._modules.get('avgpool')
    layer_output_size = 512
    model.eval()

    trans = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    images = iter(images)
    while batch_images := list(islice(images, batch_size)):
        batch_imgs = [
            trans(image.img.convert('RGB')) for image in batch_images
        ]
        X = torch.stack(batch_imgs).to(device=device)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        my_embedding = torch.zeros(
            len(batch_imgs), layer_output_size, 1, 1, device=device
        )
        h = out_layer.register_forward_hook(copy_data)
        model(X)
        h.remove()

        vecs = my_embedding.cpu().numpy()[:, :, 0, 0]
        assert len(vecs) == len(X)
        for image, vec in zip(batch_images, vecs):
            yield replace(image, hidden_vector=vec)
