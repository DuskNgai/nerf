from pathlib import Path

from PIL import Image
import torch
from torchvision.transforms.transforms import F

def load_image(image_path: Path, bg_color: torch.Tensor) -> torch.Tensor:
    with Image.open(image_path) as img:
        image = F.to_tensor(img)
        image = image.permute(1, 2, 0) # [H, W, C]

    # Blend alpha channel to RGB
    if image.shape[-1] == 4:
        rgb, a = image.split([3, 1], dim=-1)
        image = rgb * a + bg_color * (1.0 - a)
    return image

def to_numpy(image: torch.Tensor) -> torch.Tensor:
    return (image * 255.0).detach().cpu().permute(2, 0, 1).byte().numpy()
