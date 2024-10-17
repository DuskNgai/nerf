from fvcore.common.registry import Registry
from omegaconf import DictConfig

from .renderer import Renderer

NERF_RENDERER_REGISTRY = Registry("NERF_RENDERER")
NERF_RENDERER_REGISTRY.__doc__ = """
Composite the attributes to obtain the final rendering result (color, normal, depth ...).
"""

def build_renderer(cfg: DictConfig, **kwargs) -> Renderer:
    """
    Build the renderer defined by `cfg.MODEL.RENDERER.NAME`.
    It does not load checkpoints from `cfg`.
    """
    renderer_name = cfg.MODEL.RENDERER.NAME
    renderer = NERF_RENDERER_REGISTRY.get(renderer_name)(cfg, **kwargs)
    return renderer
