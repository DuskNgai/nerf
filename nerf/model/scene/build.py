from fvcore.common.registry import Registry
from omegaconf import DictConfig

from .scene import Scene

NERF_SCENE_REGISTRY = Registry("NERF_SCENE")
NERF_SCENE_REGISTRY.__doc__ = """
Registry for scene, i.e. the 3D world.
It can be neural networks, grid, or unstructed representations.

One can sample point locations from a scene,
and obtain the corresponding attributes (e.g. color, normal, etc.).
"""

def build_scene(cfg: DictConfig, **kwargs) -> Scene:
    """
    Build the scene defined by `cfg.MODEL.SCENE.NAME`.
    It does not load checkpoints from `cfg`.
    """
    scene_name = cfg.MODEL.SCENE.NAME
    scene = NERF_SCENE_REGISTRY.get(scene_name)(cfg, **kwargs)
    return scene
