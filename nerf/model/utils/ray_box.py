import torch

def ray_box_intersection(
    origin: torch.Tensor,
    direction: torch.Tensor,
    aabb_min: torch.Tensor,
    aabb_max: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
    """
    Ray-box intersection.

    Args:
        origin (torch.Tensor): The origin of the ray. [B, 3]
        direction (torch.Tensor): The direction of the ray. [B, 3]
        aabb_min (torch.Tensor): The minimum corner of the axis-aligned bounding box. [3]
        aabb_max (torch.Tensor): The maximum corner of the axis-aligned bounding box. [3]

    Returns:
        (torch.Tensor): The near intersection distance. [B]
        (torch.Tensor): The far intersection distance. [B]
        (torch.BoolTensor): Whether the ray intersects the box. [B]
    """
    inv_dir = 1.0 / direction
    t_min = (aabb_min - origin) * inv_dir # [B, 3]
    t_max = (aabb_max - origin) * inv_dir # [B, 3]

    near = torch.max(torch.minimum(t_min, t_max), dim=-1).values # [B]
    far  = torch.min(torch.maximum(t_min, t_max), dim=-1).values # [B]
    invalid = near >= far # [B]
    return near, far, invalid
