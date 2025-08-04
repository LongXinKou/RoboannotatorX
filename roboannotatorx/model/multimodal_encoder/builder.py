import os
from .clip_encoder import CLIPVisionTower
from .eva_vit import EVAVisionTowerLavis

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    image_processor = getattr(vision_tower_cfg, 'image_processor', getattr(vision_tower_cfg, 'image_processor', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    
    # If local path doesn't exist, try to use HuggingFace model identifier
    if not is_absolute_path_exists:
        if "openai" in vision_tower.lower() or "clip" in vision_tower.lower():
            vision_tower = "openai/clip-vit-large-patch14"
        elif "lavis" in vision_tower.lower() or "eva" in vision_tower.lower():
            import roboannotatorx 
            base_dir = os.path.dirname(os.path.dirname(roboannotatorx.__file__))
            vision_tower = os.path.join(base_dir, vision_tower.lstrip("./\\"))
            image_processor = os.path.join(base_dir, image_processor.lstrip("./\\"))

        else:
            raise ValueError(f'Not find vision tower: {vision_tower}')
    
    if "openai" in vision_tower.lower() or "laion" in vision_tower.lower():
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "lavis" in vision_tower.lower() or "eva" in vision_tower.lower():
        return EVAVisionTowerLavis(vision_tower, image_processor, args=vision_tower_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')
    
