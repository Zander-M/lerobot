"""
    IMLE Policy Config
"""

from dataclasses import dataclass

@dataclass
class IMLEConfg:
    vision_feature_dim: int 
    lowdim_obs_dim:int
    obs_dim: int
    action_dim: int
    noise_dim: int
    pred_horizon: int
    obs_horizon: int
    action_horizon: int 
    dataset_path: str 
    num_diffusion_iters: int
    device: str 
    max_steps: int
    seed_start: int
    num_trails: int 
    num_epochs: int
    timestep_integer_scaler: int
    num_cameras: int
    batch_size: int
    num_flow_iters: int
