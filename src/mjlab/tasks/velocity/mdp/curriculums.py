from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


class RewardWeightStage(TypedDict):
  step: int
  weight: float


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
  move_up_threshold: float = 0.8,
  move_down_threshold: float = 0.4,
) -> torch.Tensor:
  """Update terrain levels based on velocity tracking quality.

  Uses the velocity tracking reward as the curriculum metric, considering both
  linear and angular velocity tracking. This ensures robots that track angular
  velocity well can progress even with zero linear velocity commands.

  Args:
    env: The environment instance.
    env_ids: IDs of environments that terminated this step.
    command_name: Name of the velocity command term.
    asset_cfg: Configuration for the robot asset.
    move_up_threshold: Tracking quality threshold to progress to harder terrain.
    move_down_threshold: Tracking quality threshold to regress to easier terrain.
  """
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute velocity tracking quality for terminating envs.
  actual_lin = asset.data.root_link_lin_vel_b[env_ids]
  actual_ang = asset.data.root_link_ang_vel_b[env_ids]

  # Linear tracking (xy only, z assumed 0).
  lin_error = torch.sum(torch.square(command[env_ids, :2] - actual_lin[:, :2]), dim=1)
  lin_error += torch.square(actual_lin[:, 2])
  lin_reward = torch.exp(-lin_error / 0.25)

  # Angular tracking (z only, xy assumed 0).
  ang_error = torch.square(command[env_ids, 2] - actual_ang[:, 2])
  ang_error += torch.sum(torch.square(actual_ang[:, :2]), dim=1)
  ang_reward = torch.exp(-ang_error / 0.5)

  # Combined tracking quality.
  tracking_quality = 0.5 * lin_reward + 0.5 * ang_reward

  # Progress based on tracking quality.
  move_up = tracking_quality > move_up_threshold
  move_down = (tracking_quality < move_down_threshold) & ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages."""
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])
