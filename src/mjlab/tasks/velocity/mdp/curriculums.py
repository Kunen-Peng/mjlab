from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

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
  lin_reward_name: str = "track_linear_velocity",
  ang_reward_name: str = "track_angular_velocity",
  threshold_up: float = 0.5,
  threshold_down: float = 0.15,
) -> torch.Tensor:
  """Update terrain levels based on episode-averaged raw (unweighted) rewards.

  Args:
    env: The environment instance.
    env_ids: IDs of environments that terminated this step.
    command_name: Name of the velocity command term (unused, kept for compatibility).
    lin_reward_name: Name of the linear velocity tracking reward term.
    ang_reward_name: Name of the angular velocity tracking reward term.
    threshold_up: Raw reward above which to progress (0.5 ≈ 50% tracking quality).
    threshold_down: Raw reward below which to regress (0.15 ≈ 15% tracking quality).
  """
  del command_name  # Unused, kept for config compatibility.

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  # Get episode-accumulated rewards and normalize by weight to get raw rewards.
  lin_weight = env.reward_manager.get_term_cfg(lin_reward_name).weight
  ang_weight = env.reward_manager.get_term_cfg(ang_reward_name).weight

  lin_sum = env.reward_manager._episode_sums[lin_reward_name][env_ids]
  ang_sum = env.reward_manager._episode_sums[ang_reward_name][env_ids]

  # Normalize by episode length AND weight to get average raw reward.
  avg_lin = lin_sum / env.max_episode_length_s / lin_weight
  avg_ang = ang_sum / env.max_episode_length_s / ang_weight

  # Progress based on average raw tracking reward.
  move_up = (avg_lin > threshold_up) & (avg_ang > threshold_up)
  move_down = (avg_lin < threshold_down) | (avg_ang < threshold_down)
  move_down = move_down & ~move_up

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
