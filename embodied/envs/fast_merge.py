"""
Fast Merge Environment - A configurable highway merge environment with faster speeds.

This module provides a subclass of the highway-env MergeEnv that allows
configuring vehicle speeds through the environment config.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np

from highway_env import utils
from highway_env.envs.merge_env import MergeEnv
from highway_env.vehicle.controller import ControlledVehicle


class FastMergeEnv(MergeEnv):
    """
    A highway merge environment with configurable vehicle speeds.
    
    Additional config options:
        ego_speed: Initial speed of ego vehicle (default: 30.0)
        other_vehicles_speed_range: [min, max] speed for other vehicles (default: [29, 32])
        merging_vehicle_speed: Initial speed of merging vehicle (default: 20.0)
        merging_vehicle_target_speed: Target speed for merging vehicle (default: 30.0)
        speed_multiplier: Multiply all speeds by this factor (default: 1.0)
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            # Speed configuration
            "ego_speed": 30.0,
            "other_vehicles_speed_range": [29.0, 32.0],
            "merging_vehicle_speed": 20.0,
            "merging_vehicle_target_speed": 30.0,
            "speed_multiplier": 1.0,
            # Higher reward speed range for faster driving
            "reward_speed_range": [20, 30],
        })
        return cfg

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane.
        Speeds are configurable through the environment config.
        """
        road = self.road
        multiplier = self.config.get("speed_multiplier", 1.0)
        
        # Ego vehicle with configurable speed
        ego_speed = self.config.get("ego_speed", 30.0) * multiplier
        ego_vehicle = self.action_type.vehicle_class(
            road, 
            road.network.get_lane(("a", "b", 1)).position(30.0, 0.0), 
            speed=ego_speed
        )
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        
        # Other vehicles with configurable speed range
        speed_range = self.config.get("other_vehicles_speed_range", [29.0, 32.0])
        base_positions_speeds = [(90.0, 29.0), (70.0, 31.0), (5.0, 31.5)]
        
        for position, default_speed in base_positions_speeds:
            lane = road.network.get_lane(("a", "b", self.np_random.integers(2)))
            pos = lane.position(position + self.np_random.uniform(-5.0, 5.0), 0.0)
            # Scale speed based on config
            speed = self.np_random.uniform(speed_range[0], speed_range[1]) * multiplier
            road.vehicles.append(other_vehicles_type(road, pos, speed=speed))

        # Merging vehicle with configurable speed
        merging_speed = self.config.get("merging_vehicle_speed", 20.0) * multiplier
        merging_target = self.config.get("merging_vehicle_target_speed", 30.0) * multiplier
        
        merging_v = other_vehicles_type(
            road, 
            road.network.get_lane(("j", "k", 0)).position(110.0, 0.0), 
            speed=merging_speed
        )
        merging_v.target_speed = merging_target
        road.vehicles.append(merging_v)
        
        self.vehicle = ego_vehicle


def register_fast_merge_env():
    """Register the fast merge environment with gymnasium."""
    try:
        register(
            id='fast-merge-v0',
            entry_point='embodied.envs.fast_merge:FastMergeEnv',
            max_episode_steps=100,
        )
    except gym.error.Error:
        # Already registered
        pass


# Auto-register when module is imported
register_fast_merge_env()
