from typing import Tuple
import numpy as np
from ..actioner import Actioner
from .autonomous import HueristicNavigationStack

from ... import MAP_OBS_VAL, MAP_PASS_VAL, MAP_UNK_VAL, PASSABLE_COLOR, RESOLUTION

class HeuristicLocalAutonomousActioner(Actioner):
    def __init__(
        self, 
        path_exploration_count: int=20000,
        path_planning_count: float=10, 
        allowable_angle: float=np.pi/8,
        allowable_norm: float=0.5,
        avoidance_size: int=1,
        move_limit: int=-1,
        resolution: int=RESOLUTION,
        passable_color: int=PASSABLE_COLOR,
        map_obs_val: int=MAP_OBS_VAL,
        map_pass_val: int=MAP_PASS_VAL,
        map_unk_val: int=MAP_UNK_VAL
    ) -> None:

        super().__init__(resolution)
        self.path_exploration_count = path_exploration_count
        self.allowable_angle = allowable_angle
        self.allowable_norm = allowable_norm
        self.avoidance_size = avoidance_size
        self.path_planning_count = path_planning_count
        self.move_limit = move_limit

        self.passable_color = passable_color
        self.map_obs_val = map_obs_val
        self.map_pass_val = map_pass_val
        self.map_unk_val = map_unk_val

        self.navs: HueristicNavigationStack = None
        self.occupancy_map: np.ndarray = None
    
    def initialize(self, env_pixel, global_pose) -> None:
        '''
        env_pixel should be transformed image.
        '''
        super().initialize(env_pixel, global_pose)
        self.navs = HueristicNavigationStack(
            self.env_pixel, 
            (self.local_pose_x,self.local_pose_y,self.local_pose_yaw),
            self.path_exploration_count,
            self.path_planning_count,
            self.allowable_angle,
            self.allowable_norm,
            self.avoidance_size,
            self.move_limit,
            self.resolution,
            self.passable_color,
            self.map_obs_val,
            self.map_pass_val,
            self.map_unk_val
        )
        self.occupancy_map = self.navs.mapper.occupancy_map

    def register_env_pixel(self, env_pixel: np.array) -> None:
        self.navs.mapper.surface = env_pixel

    def do_action(self, action) -> None:
        reached_pose = self.navs.goto(action) # should be pose of the local frame
        self.occupancy_map = self.navs.mapper.occupancy_map
        self.local_pose_x = reached_pose[0]
        self.local_pose_y = reached_pose[1]
        self.local_pose_yaw = reached_pose[2]

    def do_action_visualize(self, action, filename='move') -> None:
        reached_pose = self.navs.goto_visualize(action, filename) # should be pose of the global frame
        self.occupancy_map = self.navs.mapper.occupancy_map
        self.global_pose_x = reached_pose[0]
        self.global_pose_y = reached_pose[1]
        self.global_pose_yaw = reached_pose[2]    

    @property
    def pose(self) -> Tuple[float,float,float]:
        return (self.local_pose_x, self.local_pose_y, self.local_pose_yaw)

class HeuristicAutonomousActioner(Actioner):
    def __init__(
        self, 
        path_exploration_count: int=20000,
        path_planning_count: float=10, 
        allowable_angle: float=np.pi/8,
        allowable_norm: float=0.5,
        avoidance_size: int=1,
        move_limit: int=-1,
        resolution: int=RESOLUTION,
        passable_color: int=PASSABLE_COLOR,
        map_obs_val: int=MAP_OBS_VAL,
        map_pass_val: int=MAP_PASS_VAL,
        map_unk_val: int=MAP_UNK_VAL
    ) -> None:

        super().__init__(resolution)
        self.path_exploration_count = path_exploration_count
        self.allowable_angle = allowable_angle
        self.allowable_norm = allowable_norm
        self.avoidance_size = avoidance_size
        self.path_planning_count = path_planning_count
        self.move_limit = move_limit

        self.passable_color = passable_color
        self.map_obs_val = map_obs_val
        self.map_pass_val = map_pass_val
        self.map_unk_val = map_unk_val

        self.navs: HueristicNavigationStack = None
        self.occupancy_map: np.ndarray = None
    
    def initialize(self, env_pixel, global_pose) -> None:
        '''
        env_pixel should use image that using same frame as global_pose's.
        '''
        super().initialize(env_pixel, global_pose)
        self.navs = HueristicNavigationStack(
            self.env_pixel, 
            (self.global_pose_x,self.global_pose_y,self.global_pose_yaw),
            self.path_exploration_count,
            self.path_planning_count,
            self.allowable_angle,
            self.allowable_norm,
            self.avoidance_size,
            self.move_limit,
            self.resolution,
            self.passable_color,
            self.map_obs_val,
            self.map_pass_val,
            self.map_unk_val
        )
        self.occupancy_map = self.navs.mapper.occupancy_map

    def register_env_pixel(self, env_pixel: np.array) -> None:
        self.navs.mapper.surface = env_pixel

    def do_action(self, action) -> None:
        reached_pose = self.navs.goto(action) # should be pose of the global frame
        self.occupancy_map = self.navs.mapper.occupancy_map
        self.global_pose_x = reached_pose[0]
        self.global_pose_y = reached_pose[1]
        self.global_pose_yaw = reached_pose[2]

    def do_action_visualize(self, action, filename='move') -> None:
        reached_pose = self.navs.goto_visualize(action, filename) # should be pose of the global frame
        self.occupancy_map = self.navs.mapper.occupancy_map
        self.global_pose_x = reached_pose[0]
        self.global_pose_y = reached_pose[1]
        self.global_pose_yaw = reached_pose[2]

    @property
    def pose(self) -> Tuple[float,float,float]:
        return (self.global_pose_x, self.global_pose_y, self.global_pose_yaw)
