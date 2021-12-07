from typing import Tuple
import numpy as np
from ..actioner import Actioner
from .autonomous import HueristicNavigationStack

from ... import RESOLUTION

class HeuristicLocalAutonomousActioner(Actioner):
    def __init__(
        self, 
        path_exploration_count: int=20000,
        path_planning_count: float=10, 
        allowable_angle: float=np.pi/8,
        allowable_norm: float=0.5,
        avoidance_size: int=1,
        resolution=RESOLUTION
    ) -> None:

        super().__init__(resolution)
        self.path_exploration_count = path_exploration_count
        self.allowable_angle = allowable_angle
        self.allowable_norm = allowable_norm
        self.avoidance_size = avoidance_size
        self.path_planning_count = path_planning_count

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
            self.avoidance_size
        )
        self.occupancy_map = self.navs.mapper.occupancy_map

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
        resolution=RESOLUTION
    ) -> None:

        super().__init__(resolution)
        self.path_exploration_count = path_exploration_count
        self.allowable_angle = allowable_angle
        self.allowable_norm = allowable_norm
        self.avoidance_size = avoidance_size
        self.path_planning_count = path_planning_count

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
            self.avoidance_size
        )
        self.occupancy_map = self.navs.mapper.occupancy_map

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
