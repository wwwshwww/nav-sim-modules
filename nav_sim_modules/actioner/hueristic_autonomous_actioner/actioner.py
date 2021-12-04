from typing import Tuple
import numpy as np
from ..actioner import Actioner
from .autonomous import HueristicNavigationStack

from ... import RESOLUTION, MAP_UNK_VAL, MAP_PASS_VAL, MAP_OBS_VAL

class HeuristicLocalAutonomousActioner(Actioner):
    def __init__(self, resolution=RESOLUTION) -> None:
        super().__init__(resolution)
        self.navs: HueristicNavigationStack = None
        self.occupancy_map: np.ndarray = None
    
    def initialize(self, env_pixel, global_pose) -> None:
        '''
        env_pixel should be transformed image.
        '''
        super().initialize(env_pixel, global_pose)
        self.navs = HueristicNavigationStack(self.env_pixel, (self.local_pose_x,self.local_pose_y,self.local_pose_yaw))
        self.occupancy_map = self.navs.mapper.occupancy_map

    def do_action(self, action) -> None:
        reached_pose = self.navs.goto(action) # should be pose of the local frame
        self.occupancy_map = self.navs.mapper.occupancy_map
        self.local_pose_x = reached_pose[0]
        self.local_pose_y = reached_pose[1]
        self.local_pose_yaw = reached_pose[2]

    @property
    def pose(self) -> Tuple[float,float,float]:
        return (self.local_pose_x, self.local_pose_y, self.local_pose_yaw)

class HeuristicAutonomousActioner(Actioner):
    def __init__(self, resolution=RESOLUTION) -> None:
        super().__init__()
        self.resolution = resolution
        self.navs: HueristicNavigationStack = None
        self.occupancy_map: np.ndarray = None
    
    def initialize(self, env_pixel, global_pose) -> None:
        '''
        env_pixel should use image that using same frame as global_pose's.
        '''
        super().initialize(env_pixel, global_pose)
        self.navs = HueristicNavigationStack(self.env_pixel, (self.global_pose_x,self.global_pose_y,self.global_pose_yaw))
        self.occupancy_map = self.navs.mapper.occupancy_map

    def do_action(self, action) -> None:
        reached_pose = self.navs.goto(action) # should be pose of the global frame
        self.occupancy_map = self.navs.mapper.occupancy_map
        self.global_pose_x = reached_pose[0]
        self.global_pose_y = reached_pose[1]
        self.global_pose_yaw = reached_pose[2]

    @property
    def pose(self) -> Tuple[float,float,float]:
        return (self.global_pose_x, self.global_pose_y, self.global_pose_yaw)
