from typing import Tuple, List

from ..utils import con2pix, pix2con
from .. import RESOLUTION

class Actioner():
    def __init__(self, resolution=RESOLUTION) -> None:
        self.action_space = None
        self.observation_space = None
        self.env_pixel = None
        self.resolution = RESOLUTION

        self.pix_center_x = 0
        self.pix_center_y = 0

        self.global_pose_x = 0
        self.global_pose_y = 0
        self.global_pose_yaw = 0

        self.local_pose_x = 0
        self.local_pose_y = 0
        self.local_pose_yaw = 0
        
    def initialize(self, env_pixel, global_pose: Tuple) -> None:
        self.env_pixel = env_pixel
        self.pix_center_x = len(self.env_pixel) // 2
        self.pix_center_y = len(self.env_pixel[0]) // 2
        self.global_pose_x = global_pose[0]
        self.global_pose_y = global_pose[1]
        self.global_pose_yaw = global_pose[2]

    def do_action(self, action: any) -> None:
        pass

    @property
    def pose(self):
        return None

    def con2pix(self, pose):
        return con2pix(pose, [self.pix_center_x,self.pix_center_y],self.resolution)

    def pix2con(self, pose):
        return pix2con(pose, [self.pix_center_x,self.pix_center_y],self.resolution)
