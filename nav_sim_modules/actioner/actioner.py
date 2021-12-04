from typing import Tuple, List
import gym.spaces

class Actioner():
    def __init__(self) -> None:
        self.action_space = None
        self.observation_space = None
        self.env_pixel = None

        self.global_pose_x = 0
        self.global_pose_y = 0
        self.global_pose_yaw = 0

        self.local_pose_x = 0
        self.local_pose_y = 0
        self.local_pose_yaw = 0
        
    def initialize(self, env_pixel, global_pose: Tuple) -> None:
        self.env_pixel = env_pixel
        self.global_pose_x = global_pose[0]
        self.global_pose_y = global_pose[1]
        self.global_pose_yaw = global_pose[2]

    def do_action(self, action: any) -> None:
        pass

    @property
    def pose(self):
        return None

