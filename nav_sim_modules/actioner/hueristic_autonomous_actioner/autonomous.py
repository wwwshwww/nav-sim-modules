from typing import Tuple
import numpy as np
from nav_components.mapping import Mapper
from nav_components.planning import Planner

from ... import MAP_UNK_VAL, MAP_OBS_VAL, MAP_PASS_VAL, PASSABLE_COLOR, RESOLUTION

class HueristicNavigationStack():
    passable_color = PASSABLE_COLOR #移動可能の色　この色以外は障害物とみなす
    map_obs_val = MAP_OBS_VAL# 地図の障害物の色
    map_pass_val = MAP_PASS_VAL # 地図における移動可能
    map_unk_val = MAP_UNK_VAL #地図における未知の領域

    path_exploration_count = 10 # 許容最大試行回数
    allowable_angle = np.pi/8
    allowable_norm = 1
    avoidance_size = 1 # ロボットの体の大きさ、どこまでの障害物を避けるか
    path_planning_count = 30000 
    
    def __init__(self, env_pixel: np.ndarray, pose: Tuple, resolution: float=RESOLUTION) -> None:
        self.env_pixel = env_pixel
        self.pose = pose # continuous 
        self.resolution = resolution

        self.pix_center_x = len(self.env_pixel) // 2
        self.pix_center_y = len(self.env_pixel[0]) // 2

        self.mapper = Mapper(
            surface=self.env_pixel, 
            agent_initial_pos=self.con2pix(self.pose)[:2], 
            passable_color=self.passable_color, 
            map_obs_val=self.map_obs_val, 
            map_pass_val=self.map_pass_val, 
            map_unk_val=self.map_unk_val
        )
        self.mapper.scan()
        self.planner = Planner(
            occupancy_map = self.mapper.occupancy_map,
            turnable=self.allowable_angle,
            path_color=self.passable_color,
            obs_color=self.map_obs_val,
            unk_color=self.map_unk_val,
            avoidance_size= self.avoidance_size,
            exp_max=self.path_planning_count
        )

    def con2pix(self, continuous_pos) -> tuple:
        '''
        連続値座標系をピクセル座標系に変換する関数
        ''' 
        pos_x = round(continuous_pos[0] // self.resolution + self.pix_center_x)
        pos_y = round(continuous_pos[1] // self.resolution + self.pix_center_y)

        return (pos_x, pos_y, continuous_pos[2])

    def pix2con(self, pixel_pos) -> tuple:
        '''
        ピクセル座標系を連続値座標系に変換する関数
        ''' 
        pos_x = (pixel_pos[0] - self.pix_center_x) * self.resolution
        pos_y = (pixel_pos[1] - self.pix_center_y) * self.resolution

        return (pos_x, pos_y, pixel_pos[2])

    def goto(self, goal) -> Tuple:
        pix_goal = self.con2pix(goal)
        print(f'start: {self.con2pix(self.pose)}, goal: {pix_goal}')
        self.mapper.set_agent_pos(self.con2pix(self.pose)[:2])
        self.mapper.scan()
        count = 1
        while True:
            if count > self.path_exploration_count:
                break

            if  (np.linalg.norm(np.array(goal[:2])-self.pose[:2]) <= self.allowable_norm) and\
                ((np.abs(self.pose[2] - goal[2]) <= self.allowable_angle) or\
                (np.abs(self.pose[2] - goal[2]) >= (np.pi*2 - self.allowable_angle))):
                    break

            self.planner.occupancy_map = self.mapper.occupancy_map
            path = self.planner.get_path(
                start_pos=self.con2pix(self.pose), 
                goal_pos=pix_goal
            )
            
            path_mask = np.full_like(self.mapper.occupancy_map, False, dtype=np.bool8)
            for p in path:
                if self.mapper.occupancy_map[p[0], p[1]] == self.map_unk_val:
                    path_mask[p[0], p[1]] = True

            ops_flag = False
            for i in range(len(path)):
                self.pose = self.pix2con(path[i])
                self.mapper.set_agent_pos(path[i][:2])
                self.mapper.scan()
                if self.map_obs_val in self.mapper.occupancy_map[path_mask]:
                    ops_flag = True
                    break

            if ops_flag:
                self.pose = (self.pose[0], self.pose[1], self.planner.angle_approx[self.pose[2]])
            else:
                self.pose = (self.pose[0], self.pose[1], goal[2])
            count +=1
            
        return self.pose