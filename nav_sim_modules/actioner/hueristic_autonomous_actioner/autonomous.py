from typing import Tuple
import numpy as np
from nav_sim_modules.nav_components.mapping import Mapper
from nav_sim_modules.nav_components.planning import Planner
from ...utils import con2pix, pix2con

from ... import MAP_UNK_VAL, MAP_OBS_VAL, MAP_PASS_VAL, PASSABLE_COLOR, RESOLUTION
from PIL import Image

class HueristicNavigationStack():
    passable_color = PASSABLE_COLOR #移動可能の色　この色以外は障害物とみなす
    map_obs_val = MAP_OBS_VAL# 地図の障害物の色
    map_pass_val = MAP_PASS_VAL # 地図における移動可能
    map_unk_val = MAP_UNK_VAL #地図における未知の領域
    
    def __init__(
        self, 
        env_pixel: np.ndarray, 
        initial_pose: Tuple, 
        path_exploration_count: int,
        path_planning_count: int, 
        allowable_angle: float,
        allowable_norm: float,
        avoidance_size: int,
        move_limit: int=-1,
        resolution: float=RESOLUTION
    ) -> None:

        self.env_pixel = env_pixel
        self.pose = initial_pose # continuous 
        self.path_exploration_count = path_exploration_count
        self.allowable_angle = allowable_angle
        self.allowable_norm = allowable_norm
        self.avoidance_size = avoidance_size
        self.path_planning_count = path_planning_count
        self.move_limit = move_limit
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
            exp_max=self.path_exploration_count
        )

    def con2pix(self, continuous_pos) -> tuple:
        '''
        連続値座標系をピクセル座標系に変換する関数
        ''' 
        return con2pix(continuous_pos, [self.pix_center_x,self.pix_center_y], self.resolution)

    def pix2con(self, pixel_pos) -> tuple:
        '''
        ピクセル座標系を連続値座標系に変換する関数
        ''' 
        return pix2con(pixel_pos, [self.pix_center_x, self.pix_center_y], self.resolution)

    def goto(self, goal) -> Tuple:
        pix_goal = self.con2pix(goal)
        self.mapper.set_agent_pos(self.con2pix(self.pose)[:2])
        self.mapper.scan()
        planning_count = 1
        footprint = 0
        ops_flag = False
        while planning_count <= self.path_planning_count:
            if  (np.linalg.norm(np.array(goal[:2])-self.pose[:2]) <= self.allowable_norm) and\
                ((np.abs(self.pose[2] - goal[2]) <= self.allowable_angle/2) or\
                (np.abs(self.pose[2] - goal[2]) >= (np.pi*2 - self.allowable_angle/2))):
                    break

            path = np.array(self.planner.get_path(start_pos=self.con2pix(self.pose), goal_pos=pix_goal))
            
            if len(path) > 0:
                path_mask = create_path_mask(self.mapper.occupancy_map, path, self.avoidance_size, self.map_unk_val)

            replan_flag = False
            current = self.pose
            for i in range(len(path)):
                current = self.pix2con(path[i])
                self.mapper.set_agent_pos(tuple(path[i][:2]))
                self.mapper.scan()
                footprint += 1
                
                if ((footprint >= self.move_limit) and (self.move_limit != -1)) or (self.mapper.occupancy_map[pix_goal[0],pix_goal[1]] == self.map_obs_val):
                    ops_flag = True
                    break
                if self.map_obs_val in self.mapper.occupancy_map[path_mask]:
                    replan_flag = True
                    break

            planning_count +=1

            if ops_flag:
                self.pose = (current[0], current[1], self.planner.angle_approx[current[2]])
                break
            elif replan_flag:
                self.pose = (current[0], current[1], self.planner.angle_approx[current[2]])
            else:
                self.pose = (current[0], current[1], goal[2])
                   
        return self.pose

    def goto_visualize(self, goal, output_filename) -> Tuple:
        pixlize_start = self.con2pix((self.pose[0], self.pose[1]))
        pix_goal = self.con2pix(goal)
        pixlize_start_coler = (255, 0, 0)
        pixlize_goal_coler = (0, 0, 255)
        pixlize_path_coler = (0, 255, 0)
        pixlize_current_coler = (255, 0, 255)
        pixlize_trajectoly_coler = (255, 255, 0)
        pixlize_angle_color = (0, 255, 255)
        pixlize_angle_len = 3
        pixlize_pics = []
        pixlize_trajectoly = [pixlize_start]
        pixlize_trajectoly_mask = np.full_like(self.mapper.occupancy_map, False, dtype=np.bool8)
        pixlize_trajectoly_mask[pixlize_start[0], pixlize_start[1]] = True
        
        print(f'Navigation started: {self.con2pix(self.pose)} to {pix_goal}')
        self.mapper.set_agent_pos(self.con2pix(self.pose)[:2])
        self.mapper.scan()
        planning_count = 1
        footprint = 0
        ops_flag = False
        while planning_count <= self.path_planning_count:
            if  (np.linalg.norm(np.array(goal[:2])-self.pose[:2]) <= self.allowable_norm) and\
                ((np.abs(self.pose[2] - goal[2]) <= self.allowable_angle) or\
                (np.abs(self.pose[2] - goal[2]) >= (np.pi*2 - self.allowable_angle))):
                    break
            
            # self.planner.occupancy_map = self.mapper.occupancy_map
            path = np.array(self.planner.get_path(start_pos=self.con2pix(self.pose), goal_pos=pix_goal))
            
            if len(path) > 0:
                # path_mask[path[:,0],path[:,1]] = self.mapper.occupancy_map[path[:,0],path[:,1]] == self.map_unk_val
                path_mask = create_path_mask(self.mapper.occupancy_map, path, self.avoidance_size, self.map_unk_val)
                full_path_mask = np.full_like(self.mapper.occupancy_map, False, dtype=np.bool8)
                full_path_mask[path[:,0],path[:,1]] = True

            replan_flag = False
            current = self.pose
            for i in range(len(path)):
                current = self.pix2con(path[i])
                self.mapper.set_agent_pos(tuple(path[i][:2]))
                self.mapper.scan()
                footprint += 1

                pixlize_trajectoly.append(path)
                pixlize_trajectoly_mask[path[i][0],path[i][1]] = True

                pixlize_pic = np.empty((*self.mapper.occupancy_map.shape,3), dtype=np.uint8)
                pixlize_pic[:,:,0] = self.planner.occupancy_map
                pixlize_pic[:,:,1] = self.planner.occupancy_map
                pixlize_pic[:,:,2] = self.planner.occupancy_map
                pixlize_pic[full_path_mask] = pixlize_path_coler
                pixlize_pic[pixlize_trajectoly_mask] = pixlize_trajectoly_coler 
                pixlize_pic[pixlize_start[0], pixlize_start[1]] = pixlize_start_coler
                pixlize_pic[pix_goal[0], pix_goal[1]] = pixlize_goal_coler
                pixlize_pic[path[i][0],path[i][1]] = pixlize_current_coler
                ang = self.planner.angle_approx[path[i][2]]
                x = round(path[i][0]+np.cos(ang)*pixlize_angle_len)
                y = round(path[i][1]+np.sin(ang)*pixlize_angle_len)
                pixlize_pic[x,y] = pixlize_angle_color
                pixlize_pics.append(pixlize_pic)

                if ((footprint >= self.move_limit) and (self.move_limit != -1)) or (self.mapper.occupancy_map[pix_goal[0],pix_goal[1]] == self.map_obs_val):
                    ops_flag = True
                    break
                if self.map_obs_val in self.mapper.occupancy_map[path_mask]:
                    replan_flag = True
                    break

            planning_count +=1

            if ops_flag:
                self.pose = (current[0], current[1], self.planner.angle_approx[current[2]])
                break
            elif replan_flag:
                self.pose = (current[0], current[1], self.planner.angle_approx[current[2]])
            else:
                self.pose = (current[0], current[1], goal[2])

        if len(pixlize_pics) > 1:
            create_gif(pixlize_pics, output_filename)    

        return self.pose

from numba import njit, b1, i8, prange

@njit(b1[:,:](i8[:,:], i8[:,:], i8, i8))
def create_path_mask(occupancy_map: np.ndarray, path_xy: np.ndarray, avoidance: int, map_unk_color: int):
    row = len(occupancy_map)
    col = len(occupancy_map[0])
    output_mask = np.zeros_like(occupancy_map, dtype=np.bool8)
    for i in prange(len(path_xy)):
        top = np.max(np.array([0, path_xy[i][0]-avoidance]))
        bottom = np.min(np.array([row, path_xy[i][0]+avoidance+1]))
        left = np.max(np.array([0, path_xy[i][1]-avoidance]))
        right = np.min(np.array([col, path_xy[i][1]+avoidance+1]))
        output_mask[top:bottom, left:right] = occupancy_map[top:bottom, left:right] == map_unk_color
    return output_mask


def create_gif(frames: list, filename: str="output"):
    frs = [Image.fromarray(f, mode="RGB") for f in frames]
    frs[0].save(f'{filename}.gif', save_all=True, append_images=frs[1:], optimize=False, duration=60, loop=0)