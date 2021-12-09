from ..scener import Scener
from trimesh.path.polygons import sample
from randoor.generator import ChestSearchRoomGenerator, ChestSearchRoomConfig
import numpy as np
from shapely.geometry import Polygon

from typing import List, Tuple

from ... import SPAWN_EXTENSION, PASSABLE_COLOR, MAP_OBS_VAL, MAP_PASS_VAL, RESOLUTION, ENV_SIZE

class ChestSearchRoomScener(Scener):
    passable_color = PASSABLE_COLOR #移動可能の色　この色以外は障害物とみなす
    map_obs_val = MAP_OBS_VAL # 地図の障害物の色
    map_pass_val = MAP_PASS_VAL # 地図における移動可能

    def __init__(self, spawn_extension: float=SPAWN_EXTENSION, env_size: int=ENV_SIZE, resolution: float=RESOLUTION) -> None:
        ## 直近の情報 ##
        self.room_config: ChestSearchRoomConfig = None
        self.env_pixel: np.ndarray = None
        self.sample_area: Polygon = None
        self.freespace_area: Polygon = None
        self.components_info: dict = {'obstacle': [], 'key': [], 'chest': []}
        ##############
        self.spawn_extension = spawn_extension
        self.env_size = env_size
        self.resolution = resolution
        self.generator_list = []
        self.parameter_list = []

    def _generate_room(self, *args) -> ChestSearchRoomConfig: 
        params = tuple(v for v in args)
        if params in self.parameter_list:
            generator = self.generator_list[self.parameter_list.index(params)]
        else:
            generator = ChestSearchRoomGenerator(*params)
            self.generator_list.append(generator)
            self.parameter_list.append(params)

        return generator.generate_new()

    def _pixelize(self) -> np.ndarray:
        pix = self.room_config.get_occupancy_grid(
            space_poly=self.freespace_area, 
            resolution=self.resolution, 
            map_size=self.env_size, 
            pass_color=self.map_pass_val, 
            obs_color=self.map_obs_val
        ).astype(np.int32)
        return pix.reshape([self.env_size, self.env_size]).T

    def spawn(self) -> Tuple[float, float, float]:
        '''
        Return the initial agent pose and truth occupancy map.
        '''
        xy = []
        while len(xy) == 0:
            xy = sample(self.sample_area, 1)
        yaw = (np.random.rand()*2-1)*np.pi
        return (xy[0][0], xy[0][1], yaw)

    def spawn_with_map(self) -> Tuple[Tuple[float,float,float], np.ndarray]:
        '''
        Return the initial agent pose.
        '''
        pose = self.spawn()
        occ_map = self.room_config.get_occupancy_grid(
            space_poly=self.freespace_area, 
            origin_pos=tuple(pose[:2]),
            origin_ori=pose[2],
            resolution=self.resolution, 
            map_size=self.env_size, 
            pass_color=self.map_pass_val, 
            obs_color=self.map_obs_val,
        ).astype(np.int32)
        return pose, occ_map.reshape([self.env_size, self.env_size]).T

    def generate_scene(self, 
                        obstacle_count=10,
                        obstacle_size=0.7,
                        target_size=0.2,
                        key_size=0.2,
                        obstacle_zone_thresh=1.5,
                        distance_key_placing=0.7,
                        range_key_placing=0.3, 
                        room_length_max=9,
                        room_wall_thickness=0.05, 
                        wall_threshold=0.1) -> None:

        self.room_config = self._generate_room(
            obstacle_count, 
            obstacle_size, 
            target_size, 
            key_size, 
            obstacle_zone_thresh,
            distance_key_placing, 
            range_key_placing,
            room_length_max, 
            room_wall_thickness, 
            wall_threshold
        )
        self.sample_area = self.room_config.get_freezone_poly().buffer(-self.spawn_extension)
        self.freespace_area = self.room_config.get_freespace_poly()
        self.env_pixel = self._pixelize()
        self.components_info['obstacle'] = self.room_config.get_positions(self.room_config.tag_obstacle)
        self.components_info['key'] = self.room_config.get_positions(self.room_config.tag_key)
        self.components_info['chest'] = self.room_config.get_positions(self.room_config.tag_target)

