from ..scener import Scener
from trimesh.path.polygons import sample
from randoor.generator import ChestSearchRoomGenerator, ChestSearchRoomConfig
import numpy as np
from shapely.geometry import Polygon

from typing import List, Tuple

from ... import SPAWN_EXTENSION, PASSABLE_COLOR, MAP_OBS_VAL, MAP_PASS_VAL, RESOLUTION, ENV_SIZE

REGEN_COUNT = 5

class ChestSearchRoomScener(Scener):

    def __init__(self, 
                spawn_extension: float=SPAWN_EXTENSION, 
                env_size: int=ENV_SIZE, 
                resolution: float=RESOLUTION, 
                passable_color: int=PASSABLE_COLOR, 
                map_obs_val: int=MAP_OBS_VAL,
                map_pass_val: int=MAP_PASS_VAL) -> None:

        self.tag_obstacle = 'obstacle'
        self.tag_key = 'key'
        self.tag_chest = 'chest'

        ## 直近の情報 ##
        self.room_config: ChestSearchRoomConfig = None
        self.env_pixel: np.ndarray = None
        self.sample_area: Polygon = None
        self.components_info: dict = {self.tag_obstacle: [], self.tag_key: [], self.tag_chest: []}
        ##############
        self.spawn_extension = spawn_extension
        self.env_size = env_size
        self.resolution = resolution
        self.passable_color = passable_color
        self.map_obs_val = map_obs_val
        self.map_pass_val = map_pass_val
        self.generator_list = []
        self.parameter_list = []        

    def _generate_room(self, *args) -> ChestSearchRoomConfig: 
        if args in self.parameter_list:
            generator = self.generator_list[self.parameter_list.index(args)]
        else:
            generator = ChestSearchRoomGenerator(*args)
            self.generator_list.append(generator)
            self.parameter_list.append(args)

        for i in range(REGEN_COUNT):
            try:
                room_config = generator.generate_new()
                break
            except:
                pass
        
        if i+1 == REGEN_COUNT:
            raise Exception('Room Generation Error')

        return room_config

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
            space_poly=self.room_config.get_freespace_poly(), 
            origin_pos=tuple(pose[:2]),
            origin_ori=pose[2],
            resolution=self.resolution, 
            map_size=self.env_size, 
            pass_color=self.map_pass_val, 
            obs_color=self.map_obs_val,
        ).astype(np.int64)
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
                        wall_threshold=0.1, 
                        chest_collision=False,
                        key_collision=False) -> None:

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

        if chest_collision:
            self.tweak_chest_collision_all(True)
        if key_collision:
            self.tweak_key_collision_all(True)

        self.setup()

    def setup(self) -> None:
        self.sample_area = self.room_config.get_freezone_poly().buffer(-self.spawn_extension)
        self.env_pixel = self.pixelize()
        self.components_info[self.tag_obstacle] = self.room_config.get_positions(self.room_config.tag_obstacle)
        self.components_info[self.tag_key] = self.room_config.get_positions(self.room_config.tag_key)
        self.components_info[self.tag_chest] = self.room_config.get_positions(self.room_config.tag_target)

    def pixelize(self) -> np.array:
        return self.room_config.get_occupancy_grid(
            space_poly=self.room_config.get_freespace_poly(),
            resolution=self.resolution, 
            map_size=self.env_size, 
            pass_color=self.map_pass_val, 
            obs_color=self.map_obs_val
        ).astype(np.int64).reshape([self.env_size, self.env_size]).T

    def tweak_key_collision(self, index: int, is_collision: bool) -> None:
        self.room_config.tweak_key_collision(index, is_collision)

    def tweak_chest_collision(self, index: int, is_collision: bool) -> None:
        self.room_config.tweak_target_collision(index, is_collision)

    def tweak_key_collision_all(self, is_collision: bool) -> None:
        self.room_config.set_config_collisions(self.room_config.tag_key, [is_collision for _ in range(self.room_config.key_count)])

    def tweak_chest_collision_all(self, is_collision: bool) -> None:
        self.room_config.set_config_collisions(self.room_config.tag_target, [is_collision for _ in range(self.room_config.target_count)])

    def get_current_env_pixel(self) -> np.ndarray:
        return self.env_pixel