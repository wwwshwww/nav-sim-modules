import trimesh
import numpy as np

class Scener():
    def __init__(self) -> None:
        ## 直近の情報 ##
        self.room_config = None
        self.env_pixel = None
        self.components_info = {}
        ##############

    def _pixelize(self) -> np.ndarray:
        pass    

    def _generate_room(self, **kwargs):
        pass
    
    def spawn(self):
        pass