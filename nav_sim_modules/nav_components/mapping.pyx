import numpy as np

cimport numpy as np
cimport cython

cdef class Mapper():

    cdef public np.ndarray surface, occupancy_map
    cdef public int agent_x, agent_y, size
    cdef public int passable_color, map_obs_val, map_pass_val, map_unk_val

    def __init__(
        self, 
        np.ndarray[np.int64_t, ndim=2] surface, 
        tuple agent_initial_pos, 
        int passable_color=0, 
        int map_obs_val=100, 
        int map_pass_val=0, 
        int map_unk_val=-1
    ):

        self.surface = surface
        self.size = len(self.surface)
        self.agent_x = round(agent_initial_pos[0])
        self.agent_y = round(agent_initial_pos[1])
        self.passable_color = passable_color
        self.map_obs_val = map_obs_val
        self.map_pass_val = map_pass_val
        self.map_unk_val = map_unk_val
        self.occupancy_map = np.full_like(self.surface, map_unk_val, dtype=np.int)

    @property
    def surfarray(self):
        return self.surface

    @property 
    def agent_pos(self):
        return tuple(self.agent_x, self.agent_y)

    cpdef void set_agent_pos(self, new_pos):
        self.agent_x = round(new_pos[0])
        self.agent_y = round(new_pos[1])

    cpdef void scan(self):
        cdef int i, l
        l = self.size-1
        for i in range(self.size):
            self.ray(self.agent_x, self.agent_y, 0, i)
            self.ray(self.agent_x, self.agent_y, l, i)
            self.ray(self.agent_x, self.agent_y, i, 0)
            self.ray(self.agent_x, self.agent_y, i, l)

    cdef void ray(self, int base_x, int base_y, int target_x, int target_y):
        cdef int x, y, dx, dy, sx, sy, err
        cdef np.ndarray[np.int64_t, ndim=2] buf = self.occupancy_map
        cdef np.ndarray[np.int64_t, ndim=2] surf = self.surface

        x = base_x
        y = base_y
        dx = abs(target_x - x)
        dy = abs(target_y - y)
        if x < target_x: sx = 1
        else: sx = -1
        if y < target_y: sy = 1
        else: sy = -1
        err = dx - dy

        while True:
            if (x >= self.size) or (x < 0) or (y >= self.size) or (y < 0): 
                break
            if surf[x,y] == self.passable_color:
                buf[x,y] = self.map_pass_val
            else:
                if buf[x,y] == self.map_obs_val:
                    break
                buf[x,y] = self.map_obs_val
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
