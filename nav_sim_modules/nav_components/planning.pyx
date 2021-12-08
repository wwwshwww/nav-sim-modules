import numpy as np
cimport numpy as np
cimport cython
from libcpp.queue cimport priority_queue
from libcpp.vector cimport vector
from libcpp.pair cimport pair

from collections import deque
import heapq

ctypedef pair[double, vector[int]] PAIR

cdef np.ndarray create_neighbor_mask(int length=1):
    mesh_range = np.arange(-length, length+1)
    xx, yy, aa = np.meshgrid(mesh_range, mesh_range, mesh_range)
    return np.vstack((xx.flatten(),yy.flatten(),aa.flatten())).T

cdef double convert_180(double rad360):
    '''
    [0,360] -> [-180,180]
    '''
    return ((rad360 - np.pi) % (np.pi*2)) - np.pi

cdef np.ndarray[np.int64_t] create_circum_mask(int r):
    cdef np.ndarray[np.int64_t, ndim=2] x, y
    x, y = np.meshgrid(np.arange(-r,r+1), np.arange(-r,r+1))
    cdef np.ndarray[np.uint8_t, ndim=2] mask = np.linalg.norm([x, y], axis=0) <= r
    return np.array([x[mask], y[mask]])

# cdef np.ndarray range_slice(np.ndarray img, int x, int y, int length=1):
#     cdef int top, left, down, right
#     top = max(0, x-length)
#     left = max(0, y-length)
#     down = max(len(img), x+length+1)
#     right = max(len(img[0]), x+length+1)
#     return img[top:down, left:right]

cdef class Planner():

    cdef public int NEIGHBOR_RANGE
    cdef public np.ndarray NEIGHBOR_MASK, ANGLE_CRITERION

    cdef public np.ndarray occupancy_map, local_criterion, circum_mask
    cdef public double turnable
    cdef public int path_color, obs_color, unk_color, avoidance_size, height, width, angle_num, exp_max, end_row, end_col
    cdef public vector[int] angles,
    cdef public vector[double] angle_thresh, angle_approx

    def __init__(
        self,
        np.ndarray occupancy_map,
        double turnable=np.pi/4,
        int path_color=0,
        int obs_color=100,
        int unk_color=-1,
        int avoidance_size=1,
        int exp_max=10000
    ):

        self.occupancy_map = np.asarray(occupancy_map)
        self.turnable = turnable
        self.path_color = path_color
        self.obs_color = obs_color
        self.unk_color = unk_color
        self.avoidance_size = avoidance_size
        self.exp_max = exp_max

        self.NEIGHBOR_RANGE = 1
        self.NEIGHBOR_MASK = create_neighbor_mask(self.NEIGHBOR_RANGE)
        # shape: (27,3)
        self.ANGLE_CRITERION = np.arctan2(self.NEIGHBOR_MASK[:,1], self.NEIGHBOR_MASK[:,0])
        # shape: (27,)
        # [-135,  180,  135,
        #   -90,    0,   90,
        #   -45,    0,   45]

        self.height = self.occupancy_map.shape[0]
        self.width = self.occupancy_map.shape[1]
        self.end_row = self.height - self.avoidance_size
        self.end_col = self.width - self.avoidance_size
        # self.cost_map = np.empty_like(self.occupancy_map)
        self.angle_num = int(np.pi*2 // self.turnable)
        self.angles = list(range(self.angle_num))
        self.angle_thresh = [convert_180(turnable*p - turnable/2) for p in range(self.angle_num+1)]
        self.angle_approx = [convert_180(turnable*p) for p in range(self.angle_num)]
        self.local_criterion = np.array([self.get_angle(a) for a in self.ANGLE_CRITERION])
        self.circum_mask = create_circum_mask(self.avoidance_size)

    cdef int get_angle(self, double rad):
        cdef int i, angle
        angle = -1
        for i in range(self.angle_num):
            if self.angle_thresh[i] < self.angle_thresh[i+1]:
                if self.angle_thresh[i] <= rad < self.angle_thresh[i+1]:
                    angle = i
                    break
            else:
                if (self.angle_thresh[i] <= rad <= np.pi) or (-np.pi <= rad < self.angle_thresh[i+1]):
                    angle = i
                    break
        return angle

    cpdef int get_angle_py(self, double rad):
        return self.get_angle(rad)

    cdef double cost_move(self, int current_x, int current_y, int current_ang, int target_x, int target_y, int target_ang):
        cdef double cost = 2.0
        # if (self.occupancy_map[current_x, current_y] == self.path_color) and (self.occupancy_map[target_x, target_y] == self.unk_color):
        #     cost += 1.0
        # if self.obs_color in range_slice(self.occupancy_map, target_x, target_y, self.avoidance_size):
        #     cost += 8.0
        # if (current_x != target_x) and (current_y != target_y):
        #     cost += 0.2
        if current_ang != target_ang:
            cost += 0.1
        # cost += 0.2 * (abs(current_ang - target_ang) % (self.angle_num-1))
        return cost

    cdef double cost_h(self, int current_x, int current_y, int current_ang, int target_x, int target_y, int target_ang):
        # return <double>max(abs(current_x-target_x), abs(current_y-target_y))
        return (abs(current_x-target_x) + abs(current_y-target_y)) / 2

    cdef np.ndarray get_neighbor(self, int x, int y, int ang):
        cdef int i, angle_diff
        cdef np.ndarray[np.int32_t, ndim=2] neighbors = np.full_like(self.NEIGHBOR_MASK, [x,y,ang], dtype=np.int32)
        cdef np.ndarray[np.uint8_t, ndim=1] result = np.full([len(neighbors)], False, dtype=np.bool8)

        cdef np.ndarray[np.int64_t, ndim=2] occupancy = self.occupancy_map
        cdef np.ndarray[np.int64_t, ndim=2] mask = self.NEIGHBOR_MASK
        cdef np.ndarray[np.int64_t, ndim=1] lc = self.local_criterion
        cdef np.ndarray[np.int64_t, ndim=2] circum

        for i in range(len(neighbors)):
            neighbors[i,0] += mask[i,0]
            neighbors[i,1] += mask[i,1]
            neighbors[i,2] = (neighbors[i,2] + mask[i,2]) % self.angle_num

            angle_diff = abs(lc[i]-ang)

            ####### filtering ##########
            if ((angle_diff <= 1) or (angle_diff >= self.angle_num-1)) and \
                ((neighbors[i,0] != x) or (neighbors[i,1] != y)) and \
                ((self.avoidance_size <= neighbors[i,0] < self.end_row) and \
                (self.avoidance_size <= neighbors[i,1] < self.end_col)):

                if not (self.obs_color in occupancy[self.circum_mask[0]+neighbors[i,0], self.circum_mask[1]+neighbors[i,1]]):
                    result[i] = True
        
        return neighbors[result]

    cpdef list get_path(self, tuple start_pos, tuple goal_pos):
        cdef np.ndarray[np.int32_t, ndim=4] oya
        cdef np.ndarray[np.uint8_t, ndim=3] close
        cdef list open_list
        cdef tuple goal, start
        cdef bint found
        cdef double score, score_d

        oya = np.full([self.height, self.width, self.angle_num, 3], -1, dtype=np.int32)
        close = np.full([self.height, self.width, self.angle_num], False, dtype=np.bool8)
        open_list = []
        heapq.heapify(open_list)

        goal = (round(goal_pos[0]), round(goal_pos[1]), self.get_angle(goal_pos[2]))
        start = (round(start_pos[0]), round(start_pos[1]), self.get_angle(start_pos[2]))

        if self.obs_color in self.occupancy_map[self.circum_mask[0]+goal[0], self.circum_mask[1]+goal[1]]:
            return []

        score = self.cost_h(start[0], start[1], start[2], goal[0], goal[1], goal[2])
        found = False
        heapq.heappush(open_list, (score, start))

        cdef tuple n
        cdef int i, c
        cdef np.ndarray[np.int32_t, ndim=2] neighbors

        c = 0
        while (len(open_list) != 0) and (c != self.exp_max):

            n = heapq.heappop(open_list)
            node = n[1]
            if close[node[0], node[1], node[2]]:
                continue
            if (node[0] == goal[0]) and (node[1] == goal[1]) and (node[2] == goal[2]):
                found = True
                break
            else:
                close[node[0], node[1], node[2]] = True

            neighbors = self.get_neighbor(node[0], node[1], node[2])
            # print(<list>neighbors)
            for i in range(len(neighbors)):
                if not close[neighbors[i][0], neighbors[i][1], neighbors[i][2]]:
                    score_d = score - self.cost_h(node[0], node[1], node[2], goal[0], goal[1], goal[2]) + self.cost_move(node[0], node[1], node[2], neighbors[i][0], neighbors[i][1], neighbors[i][2]) + self.cost_h(neighbors[i][0], neighbors[i][1], neighbors[i][2], goal[0], goal[1], goal[2])
                    # print(n[1], score_d, i)
                    oya[neighbors[i][0], neighbors[i][1], neighbors[i][2]] = node
                    heapq.heappush(open_list, (score_d, tuple(neighbors[i])))
                    score = min(score, score_d)
            
            c += 1

        path = deque()
        cdef tuple now

        if found:
            # print('found!')
            now = goal
            path.appendleft(goal)
            while True:
                if ((now[0] == start[0]) and (now[1] == start[1]) and (now[2] == start[2])):
                    break
                now = tuple(oya[now])
                path.appendleft(now)

        return <list>path