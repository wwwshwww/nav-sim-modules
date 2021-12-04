import matplotlib.pyplot as plt
import numpy as np
import time

from cy_module import planning_cy

width, height = 100, 100
obs_color = 100
path_color = 0

world = np.full([width,height], path_color)
world[40:60, 40:60] = obs_color

start = (80,10, np.pi/6)
goal = (10,80, 0)
# start = (80,10,np.pi/2)
# goal = (80,15,np.pi)

# start = (80,10,np.pi/2)
# goal = (50,80,np.pi)

planner = planning_cy.Planner(world, np.pi/8, avoidance_size=2, exp_max=20000)
start1 = time.time()
path = planner.get_path(start, goal)
start2 = time.time()
print(f'time1: {start2-start1}')
# path = planner.get_path_slow(start, goal)
# print(f'time2: {time.time() - start2}')

for p in path:
    world[(p[0],p[1])] = 500

plt.imshow(world, cmap='gray')
plt.imsave('path.png', world, cmap='gray')
plt.show()
