import sys
import numpy as np
import pygame

import matplotlib.pyplot as plt

import pygame.draw
from pygame.pixelarray import PixelArray

from cy_module.mapping_cy import Mapper
from tool import utils

def main():
    width = 512
    height = 512
    pass_color = (0,0,0)
    pass_color_num = utils.rgb2int(pass_color)
    map_obs = 0
    map_pass = 255
    map_unk = 128
    ## pos: (y,x)
    # pos = (height//2, width//2)
    pos = (400,100)

    screen = pygame.display.set_mode((width, height))
    screen_pix = PixelArray(screen)
    screen.fill(pass_color_num)

    # world = pygame.Surface((width, height))
    # world.fill(pass_color)

    for _ in range(3):
        pygame.draw.circle(screen, [np.random.randint(0,255) for _ in range(3)], [np.random.randint(width), np.random.randint(height)], np.random.randint(width//10))
    surfarr = pygame.surfarray.array2d(screen)
    print(np.unique(surfarr), pass_color_num)

    mapper = Mapper(surfarr, pos, pass_color_num, map_obs, map_pass, map_unk)
    mapper.scan()
    print(mapper.occupancy_map)
    plt.imsave('map.png', mapper.occupancy_map.T, cmap='gray')

    pygame.draw.circle(screen, (255,255,255), pos, 10)
    pygame.display.update()
    while True:
        pass

if __name__ == '__main__':
    main()