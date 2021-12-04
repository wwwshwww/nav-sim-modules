from hueristic_autonomous_actioner.autonomous import HueristicNavigationStack
from hueristic_autonomous_actioner.utils import rgb2int

import numpy as np
import pygame

import matplotlib.pyplot as plt

import pygame.draw
from pygame.pixelarray import PixelArray
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"


def main():
    width = 100
    height = 100
    pass_color = (0,0,0)
    pass_color_num = rgb2int(pass_color)

    # pos = (40, 10, np.pi/4)
    start = (-0.4, -2.0, np.pi/4)


    ###### 環境画像の生成 ###########
    screen = pygame.display.set_mode((width, height))
    screen_pix = PixelArray(screen)
    screen.fill(pass_color_num)

    for _ in range(2):
        pygame.draw.circle(screen, [255,255,255], [np.random.randint(width), np.random.randint(height)], np.random.randint(width//10))
    pygame.display.update()
        
    surfarr = pygame.surfarray.array2d(screen) # 環境画像
    print(np.unique(surfarr), pass_color_num)

    ##############################

    plt.imsave('map.png', surfarr, cmap='gray')

    nav =  HueristicNavigationStack(surfarr, start)
    nav.mapper.scan()
    mono = np.copy(nav.mapper.occupancy_map)
    mono[mono==-1] = 50
    plt.imsave('occupancy_map.png', mono, cmap='gray')
    
    print(nav.goto((2.0, 2.0, 0)))
    print("done")

    # pygame.draw.circle(screen, (255,255,255), pos[:2], 10)
    # pygame.display.update()
    while True:
        pass

if __name__ == '__main__':
    main()