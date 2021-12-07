from typing import Tuple, Sequence

import numpy as np
from decimal import Decimal, ROUND_HALF_UP

from . import PASSABLE_COLOR, RESOLUTION

def rgb2int(rgb: Tuple[int,int,int]) -> int:
    '''
    converting RGB to int: 
    '(255,255,255)' -> '0xffffff' -> '16777215'
    '''
    return (rgb[0] << (4*4)) + (rgb[1] << (4*2)) + (rgb[2])

def int2rgb(num: int) -> Tuple[int,int,int]:
    '''
    converting int to RGB: 
    '16777215' -> '0xffffff' -> '(255,255,255)'
    '''
    r, g, b = hex(num)[2:4], hex(num)[4:6], hex(num)[6:8]
    return (int(r, 16), int(g, 16), int(b, 16))

def half_up(target_float: float) -> int:
    d = Decimal(str(target_float))
    return int(d.quantize(Decimal('0'), rounding=ROUND_HALF_UP))

def convert_180(rad360):
    '''
    [0,360] -> [-180,180]
    '''
    return ((rad360 - np.pi) % (np.pi*2)) - np.pi

def convert_360(rad180):
    '''
    [-180,180] -> [0,360]
    '''
    return rad180 % (np.pi*2)

def con2pix(continuous_pos: Sequence, pix_center, resolution=RESOLUTION) -> tuple:
    '''
    Convert continuous pose to pixel pose
    ''' 
    pos_1 = round(continuous_pos[0] / resolution + pix_center[0])
    pos_2 = round(continuous_pos[1] / resolution + pix_center[1])

    return (pos_1, pos_2, *continuous_pos[2:])

def pix2con(pixel_pos: Sequence, pix_center, resolution=RESOLUTION) -> tuple:
    '''
    Convert pixel pose to continuous pose
    ''' 
    pos_1 = (pixel_pos[0] - pix_center[0]) * resolution
    pos_2 = (pixel_pos[1] - pix_center[1]) * resolution

    return (pos_1, pos_2, *pixel_pos[2:])