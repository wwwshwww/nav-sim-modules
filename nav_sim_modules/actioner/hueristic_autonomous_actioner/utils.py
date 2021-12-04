from typing import Tuple
from numpy.typing import ArrayLike

import numpy as np
from decimal import Decimal, ROUND_HALF_UP


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