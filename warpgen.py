#!/usr/bin/python

import logging
import math
from math import sqrt, degrees, radians, sin, asin, cos, tan, atan
import numpy as np

BOT_CENT_HEIGHT = 0.5   #labeled 'a' in the diagram

# calculates the x of the pixel created by projecting a cylinder-screen-pixel onto a flat screen
"""
           sqrt(2) * cos(A)
D = ---------------------------------
        /     / tan(A) - 0.1177*x \\
    cos( atan(-------------------- ))
        \     \ sqrt(2) * cos(A)  //
"""
def xcoord(x):
    x = x * math.pi   # convert x to meters
    theta = radians(90 * x / math.pi)
    xx = 2 * sqrt( 2*(1 + 1/(1+2*cos(theta)*sin(theta)) - 2/(1+tan(theta)) ) )
    return xx / 4   # convert meters to [0-1]
    
def ycoord(x, y):
    # convert x and y to meters
    x = x * math.pi
    y = y * 2.4 
    theta = radians(90 * x / math.pi)
    D = 2 * sqrt(2) / (cos(theta) + sin(theta))
    yy = 0.5 * ( D*y - (D - 2)*(2.4 + BOT_CENT_HEIGHT) ) 
    return yy / 2.4 # convert meters to [0-1]

def main():
    xstep = 0.02
    ystep = 0.04

    y_range = np.arange(1, 0, -ystep)
    x_range = np.arange(0, 1, xstep)
    print "%d %d" % (len(y_range), len(x_range))
    for y in y_range:
        for x in x_range:
            print "%f %f %f %f 1" % (x, y, xcoord(x), ycoord(x, y))


if __name__=='__main__':
    main()
