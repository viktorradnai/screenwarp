#!/usr/bin/python

import logging
import math
from math import sqrt, degrees, radians, sin, asin, cos, tan, atan
import numpy as np

A = radians(32) # Vertical sweep / 2
B = radians(90) # Horizontal sweep

# calculates the x of the pixel created by projecting a cylinder-screen-pixel onto a flat screen
"""
           sqrt(2) * cos(A)
D = ---------------------------------
        /     / tan(A) - 0.1177*x \\
    cos( atan(-------------------- ))
        \     \ sqrt(2) * cos(A)  //
"""
def xcoord(x):
    D = sqrt(2) * cos(A) / cos( atan ( tan(A) - 0.1177*x /(sqrt(2) * cos(A)) ) )
    t = tan(B * x)
    return D * t / (1 + t) / (sqrt(2)*cos(A))

def ycoord(x, y):
    theta = atan(tan(2 * A)*(1 - y)) - A
    phi = B * x
    F = 2 / (cos(phi) + sin(phi))
    h = sqrt(2) * sin(A)
    D = F * cos(asin(h / F))
    return (h - D * tan(theta)) / (2 * h)


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
