#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from math import sqrt, degrees, radians, sin, asin, cos, tan, atan
import numpy as np

import sys
import logging
import argparse

logger = logging.getLogger(__name__)

# units in mm
proj_origin = [ -250, 0, 0 ]
viewpoint_origin = [ 0, 0, 800 ]
cylscreen_origin = [ 0, 0, 0 ]
cylscren_radius = 2000

class FlatScreen:
    # units in mm
    ctr = [ 2000, 0, 1200.375 + 438.75 ]
    size = [ 4302, 2400.75 ]
    pixels = [ 1280, 768 ]

    def __init__(self):
        self.pixel_size = [
            float(self.size[0]) / self.pixels[0],
            float(self.size[1]) / self.pixels[1]
        ]
        logger.debug("Pixel size: %s x %s", *self.pixel_size)
        self.top    = self.ctr[2] + self.size[1]/2
        self.bottom = self.ctr[2] - self.size[1]/2
        self.left   = self.ctr[1] - self.size[0]/2
        self.right  = self.ctr[1] + self.size[0]/2


def parse_cmdline():
    parser = argparse.ArgumentParser(description='''
        TODO: insert description.'''
    )
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose output")
    parser.add_argument('-q', '--quiet', action='store_true', help="Output errors only")
    parser.add_argument('-r', '--rows', type=int, help="Number of squares per row", default=24)
    parser.add_argument('-c', '--cols', type=int, help="Number of inner corners per column", default=53)
    parser.add_argument('-x', '--viewpoint-distance', type=int, help="Distance of view point from screen centre", default=0)
    parser.add_argument('-z', '--viewpoint-height', type=int, help="Height of view point", default=1000)
    parser.add_argument('outfile', help="Warped image")
    args = parser.parse_args()

    if args.verbose: loglevel = logging.DEBUG
    elif args.quiet: loglevel = logging.ERROR
    else:            loglevel = logging.INFO

    #logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s %(message)s')
    logging.basicConfig(level=loglevel, format='%(levelname)s %(message)s')

    return args


def calculate_point(fs, c, r):
    logger.debug("X%%: %s, Y%%: %s", c, r)
    # Step 1: find coordinates for pixel
    # This is where we project, we then try to find what part of the picture to project to this location
    pixel_loc_flat = [
        fs.ctr[0],
        fs.left + fs.size[0] * c,
        fs.bottom + fs.size[1] * r
    ]
    logger.debug("Pixel coordinates on ideal flat screen: (%s %s %s)", *pixel_loc_flat)
    '''
    # Step 2: draw line from pixel location to projector
    #
    # Equation for line, vector form: (x0,y0,z0) + t * (x1,y1,z1)
    #
    # (proj_origin[0], proj_origin[1], proj_origin[2]) + t * (pixel_loc_flat[0], pixel_loc_flat[1], pixel_loc_flat[2])
    #
    # (proj_origin[0] + t * pixel_loc_flat[0], proj_origin[1] + t * pixel_loc_flat[1], proj_origin[2] + t * pixel_loc_flat[2])
    #

    # Step 3: add line equation into cylinder equation
    #
    # Equation for infinitely tall cylinder (we're ignoring the z axis for now):
    # r**2 = x**2 + y**2
    #
    # cylscren_radius**2 = (proj_origin[0] + t * pixel_loc_flat[0])**2 + (proj_origin[1] + t * pixel_loc_flat[1])**2
    #
    # cylscren_radius**2 = (proj_origin[0]**2 + 2 * proj_origin[0] * t * pixel_loc_flat[0] + t**2 * pixel_loc_flat[0]**2)
    #                    + (proj_origin[1]**2 + 2 * proj_origin[1] * t * pixel_loc_flat[1] + t**2 * pixel_loc_flat[1]**2)
    #
    # 0 = t**2 * (pixel_loc_flat[0]**2 + pixel_loc_flat[1]**2)
    #     + t * 2 * ( proj_origin[0] * pixel_loc_flat[0] + proj_origin[1] * pixel_loc_flat[1])
    #     + proj_origin[0]**2 + proj_origin[1]**2 - cylscren_radius**2
    #
    '''
    a = pixel_loc_flat[0]**2 + pixel_loc_flat[1]**2
    b = 2 * ( proj_origin[0] * pixel_loc_flat[0] + proj_origin[1] * pixel_loc_flat[1])
    c = proj_origin[0]**2 + proj_origin[1]**2 - cylscren_radius**2

    # Step 4: solve quadratic equation to determine the coordinates where the beam intersects the cylinder

    discriminant = b * b - 4 * a * c
    # logger.debug("a: %s, b: %s, c: %s, disc: %s", a, b, c, discriminant)

    if discriminant < 0:
        raise Exception("Beam does not intersect cylinder (discriminant is negative)")
    discriminant = sqrt(discriminant);
    # find solutions
    s1 = (-b + discriminant) / (2 * a);
    s2 = (-b - discriminant) / (2 * a);

    if s1 < 0 and s2 < 0:
        # No intersection
        raise Exception("Beam does not intersect cylinder (results are negative)")
    elif s1 > 0 and s2 > 0:
        # Two intersections
        if s1 < s2: t = s1
        else: t = s2
    else:
        # One intersection
        if s1 > 0: t = s1
        else: t = s2

    #  Step 5: use the value of t to get pixel coordinates
    pixel_loc_cyl = [
        proj_origin[0] + t * pixel_loc_flat[0],
        proj_origin[1] + t * pixel_loc_flat[1],
        proj_origin[2] + t * pixel_loc_flat[2],
    ]
    # Step 6: verify that the point is located on the cylinder
    # This should be 0 but we need to account for floating point calculation inaccuracy
    if abs(pixel_loc_cyl[0]**2 + pixel_loc_cyl[1]**2 - cylscren_radius**2) > 1e-6:
        logger.debug("X: %s Y: %s Z: %s Check: %s",
            pixel_loc_cyl[0], pixel_loc_cyl[1], pixel_loc_cyl[2], (x**2 + y**2 - cylscren_radius**2))
        raise Exception("Point not on cylinder surface")


    logger.debug("Pixel coordinates on cylindrical screen: (%s %s %s)", *pixel_loc_cyl)

    # Step 7: draw a line from the viewpoint through the point on the cylinder and
    # find where it intersects the ideal screen. This is the apparent pixel location on the screen.
    #
    # (x0,y0,z0) + t * ((x1,y1,z1) - (x0,y0,z0))
    #
    # we know that the x coordinate is fs.ctr[0] as the screen is perpendicular to the X axis
    # and we can use that to calculate t
    #
    # fs.ctr[0] = viewpoint_origin[0] + t * (pixel_loc_cyl[0] - viewpoint_origin[0])
    # fs.ctr[0] - viewpoint_origin[0] = t * (pixel_loc_cyl[0] - viewpoint_origin[0])
    # t = (fs.ctr[0] - viewpoint_origin[0]) / (pixel_loc_cyl[0] - viewpoint_origin[0])

    t = (fs.ctr[0] - viewpoint_origin[0]) / (pixel_loc_cyl[0] - viewpoint_origin[0])
    logger.debug("*** T: %s ***", t)

    pixel_val_flatscreen = [
        viewpoint_origin[0] + t * (pixel_loc_cyl[0] - viewpoint_origin[0]),
        viewpoint_origin[1] + t * (pixel_loc_cyl[1] - viewpoint_origin[1]),
        viewpoint_origin[2] + t * (pixel_loc_cyl[2] - viewpoint_origin[2])
    ]

    logger.debug("Source coordinates on flat screen: (%s %s %s)", *pixel_val_flatscreen)
    return (pixel_val_flatscreen[1], pixel_val_flatscreen[2])


def main():

    args = parse_cmdline()
    rows = args.rows
    cols = args.cols
    viewpoint_origin[0] = args.viewpoint_distance
    viewpoint_origin[2] = args.viewpoint_height

    fs = FlatScreen()
    grid = np.empty((rows, cols, 2), np.float32)

    for r in range(rows-1, -1, -1):
        for c in range(cols):
            cp = c * 1/float(cols-1)
            rp = r * 1/float(rows-1)
            try:
                (x_, y_) = calculate_point(fs, cp, rp)
            except Exception, e:
                logger.warn(e)
                # number of lines must match rows * cols so output 0 instead
                (x_, y_) = (0, 0)
                raise e
            grid[r][c] = (x_, y_)

    mins, maxes = get_bounding_box(grid)
    grid_width = maxes[0] - mins[0]
    grid_height = maxes[1] - mins[1]
    with open(args.outfile, 'w+') as f:
        f.write("screenwarp 1 {0} {1} 0 0 0 0\n".format(rows, cols))
        for r in range(rows-1, -1, -1):
            for c in range(cols):
                cp = c * 1/float(cols-1)
                rp = r * 1/float(rows-1)
                x_ = (grid[r][c][0]- mins[0]) / grid_width
                y_ = (grid[r][c][1]- mins[1]) / grid_height
                f.write("{0} {1} {2} {3} {4}\n".format(cp, rp, x_, y_, 1.0))

    sys.exit(0)


def get_bounding_box(grid):
    mins = [None, None]
    maxes = [None, None]

    for r in range(len(grid)):
        for c in range(len(grid[r])):
            curr = grid[r][c]
            if mins[0] == None or mins[0] > curr[0]:
                mins[0] = curr[0]
            if mins[1] == None or mins[1] > curr[1]:
                mins[1] = curr[1]
            if maxes[0] == None or maxes[0] < curr[0]:
                maxes[0] = curr[0]
            if maxes[1] == None or maxes[1] < curr[1]:
                maxes[1] = curr[1]

    return(mins, maxes)


# call main()
if __name__ == '__main__':
    main()

