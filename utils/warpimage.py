#!/usr/bin/python

import cv2
import numpy as np
import wand.image
import wand.color
import wand.drawing
import sys
import logging
import argparse
import operator
import os.path

logger = logging.getLogger(__name__)

def parse_cmdline():
    parser = argparse.ArgumentParser(description='''
        Warp an image with the provided warp data file.'''
    )
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose output")
    parser.add_argument('-q', '--quiet', action='store_true', help="Output errors only")
    parser.add_argument('-W', '--width', type=int, help="Target screen width. This will be the width of the output images.", default=1280)
    parser.add_argument('-H', '--height', type=int, help="Target screen height. This will be the height of the output images.", default=800)
    parser.add_argument('-r', '--rows', type=int, help="Number of squares per row", default=16)
    parser.add_argument('-c', '--cols', type=int, help="Number of inner corners per column", default=9)
    parser.add_argument('-a', '--align', action='store_true', help="Consider the bottom row of points as a reference line")
    parser.add_argument('-u', '--unwarp', action='store_true', help="Reverse the warp coordinates before applying (unwarps an already warped image)")
    parser.add_argument('-f', '--flip', action='store_true', help="Consider the bottom row of points as a reference line")
    parser.add_argument('warpfile', help="Data file containing warp coordinates")
    parser.add_argument('infile', help="Input image to warp")
    parser.add_argument('outfile', help="Warped image")
    args = parser.parse_args()

    if args.verbose: loglevel = logging.DEBUG
    elif args.quiet: loglevel = logging.ERROR
    else:            loglevel = logging.INFO

    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s %(message)s')

    return args


def main():
    args = parse_cmdline()

    src = cv2.imread(args.infile)
    if src is None:
        logger.error("File '%s' could not be read", args.infile)
        exit(1)

    (screen_height, screen_width) = src.shape[:2]

    with open(args.warpfile) as f:
        s, v, rows, cols = f.readline().split()[0:4]
        if s != 'screenwarp':
            raise Exception("File {0} does not start with 'screenwarp'", args.warpfile)
        rows = int(rows)
        cols = int(cols)
        logger.debug("Rows: %s, cols: %s", rows, cols)

        # Width and height of each square
        square_width = screen_width / cols
        square_height = screen_height / rows

        # Width and height of the grid area, which may be smaller than the screen
        grid_width = square_width * cols
        grid_height = square_height * rows

        base_grid = get_base_grid(rows, cols, square_width, square_height)
        loaded_grid = np.empty((rows, cols, 2), np.float32)
        corrected_grid = np.empty((rows, cols, 2), np.float32)
        grid = np.empty((rows, cols, 2), np.float32)

        for r in range(rows-1, -1, -1):
            for c in range(cols):
                l = f.readline().split()
                #logger.debug(l)
                rv = float(l[3]) * (rows)
                cv = float(l[2]) * (cols)
                logger.debug("%s -> %s, %s -> %s", r, rv, c, cv)
                rv = int(round(rv))
                cv = int(round(cv))
                if r != rv:
                    logger.error("Row %s != %s", r, rv)
                if c != cv:
                    logger.error("Col %s != %s", c, cv)

                loaded_grid[r][c][0] = float(l[0]) * screen_width
                loaded_grid[r][c][1] = float(l[1]) * screen_height


    #loaded_grid.shape = (rows-2, cols-2, 2)
    #loaded_grid = grow_grid(loaded_grid)
    corrected_grid = loaded_grid.copy()

    mins, maxes = get_bounding_box(corrected_grid)
    logger.info("Edges: %s %s - %s %s", mins[0], mins[1], maxes[0], maxes[1])
    mins, maxes = get_contained_box(corrected_grid)
    logger.info("Edges: %s %s - %s %s", mins[0], mins[1], maxes[0], maxes[1])
    edgex = mins[0] + (grid_width -  maxes[0])
    edgey = mins[1] + (grid_height -  maxes[1])
    mulx = grid_width / (grid_width - edgex)
    muly = grid_height / (grid_height - edgey)

    if args.unwarp:
        grid = corrected_grid.copy()
    else:
        for r in range(rows-1, -1, -1):
            for c in range(cols):
                grid[r][c] = extrapolate(base_grid[r][c], corrected_grid[r][c])

    #src = cv2.resize(src, (1280,800), interpolation=cv2.INTER_AREA)
    grid = cv2.resize(grid, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)
    if args.flip:
        grid = np.flip(grid, 0)

    dst = cv2.remap(src, grid, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    cv2.imwrite(args.outfile, dst)
    exit(0)


def extrapolate(curr, prev):
    delta = map(operator.sub, curr, prev)
    #logger.debug(delta)
    return map(operator.add, curr, delta)


def get_base_grid(rows, cols, square_width, square_height):
    base_grid = np.empty((rows, cols, 2))
    for r in range(rows):
        for c in range(cols):
            base_grid[r][c] = ( c * square_width, r * square_height )
            # logger.debug("ref[%s][%s] = [ %s %s ]", r, ci, (r)*square_width, (c)*square_height)

    return base_grid


def grow_grid(grid):
    # expand by one for top, bottom, left and right
    grid = np.pad(grid, ((1,1), (1,1), (0,0)), mode='constant', constant_values=0)

    (rows, cols) = grid.shape[:2]
    logger.debug(grid.shape[:2])

    # calculate the missing edge points based on the inner points
    for i in range(1, cols-1):
        grid[0][i] = extrapolate(grid[1][i], grid[2][i])
        grid[rows-1][i] = extrapolate(grid[rows-2][i], grid[rows-3][i])

    for i in range(1, rows-1):
        grid[i][0] = extrapolate(grid[i][1], grid[i][2])
        grid[i][cols-1] = extrapolate(grid[i][cols-2], grid[i][cols-3])

    grid[0][0] = extrapolate(grid[1][1], grid[2][2])
    grid[rows-1][0] = extrapolate(grid[rows-2][1], grid[rows-3][2])
    grid[0][cols-1] = extrapolate(grid[1][cols-2], grid[2][cols-3])
    grid[rows-1][cols-1] = extrapolate(grid[rows-2][cols-2], grid[rows-3][cols-3])

    return grid


def get_contained_box(grid):
    mins = [None, None]
    maxes = [None, None]

    rows = len(grid)
    cols = len(grid[0])
    for r in range(rows):
        for c in range(cols):
            curr = grid[r][c]
            if r == 0:
                if mins[1] == None or mins[1] < curr[1]:
                    mins[1] = curr[1]
            elif r == (rows-1):
                if maxes[1] == None or maxes[1] > curr[1]:
                    maxes[1] = curr[1]

            if c == 0:
                if mins[0] == None or mins[0] < curr[0]:
                    mins[0] = curr[0]
            elif c == (cols-1):
                if maxes[0] == None or maxes[0] > curr[0]:
                    maxes[0] = curr[0]

    return (mins, maxes)


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
