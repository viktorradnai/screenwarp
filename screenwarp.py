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
        TODO: insert description.'''
    )
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose output")
    parser.add_argument('-q', '--quiet', action='store_true', help="Output errors only")
    parser.add_argument('-W', '--width', type=int, help="Target screen width. This will be the width of the output images.", default=1920)
    parser.add_argument('-H', '--height', type=int, help="Target screen height. This will be the height of the output images.", default=1080)
    parser.add_argument('-r', '--rows', type=int, help="Number of squares per row", default=16)
    parser.add_argument('-c', '--cols', type=int, help="Number of inner corners per column", default=9)
    parser.add_argument('-a', '--align', action='store_true', help="Consider the bottom row of points as a reference line")
    parser.add_argument('infile', help="Input image to find the chessboard in")
    parser.add_argument('outfile', help="Data file containing warp coordinates")
    args = parser.parse_args()

    if args.verbose: loglevel = logging.DEBUG
    elif args.quiet: loglevel = logging.ERROR
    else:            loglevel = logging.INFO

    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s %(message)s')

    return args


def main():
    args = parse_cmdline()
    # Width and height of the screen
    screen_width = args.width
    screen_height = args.height

    # Width and height of each square
    square_width = screen_width / args.cols
    square_height = screen_height / args.rows

    # Width and height of the grid area, which may be smaller than the screen
    grid_width = square_width * args.cols
    grid_height = square_height * args.rows

    cols = args.cols + 1
    rows = args.rows + 1

    corrected_grid = np.empty((rows, cols, 2), np.float32)
    inverse_grid = np.empty((rows, cols, 2), np.float32)

    base_grid = get_base_grid(rows, cols, square_width, square_height)

    draw_grid("out0.png", base_grid, screen_width+20, screen_height+20, 10, 10)

    img = cv2.imread(args.infile, 0)
    if img is None:
        logger.error("File '%s' could not be read", args.infile)
        exit(1)

    logger.info("Looking for %s x %s inside corners", cols-2, rows-2)
    status, captured_grid = cv2.findChessboardCorners(img, (cols-2, rows-2), flags=cv2.cv.CV_CALIB_CB_ADAPTIVE_THRESH)
    if status == False:
        logger.error("Failed to parse checkerboard pattern in image")
        exit(2)

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(vis, (rows-2, cols-2), captured_grid, status)
    cv2.imwrite('chess.png', vis)

    # create 2D array from 1D
    captured_grid.shape = (rows-2, cols-2, 2)

    captured_grid = grow_grid(captured_grid)

    draw_grid("out1.png", captured_grid, screen_width+20, screen_height+20, 10, 10)

    if args.align:
        # The bottom edge is the line we consider "straight". Straighten image using bottom edge as ref and align to bottom of pic
        for c in range(cols):
            offset = captured_grid[rows-1][c]
            for r in range(rows):
                corrected_grid[r][c] = [ captured_grid[r][c][0], captured_grid[r][c][1] - offset[1] + screen_height ]
    else:
        corrected_grid = captured_grid.copy()

    draw_grid("out2.png", corrected_grid, screen_width+20, screen_height+20, 10, 10)

    mins, maxes = get_bounding_box(corrected_grid)
    logger.info("Edges: %s %s - %s %s", mins[0], mins[1], maxes[0], maxes[1])
    mins, maxes = get_contained_box(corrected_grid)
    logger.info("Edges: %s %s - %s %s", mins[0], mins[1], maxes[0], maxes[1])
    edgex = mins[0] + (grid_width -  maxes[0])
    edgey = mins[1] + (grid_height -  maxes[1])
    mulx = grid_width / (grid_width - edgex)
    muly = grid_height / (grid_height - edgey)

    logger.info("mulx: %s, muly = %s", mulx, muly)

    # Scale to screen size
    for r in range(rows):
        for c in range(cols):
            corrected_grid[r][c][0] = (corrected_grid[r][c][0] - mins[0]) * mulx
            corrected_grid[r][c][1] = (corrected_grid[r][c][1] - mins[1]) * muly

    draw_grid("out3.png", corrected_grid, screen_width+20, screen_height+20, 10, 10)
    for r in range(rows-1, -1, -1):
        for c in range(cols):
            inverse_grid[r][c] = extrapolate(base_grid[r][c], corrected_grid[r][c])
            #logger.debug("[%s][%s] %s -> %s -> %s", r, c, corrected_grid[r][c][0], base_grid[r][c][0], inverse_grid[r][c][0])
            #logger.debug("[%s][%s] %s -> %s -> %s", r, c, corrected_grid[r][c][1], base_grid[r][c][1], inverse_grid[r][c][1])

    draw_grid("out4.png", inverse_grid, screen_width+20, screen_height+20, 10, 10)


    with open(args.outfile, 'w+') as f:
        f.write("{0} {1}\n".format(rows, cols))
        for r in range(rows-1, -1, -1):
            for c in range(cols):
                if c == 0 or c == cols-1: intensity = 0.2
                else: intensity = 1
                f.write("{0} {1} {2} {3} {4}\n".format(c / float(cols), r / float(rows),
                    inverse_grid[r][c][0]/screen_width, inverse_grid[r][c][1]/screen_height, intensity))

    exit(0)


def extrapolate(curr, prev):
    delta = map(operator.sub, curr, prev)
    #logger.debug(delta)
    return map(operator.add, curr, delta)


def draw_grid(filename, grid, width, height, xoff, yoff):
    image = wand.image.Image(width=width, height=height, background=wand.color.Color('#fff'))

    with wand.drawing.Drawing() as draw:
        draw.fill_color = wand.color.Color('#f00')
        for r in range(len(grid)):
            # draw.fill_color = wand.color.Color('#{0:x}{0:x}{0:x}'.format(r*2))
            for c in range(len(grid[r])):
                #logger.info("r: %s, c: %s", r, c)
                x = grid[r][c][0] + xoff
                y = grid[r][c][1] + yoff
                draw_point(draw, x, y)

        draw.draw(image)
    image.save(filename=filename)


def draw_point(draw, x, y):
    draw.point(x, y)
    draw.point(x+1, y)
    draw.point(x-1, y)
    draw.point(x, y+1)
    draw.point(x, y-1)


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
