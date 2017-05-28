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
        Visualises the points in a warp file.'''
    )
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose output")
    parser.add_argument('-q', '--quiet', action='store_true', help="Output errors only")
    parser.add_argument('-W', '--width', type=int, help="Target screen width. This will be the width of the output images.", default=1920)
    parser.add_argument('-H', '--height', type=int, help="Target screen height. This will be the height of the output images.", default=1080)
    parser.add_argument('-a', '--align', action='store_true', help="Consider the bottom row of points as a reference line")
    parser.add_argument('warpfile', help="Data file containing warp coordinates")
    parser.add_argument('outfile', help="Image showing visualised warp coordinates")
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

    with open(args.warpfile) as f:
        s, v, rows, cols = f.readline().split()[0:4]
        if s != 'screenwarp':
            raise Exception("File {0} does not start with 'screenwarp'", args.warpfile)
        rows = int(rows)
        cols = int(cols)
        logger.debug("Rows: %s, cols: %s", rows, cols)

        grid = np.empty((rows, cols, 2), np.float32)

        for r in range(rows-1, -1, -1):
            for c in range(cols):
                l = f.readline().split()
                #logger.debug(l)
                rv = float(l[0]) * (rows)
                cv = float(l[1]) * (cols)
                logger.debug("%s -> %s, %s -> %s", r, rv, c, cv)
                rv = int(round(rv))
                cv = int(round(cv))
                if r != rv:
                    logger.error("Row %s != %s", r, rv)
                if c != cv:
                    logger.error("Col %s != %s", c, cv)

                grid[r][c][0] = float(l[2]) * screen_width
                grid[r][c][1] = float(l[3]) * screen_height
                logger.debug("[%s][%s] = ( %s, %s)", r, c, grid[r][c][0], grid[r][c][1])

    draw_grid(args.outfile, grid, screen_width+20, screen_height+20, 10, 10)

    exit(0)


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


# call main()
if __name__ == '__main__':
    main()
