#!/usr/bin/python

import cv2
import sys
import logging
import argparse
import os.path

logger = logging.getLogger(__name__)

def parse_cmdline():
    parser = argparse.ArgumentParser(description='''
        TODO: insert description.'''
    )
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose output")
    parser.add_argument('-q', '--quiet', action='store_true', help="Output errors only")
    parser.add_argument('-r', '--rows', type=int, help="Number of squares per row", default=16)
    parser.add_argument('-c', '--cols', type=int, help="Number of inner corners per column", default=9)
    parser.add_argument('infile', help="Input image to find the chessboard in")
    parser.add_argument('outfile', help="Output image marked with the chessboard points and their order")
    args = parser.parse_args()

    if args.verbose: loglevel = logging.DEBUG
    elif args.quiet: loglevel = logging.ERROR
    else:            loglevel = logging.INFO

    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s %(message)s')

    return args



def main():
    args = parse_cmdline()

    cols = args.cols + 1
    rows = args.rows + 1

    img = cv2.imread(args.infile, 0)
    if img is None:
        logger.error("File '%s' could not be read", args.infile)
        exit(1)

    logger.info("Looking for %s x %s inside corners", cols-2, rows-2)
    status, captured_grid = cv2.findChessboardCorners(img, (cols-2, rows-2))
    if status == False:
        logger.error("Failed to parse checkerboard pattern in image")
        exit(2)

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(vis, (rows-2, cols-2), captured_grid, status)
    cv2.imwrite(args.outfile, vis)


# call main()
if __name__ == '__main__':
    main()
