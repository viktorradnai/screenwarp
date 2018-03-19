#!/usr/bin/python

import sys
import logging
import argparse

logger = logging.getLogger(__name__)

def parse_cmdline():
    parser = argparse.ArgumentParser(description='''
        TODO: insert description.'''
    )
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose output")
    parser.add_argument('-q', '--quiet', action='store_true', help="Output errors only")
    parser.add_argument('infile1', help="Warp file to use for display (vertex) coordinates and intensity")
    parser.add_argument('infile2', help="Warp file to use for source (texture) coordinates")
    parser.add_argument('outfile', help="Warp file containing combined coordinates")
    args = parser.parse_args()

    if args.verbose: loglevel = logging.DEBUG
    elif args.quiet: loglevel = logging.ERROR
    else:            loglevel = logging.INFO

    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s %(message)s')

    return args


def main():

    args = parse_cmdline()
    with open(args.infile1) as xyf:
        with open(args.infile2) as uvf:
            with open(args.outfile, 'w+') as of:
                s, v, r, c, dxi, dyi, dui, dvi = xyf.readline().split()
                if s != 'screenwarp':
                    raise Exception("File {0} does not start with 'screenwarp'", args.infile1)
                rows = int(r)
                cols = int(c)
                s, v, r, c, sxi, syi, dxi, dyi = uvf.readline().split()
                if s != 'screenwarp':
                    raise Exception("File {0} does not start with 'screenwarp'", args.infile2)
                if int(r) != rows or int(c) != cols:
                    raise Exception("The two files must have the same number of rows and columns")

                of.write("{0} {1} {2} {3} {4} {5} {6} {7}\n".format(s, v, rows, cols, sxi, syi, dxi, dyi))
                for r in range(rows-1, -1, -1):
                    for c in range(cols):
                        xyl = xyf.readline().split()
                        uvl = uvf.readline().split()

                        of.write("{0} {1} {2} {3} {4}\n".format(float(xyl[0]), float(xyl[1]), float(uvl[0]), 1-float(uvl[1]), xyl[4]))

    sys.exit(0)


# call main()
if __name__ == '__main__':
    main()
