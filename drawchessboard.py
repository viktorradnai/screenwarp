#!/usr/bin/python

import cv2
import wand.image
import wand.color
import wand.drawing
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
    parser.add_argument('-W', '--width', type=int, help="Target screen width. This will be the width of the output image.", default=1920)
    parser.add_argument('-H', '--height', type=int, help="Target screen height. This will be the height of the output image.", default=1080)
    parser.add_argument('-c', '--cols', type=int, help="Number of squares per column", default=16)
    parser.add_argument('-r', '--rows', type=int, help="Number of squares per row", default=9)
    parser.add_argument('filename', help="Image file")
    args = parser.parse_args()

    if args.verbose: loglevel = logging.DEBUG
    elif args.quiet: loglevel = logging.ERROR
    else:            loglevel = logging.INFO

    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s %(message)s')

    return args


def main():
    args = parse_cmdline()

    screen_width = args.width
    screen_height = args.height
    square_width = screen_width / args.cols
    square_height = screen_height / args.rows

    image = wand.image.Image(width=screen_width, height=screen_height, background=wand.color.Color('#fff'))
    with wand.drawing.Drawing() as draw:
        draw.fill_color = wand.color.Color('#000')
        for r in range(args.rows):
            for c in range(args.cols):
                if not (c + r) % 2:
                    continue
                x = square_width * c
                y = square_height * r

                logger.debug("%s %s %s %s", x, y, square_width, square_height)
                draw.rectangle(x, y, width=square_width, height=square_height)

        draw.draw(image)

    image.save(filename=args.filename)
    exit(0)


# call main()
if __name__ == '__main__':
    main()
