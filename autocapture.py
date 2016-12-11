#!/usr/bin/python

import os
import sys
import time
import logging
import argparse
import cStringIO
import __builtin__

import wx
import cv2
import traceback
import numpy as np
from threading import *
from screenwarp.camera import *
from screenwarp.gui import *
from screenwarp.draw import *
from screenwarp.util import *

logger = logging.getLogger(__name__)


def parseCmdline():
    parser = argparse.ArgumentParser(description='''
        TODO: insert description.'''
    )
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose output")
    parser.add_argument('-q', '--quiet', action='store_true', help="Output errors only")

    parser.add_argument('-a', '--autorun', action='store_true', help="Do not stop after each step")
    parser.add_argument('-C', '--usecache', action='store_true', help="Use images from cache")
    parser.add_argument('-D', '--delay', type=float, help="Wait D seconds before starting capture", metavar='D')

    parser.add_argument('-W', '--width', type=int, help="Target screen width. This will be the width of the output image.", default=1920)
    parser.add_argument('-H', '--height', type=int, help="Target screen height. This will be the height of the output image.", default=1080)
    parser.add_argument('-c', '--cols', type=int, help="Number of squares per column", default=16)

    parser.add_argument('-r', '--rows', type=int, help="Number of squares per row", default=9)
    parser.add_argument('displays', metavar='N', nargs='+')

    args = parser.parse_args()

    if args.verbose: loglevel = logging.DEBUG
    elif args.quiet: loglevel = logging.ERROR
    else:            loglevel = logging.INFO

    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s %(message)s')

    return args


if __name__ == '__main__':
    __builtin__.args = parseCmdline()

    app = MainApp(0)
    app.MainLoop()
