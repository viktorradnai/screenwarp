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

IMAGE_CAPTURE_URL = 'http://192.168.0.237:8080'
#IMAGE_CAPTURE_URL = 'http://192.168.43.194:8080'


class ImageCaptureError(Exception):
    pass

class NotFoundError(Exception):
    pass


class WorkerThread(Thread):
    def __init__(self, gui, views, screens, cond):
        Thread.__init__(self)
        self.gui = gui
        self.views = views
        self.screens = screens
        self._want_abort = 0
        self.args = args # from global
        self.cond = cond
        self.defaultcamera = AndroidIPCamera(IMAGE_CAPTURE_URL)

        seq = 0
        while os.path.isdir('out' + str(seq)):
            seq += 1
        self.outdir = 'out' + str(seq)

        self.start()


    def run(self):
        try:
            # Take a photo of each screen from each view, then process them
            for vname, view in iter(sorted(self.views.iteritems())):
                # Identify each screen
                for sname in self.screens:
                    w = self.screens[sname]
                    screenid = draw_text(sname, w['width'], w['height'])
                    self.sendCommand('displayImage', window=sname, image=screenid)
                self.wait(force_wait=True)

                # Configure camera
                if self.args.usecache:
                    self.camera = FileCacheCamera()
                else:
                    if 'camera' in view:
                        camera = view['camera']['type']
                        cameraparams = view['camera']['params']
                        logger.info("Using camera %s for view %s", camera, vname)
                        logger.debug(cameraparams)
                        self.camera = getattr(sys.modules[__name__], camera)(**cameraparams)
                    else:
                        sef.camera = self.defaultcamera

                self.acquireImage(vname, 1280, 960, focus=True)
                # Black out all screens
                for sname in self.screens:
                    w = self.screens[sname]
                    self.sendCommand('displayColor', window=sname, color='#000')


                for sname in self.screens:
                    screen = self.screens[sname]

                    if sname not in self.args.displays:
                        logger.info("Skipping display %s", sname)
                        continue

                    if 'completeScreens' in view and sname in view['completeScreens']:
                        primaryView = True
                    elif 'partialScreens' in view and sname in view['partialScreens']:
                        secondaryView = True
                    else:
                        logger.info("Display %s not visible in %s", sname, vname)
                        continue

                    # These are used for naming saved images, so there are less parameters to pass around
                    self.currView = vname
                    self.currScreen = sname

                    logger.info("Processing view %s, display %s", vname, sname)
                    self.wait()

                    self.calibrate(screen)


                    logger.debug("Finished screen %s", sname)
                    self.sendCommand('displayColor', window=sname, color='#000')

                    #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
                    #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
                    #img3 = cv2.fastNlMeansDenoising(img3,None,40,17,31)


        except Exception as e:
            logger.error("Caught exception, exiting")
            logger.error(traceback.format_exc())
        finally:
            self.sendCommand('exit')
            exit()


    def calibrate(self, screen):
        cols = 40
        rows = 24

        mask, coords = self.doMaskAndCalibration(vname, sname, screen['width'], screen['height'], cols, rows)
        if True or coords is None:
            logger.warn("Failed to find coordinates for screen %s in view %s, trying to calibrate parts of the screen", sname, vname)
            self.doPartialCalibration(vname, sname, screen['width'], screen['height'], mask)

        if False: # this is an attempt to detect corners, it is currently much less reliable than findChessboard
            points = self.findCorners(img1m)
            img = draw_points(points, img1m.shape[1], img1m.shape[0], 0, 0)
            self.putImage(img, 'img1-points')


    def doMaskAndCalibration(self, vname, sname, width, height, cols, rows, mask=None, x=0, y=0, square_width=None, square_height=None):
        # There is one less inside corner in each row and column than the number of squares we display
        cnrx = cols - 1
        cnry = rows - 1

        logger.debug("Display and capture chessboard")
        # Put up chessboard image, take photo. Invert chessboard and repeat.
        img1, img2 = self.acquireCheckboards(vname, sname, width, height, cols, rows, x, y, square_width, square_height)

        if mask is None:
            logger.info("Creating mask")
            # Subtract positive and negative chessboard to obtain mask
            mask = self.getMask(img1, img2)

            # Warn if mask reaches edge -- full grid is probably not shown
            self.isMaskOnEdge(mask)

        logger.debug("Applying mask")
        # get masked, thresholded chessboards
        img1m = self.applyMask(img1, mask)
        self.putImage(img1m, '{0}x{1}-diff1-masked'.format(cols, rows))
        img2m = self.applyMask(img2, mask)
        self.putImage(img2m, '{0}x{1}-diff2-masked'.format(cols, rows))

        logger.debug("Find chessboard")
        # Try to find chessboard in either image. Normally both should succeed if the whole chessboard is displayed.
        try:
            coords = self.findChessboard(img1m, '{0}x{1}-diff1-cb'.format(cols, rows), cnrx, cnry)
        except NotFoundError:
            # TODO: try harder to find what's left of the chessboard
            pass

        try:
            coords = self.findChessboard(img2m, '{0}x{1}-diff2-cb'.format(cols, rows), cnrx, cnry)
        except NotFoundError:
            return mask, None

        return mask, coords


    def doPartialCalibration(self, vname, sname, width, height, mask):
        x, edges = self.isMaskOnEdge(mask)

        if 'left' in edges and 'right' in edges:
            w = width / 2
            x = width / 4
        elif 'left' in edges:
            w = width / 2
            x = width / 2
        elif 'right' in edges:
            w = width / 2
            x = 0
        else:
            w = width
            x = 0

        if 'top' in edges and 'bottom' in edges:
            h = height / 2
            y = height / 4
        elif 'top' in edges:
            h = height / 2
            y = height / 2
        elif 'bottom' in edges:
            h = height / 2
            y = 0
        else:
            h = height
            y = 0

        result = self.calibrateVisibleArea(vname, sname, width, height, mask, w, h, x, y, w/2)
        x__, y__ = self.gui.GetScreenPosition(sname)
        x_, y_, w_, h_ = result
        logger.info("Results: w: %s, x: %s, actual: %s, diff: %s", w_, x_, abs(x__), x_ + x__)


    def calibrateVisibleArea(self, vname, sname, width, height, mask, w, h, x, y, step):
        squaresize = 0
        squarelimit = 8
        #commonfactors = sorted(factors(w).intersection(factors(h)), reverse=True)
        commonfactors = [ 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 ]
        commonfactors.reverse()

        logger.debug("step %s, w: %s, h: %s", step, w, h)
        for i in commonfactors:
            cols = w/i
            rows = h/i
            if cols < 4 or rows < 4:
                logger.debug("Not enough rows or columns (%s x %s) for square size %s, skipping", cols, rows, i)
                continue
            if i > step:
                logger.debug("Square size %s bigger than step %s, reducing", i, step)
                continue
            elif i < squarelimit:
                logger.debug("Square size %s below limit %s, exiting", i, squarelimit)
                return False
            squaresize = i
            break
        if squaresize == 0:
            logger.warn("Could not find adequate square size, exiting")
            return False

        logger.info("Creating %s x %s checkerboard of %s x %s squares on screen %s with offset (%s, %s) and size (%s, %s)",
            cols, rows, i, i, sname, x, y, w, h)
        mask, coords = self.doMaskAndCalibration(vname, sname, width, height, cols, rows, mask, x=x, y=y, square_width=i, square_height=i)
        res = (x, y, w, h)
        logger.debug(res)
        if coords is not None:
            logger.info("coords found with width %s, pos %s", w, x)

            # try 2 times the width
            neww = w + step
            if step > width/8 and neww <= width:
                newx = width - neww
                res2 = self.calibrateVisibleArea(vname, sname, width, height, mask, neww, h, newx, y, step)
                if res2:
                    return res2

            # try 1.5 times the width
            step /= 2
            neww = w + step
            newx = width - neww
            res2 = self.calibrateVisibleArea(vname, sname, width, height, mask, neww, h, newx, y, step)
            if res2:
                return res2
            else:
                return res
        else:
            logger.info("Coords not found with width %s, pos %s", w, x)
            # try 0.5 times the width
            neww = w - step
            if neww <= 0:
                step = step / 2
                neww = w - step
            newx = width - neww
            res2 = self.calibrateVisibleArea(vname, sname, width, height, mask, neww, h, newx, y, step)
            return res2

            # if new mask meets the edge:
                # grid off-edge?
                # reduce from centre
            # else:
                # obstruction on screen?
                # reduce from edge

            # Try stripes, count number of polygons and check if they hit the edge



    def doFineCalibration(self, vname, sname, width, height, mask):
        commonfactors = sorted(factors(width).intersection(factors(height)), reverse=True)
        limit = 8

        for i in commonfactors:
            if 'left' in edges and 'right' in edges:
                w = width / 2
                x = width / 4
            elif 'left' in edges:
                w = width / 2
                x = width / 2
            elif 'right' in edges:
                w = width / 2
                x = 0
            else:
                w = width
                x = 0

            if 'top' in edges and 'bottom' in edges:
                h = height / 2
                y = height / 4
            elif 'top' in edges:
                h = height / 2
                y = height / 2
            elif 'bottom' in edges:
                h = height / 2
                y = 0
            else:
                h = height
                y = 0

            cols = w/i
            rows = h/i
            if cols < 4 or rows < 4:
                logger.debug("Not enough rows or columns (%s x %s), skipping", cols, rows)
                continue
            if w / cols < limit or h / rows < limit:
                logger.debug("Row or column size limit reached, skipping")
                continue

            while True:
                logger.info("Creating %s x %s checkerboard of %s x %s squares on screen %s with offset (%s, %s) and size (%s, %s)",
                    cols, rows, i, i, sname, x, y, w, h)
                mask, coords = self.doMaskAndCalibration(vname, sname, width, height, cols, rows, mask, x=x, y=y, square_width=i, square_height=i)
                if coords is not None:
                    # try larger
                    logger.debug("coords found")
                    pass
                else:
                    logger.debug("coords not found")
                    # if new mask meets the edge:
                        # grid off-edge?
                        # reduce from centre
                    # else:
                        # obstruction on screen?
                        # reduce from edge

                    # Try stripes, count number of polygons and check if they hit the edge
                    pass
                break





    def acquireCheckboards(self, view, screen, width, height, cols, rows, x=0, y=0, square_width=None, square_height=None):
        # Display chessboard image then take a photo
        cb = self.drawChessboard(width, height, cols, rows, '#fff', '#000', x, y, square_width, square_height)
        self.putImage(cb, 'cb1')

        imgname = "view{0}-screen{1}-cb{2}x{3}-img{4}".format(view, screen, rows, cols, 1)
        if rows > 50 or cols > 50:
            img1 = self.acquireImage(imgname)
        else:
            img1 = self.acquireImage(imgname, width, height)
        self.putImage(img1, 'img1')

        # Display inverted chessboard image, take another photo
        cb = self.drawChessboard(width, height, cols, rows, '#000', '#fff', x, y, square_width, square_height)
        self.putImage(cb, 'cb2')

        imgname = "view{0}-screen{1}-cb{2}x{3}-img{4}".format(view, screen, rows, cols, 2)
        if rows > 50 or cols > 50:
            img2 = self.acquireImage(imgname)
        else:
            img2 = self.acquireImage(imgname, width, height)
        self.putImage(img2, 'img2')

        return img1, img2


    def applyMask(self, img, mask):
        # Mask out everything other than the chessboard area
        imgbw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, imgbw = cv2.threshold(imgbw, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        logger.debug(imgbw)
        logger.debug(mask)
        return cv2.add(imgbw, mask)


    def getMask(self, img1, img2):
        # Subtract two images showing inverted pattern. This blacks out non-display areas that only have diffuse light.
        imgdiff1 = cv2.subtract(img1, img2)
        self.putImage(imgdiff1, 'imgdiff1')
        imgdiff2 = cv2.subtract(img2, img1)
        self.putImage(imgdiff2, 'imgdiff2')

        # Add two diff images together to fill visible area, then threshold it to create mask
        imgdiff = cv2.add(imgdiff1, imgdiff2)
        self.putImage(imgdiff, 'imgdiff')
        img = cv2.cvtColor(imgdiff, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(img, 13)
        ret, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        self.putImage(mask, 'mask')

        # Detect closed contours in the image, then pick the second largest which should be the projector's main visible area
        # The largest contour is a border around the entire picture
        border = 10
        maskb = cv2.copyMakeBorder(mask, top=border, bottom=border, left=border, right=border, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255] )

        (cnts, x) = cv2.findContours(maskb, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        logger.info("Found %s contours in mask image", len(cnts))
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[1:2]
        #cv2.drawContours(mask2, cnts, -1, (255,0,0), 2)

        mask2 = np.full(maskb.shape, 255, np.uint8)
        cv2.fillPoly(mask2, pts=cnts, color=(0,0,0))
        mask2 = mask2[border:-border, border:-border]
        self.putImage(mask2, 'mask2')

        return mask2


    def isMaskOnEdge(self, mask):
        edges = []
        if 0 in mask[0]:
            edge = 'top'
            logger.warn("View %s screen %s mask reaches %s edge", self.currView, self.currScreen, edge)
            edges.append(edge)
        if 0 in mask[-1]:
            edge = 'bottom'
            logger.warn("View %s screen %s mask reaches %s edge", self.currView, self.currScreen, edge)
            edges.append(edge)
        if 0 in mask[:, [0]]:
            edge = 'left'
            logger.warn("View %s screen %s mask reaches %s edge", self.currView, self.currScreen, edge)
            edges.append(edge)
        if 0 in mask[:, [-1]]:
            edge = 'right'
            logger.warn("View %s screen %s mask reaches %s edge", self.currView, self.currScreen, edge)
            edges.append(edge)

        return bool(len(edges)), edges


    def findCorners(self, img):
        blocksize = 2
        sobel = 3
        free = 0.04
        points = []

        #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = np.float32(img)

        gray = cv2.dilate(gray,None)
        dst = cv2.cornerHarris(gray, blocksize, sobel, free)

        logger.debug(dst.max())
        th = dst.max() * 0.005

        for i in range(len(dst)):
            for j in range(len(dst[i])):
                val = dst[i][j]
                if val > th:
                    # logger.debug("%s %s: %s", i, j, val)
                    points.append((j, i))

        logger.debug(dst.shape)
        #dst = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
        dst = np.uint8(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = np.float32(points)
        cv2.cornerSubPix(dst, corners, (5,5), (-1,-1), criteria)
        return corners


    def findChessboard(self, img, name, cnrx, cnry):
        logger.info("Looking for %s x %s inside corners", cnrx, cnry)

        status, data = cv2.findChessboardCorners(img, (cnrx, cnry), flags=cv2.cv.CV_CALIB_CB_ADAPTIVE_THRESH)
        if status == False:
            logger.error("Failed to parse checkerboard pattern in image")
            raise NotFoundError("Failed to parse checkerboard pattern in image")

        logger.info("Checkerboard found")
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(vis, (cnrx, cnry), data, status)
        self.putImage(vis, name)
        return data


    def acquireImage(self, imgname, width=None, height=None, focus=False):
        img = img_string_to_cv(self.camera.acquireImage(imgname, focus))
        if img is None:
            raise ImageCaptureError("Failed to read image %s", imgname)
        #if width is not None and height is not None:
        #    img = img_cv_resize(img, width, height, 'outer')
        return img


    def putImage(self, img, name=None, wait=0, view=None, screen=None):
        if view is None: view=self.currView
        if screen is None: screen=self.currScreen
        name = '-' + name if name is not None else ''

        outdir = "{0}/view{1}".format(self.outdir, view)
        mkdir_p(outdir)

        seq = 0
        while(True):
            path = "{0}/view{1}-screen{2}{3}.{4}.png".format(outdir, view, screen, name, seq)
            if not os.path.isfile(path): break
            seq += 1

        logger.debug("Image %s is a %s", path, type(img).__name__)

        if isinstance(img, str):
            logger.debug('string')
            with open(path, 'wb') as f: f.write(img)
            self.sendCommand('displayImage', window=screen, image=cStringIO.StringIO(img))
        elif isinstance(img, np.ndarray):
            logger.debug("ndarray")
            cv2.imwrite(path, img)
            self.sendCommand('displayImage', window=screen, image=img_cv_to_stringio(img))
        elif type(img).__name__ == 'StringI':
            with open(path, 'wb') as f: f.write(img.getvalue())
            self.sendCommand('displayImage', window=screen, image=img)
        self.wait(wait)


    def wait(self, sleep=0, force_wait=False):
        time.sleep(sleep)
        if not self.args.autorun or force_wait:
            logger.debug('worker sleep')
            with self.cond:
                self.cond.wait()
            logger.debug('worker wake')
        if self._want_abort:
            logger.debug('worker aborted')
            exit()


    def abort(self):
        self._want_abort = 1
        with self.cond:
            self.cond.notify()


    def sendCommand(self, action, **kwargs):
        logger.debug("Sending command %s", action)
        res = kwargs
        res['action'] = action
        wx.PostEvent(self.gui, CommandEvent(res))


def parseCmdline():
    parser = argparse.ArgumentParser(description='''
        TODO: insert description.'''
    )
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose output")
    parser.add_argument('-q', '--quiet', action='store_true', help="Output errors only")

    parser.add_argument('-a', '--autorun', action='store_true', help="Do not stop after each step")
    parser.add_argument('-C', '--usecache', action='store_true', help="Use images from cache")

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
