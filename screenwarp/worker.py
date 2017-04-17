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
import gui
from camera import FileCacheCamera, AndroidIPCamera, CommandLineCamera
import vision as sv
from draw import *
from util import *

logger = logging.getLogger(__name__)

#IMAGE_CAPTURE_URL = 'http://192.168.0.237:8080'
IMAGE_CAPTURE_URL = 'http://192.168.43.194:8080'

WORKER = None
def putImage(*args, **kwargs):
    return WORKER.putImage(*args, **kwargs)

# Make it callable from any module
__builtin__.putImage = putImage

class ImageCaptureError(Exception):
    pass


class WorkerThread(Thread):
    def __init__(self, views, screens, cond):
        global WORKER
        Thread.__init__(self)
        WORKER = self
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
        cols = 40
        rows = 24
        try:
            # Take a photo of each screen from each view, then process them
            for vname, view in iter(sorted(self.views.iteritems())):
                # Identify each screen
                for sname in self.screens:
                    w = self.screens[sname]
                    screenid = draw_text(sname, w['width'], w['height'])
                    gui.sendCommand('displayImage', window=sname, image=screenid)
                self.wait(force_wait=True)
                if args.delay:
                    time.sleep(args.delay)

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
                        self.camera = self.defaultcamera

                # calibrate camera focus and exposure
                self.acquireImage(vname, 1280, 960, focus=True)
                # Black out all screens
                for sname in self.screens:
                    w = self.screens[sname]
                    gui.sendCommand('displayColor', window=sname, color='#000')


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

                    self.calibrate(vname, sname, view, screen)

                    logger.debug("Finished screen %s", sname)
                    gui.sendCommand('displayColor', window=sname, color='#000')


        except Exception as e:
            logger.error("Caught exception, exiting")
            logger.error(traceback.format_exc())
        finally:
            gui.sendCommand('exit')
            exit()


    def calibrate(self, vname, sname, view, screen):

        width = screen['width']
        height = screen['height']
        screenview = sv.ScreenView(vname, sname, view, screen)
        photos = screenview.getPhotoGroup()
        commonfactors = sorted(factors(width).intersection(factors(height)), reverse=True)
        gcf = commonfactors[0]
        col_base = width / gcf
        row_base = height / gcf
        logger.debug("Greatest common divisor: %s, col_base: %s row_base: %s", gcf, col_base, row_base)

        # do baseline capture to get mask
        img1, img2 = self.acquireCheckboards(vname, sname, width, height, col_base*8, row_base*8)
        mask, coords = self.doMaskAndCalibration(vname, sname, width, height, col_base*8, row_base*8, (img1, img2))

        for i in (1, 2, 4, 8, 16, 32):
            cols = col_base * i
            rows = row_base * i

            # capture vertical bars
            vs1, vs2 = self.acquireCheckboards(vname, sname, width, height, cols, 1)
            photos.addPhotos(vs1, vs2, cols, 1)
            # capture horizontal bars
            hs1, hs2 = self.acquireCheckboards(vname, sname, width, height, 1, rows)
            photos.addPhotos(hs1, hs2, 1, rows)
            # capture chessboard
            cb1, cb2 = self.acquireCheckboards(vname, sname, width, height, cols, rows)
            photos.addPhotos(cb1, cb2, cols, rows)

            # odd stripes
            vbars1 = self.findStripes(vname, sname, width, height, cols*i, 1, (vs1, vs2), mask)
            # even stripes
            vbars2 = self.findStripes(vname, sname, width, height, cols*i, 1, (vs2, vs1), mask)

            # odd stripes
            hbars1 = self.findStripes(vname, sname, width, height, 1, rows*i, (hs1, hs2), mask)
            # even stripes
            hbars2 = self.findStripes(vname, sname, width, height, 1, rows*i, (hs2, hs1), mask)

            if i == 1:
                onedge, edges = sv.isMaskOnEdge(mask)
                vsl = {}
                vsl1 = vbars1.getSlices()
                vsl2 = vbars2.getSlices()
                for s in vsl1:
                    key = s.getBoundingRect()[0]
                    vsl[key] = s
                for s in vsl2:
                    key = s.getBoundingRect()[0]
                    vsl[key] = s

                #screenview.addSliceGroup(vbars, i+'-v')

                slices = vbars1.getSlices()
                logger.debug("vbars1 has %s slices", len(slices))
                slices = vbars2.getSlices()
                logger.debug("vbars2 has %s slices", len(slices))
                slices = hbars1.getSlices()
                logger.debug("hbars1 has %s slices", len(slices))
                slices = hbars2.getSlices()
                logger.debug("hbars2 has %s slices", len(slices))
                break

            for s in vbars1.getSlices():
                if s.getParent() is not None:
                    continue


            intersect = cv2.add(vbars.getSlice(i).toMask(), hbars.getSlice(i).toMask())
            self.putImage(vbars.getSlice(i).toMask(), 'intersecta')
            self.putImage(hbars.getSlice(i).toMask(), 'intersectb')
            self.putImage(intersect, 'intersectc')

            screenview.addSliceGroup(hbars, i+'-h')
        return

        if False or coords is None:
            logger.warn("Failed to find coordinates for screen %s in view %s, trying to calibrate parts of the screen", sname, vname)
            self.doPartialCalibration(vname, sname, screen['width'], screen['height'], mask)

        if False: # this is an attempt to detect corners, it is currently much less reliable than findChessboard
            points = self.findCorners(img1m)
            img = draw_points(points, img1m.shape[1], img1m.shape[0], 0, 0)
            self.putImage(img, 'img1-points')


    def findStripes(self, vname, sname, width, height, cols, rows, images, mask=None):
        img1, img2 = images
        imgdiff1 = cv2.subtract(img1, img2)
        self.putImage(imgdiff1, 'imgdiff1')
        imgdiff2 = cv2.subtract(img2, img1)
        self.putImage(imgdiff2, 'imgdiff2')

        img1m = sv.applyMask(imgdiff1, mask)
        img2m = sv.applyMask(imgdiff2, mask)

        img = img1m # cv2.cvtColor(img1m, cv2.COLOR_BGR2GRAY)
        #img = cv2.medianBlur(img, 13)
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        kernel = np.ones((5,5),np.uint8)
        #imgc = cv2.morphologyEx(imgb, cv2.MORPH_OPEN, kernel)
        tmp = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        imgc = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, kernel)

        self.putImage(imgc, 'stripes')
        border = 10
        imgb = cv2.copyMakeBorder(imgc, top=border, bottom=border, left=border, right=border, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0] )

        (cnts, x) = cv2.findContours(imgc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        limit = rows * cols * 2
        logger.info("Found %s contours in mask image, filtering to %s", len(cnts), limit)
        #cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:limit]
        image = np.zeros(imgb.shape, np.uint8)

        h, w = image.shape[:2]
        filtered = []
        approximated = []

        for c in cnts:
            p = []
            #logger.debug(c)
            area = cv2.contourArea(c)
            #logger.debug(area)
            if area < 200:
                continue
            filtered.append(c)

        slicegroup = sv.SliceGroup(sname, width, height, rows, cols, filtered)
        cv2.drawContours(image, filtered, -1, (255,0,0), 2)
        self.putImage(image, 'stripes2')
        return slicegroup


    def doMaskAndCalibration(self, vname, sname, width, height, cols, rows, images, mask=None, x=0, y=0, square_width=None, square_height=None):
        # There is one less inside corner in each row and column than the number of squares we display
        cnrx = cols - 1
        cnry = rows - 1

        if images is not None:
            img1, img2 = images
        else:
            logger.debug("Display and capture chessboard")
            # Put up chessboard image, take photo. Invert chessboard and repeat.
            img1, img2 = self.acquireCheckboards(vname, sname, width, height, cols, rows, x, y, square_width, square_height)

        if mask is None:
            logger.info("Creating mask")
            # Subtract positive and negative chessboard to obtain mask
            mask = sv.createMask(img1, img2)

            # Warn if mask reaches edge -- full grid is probably not shown
            sv.isMaskOnEdge(mask)

        logger.debug("Applying mask")
        # get masked, thresholded chessboards
        img1m = sv.applyMask(img1, mask)
        self.putImage(img1m, '{0}x{1}-diff1-masked'.format(cols, rows))
        img2m = sv.applyMask(img2, mask)
        self.putImage(img2m, '{0}x{1}-diff2-masked'.format(cols, rows))

        logger.debug("Find chessboard")
        # Try to find chessboard in either image. Normally both should succeed if the whole chessboard is displayed.
        try:
            coords = sv.findChessboard(img1m, cnrx, cnry)
        except sv.NotFoundError:
            # TODO: try harder to find what's left of the chessboard
            pass

        try:
            coords = sv.findChessboard(img2m, cnrx, cnry)
        except sv.NotFoundError:
            return mask, None

        return mask, coords


    def doPartialCalibration(self, vname, sname, width, height, mask):
        x, edges = sv.isMaskOnEdge(mask)

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
        x__, y__ = gui.get().GetScreenPosition(sname)
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


    def acquireCheckboards(self, view, screen, width, height, cols, rows, x=0, y=0, square_width=None, square_height=None):
        # Display chessboard image then take a photo
        cb = drawChessboard(width, height, cols, rows, '#fff', '#000', x, y, square_width, square_height)
        self.putImage(cb, 'cb1')

        imgname = "view{0}-screen{1}-cb{2}x{3}-img{4}".format(view, screen, rows, cols, 1)
        if rows > 50 or cols > 50:
            img1 = self.acquireImage(imgname)
        else:
            img1 = self.acquireImage(imgname, width, height)
        self.putImage(img1, 'img1')

        # Display inverted chessboard image, take another photo
        cb = drawChessboard(width, height, cols, rows, '#000', '#fff', x, y, square_width, square_height)
        self.putImage(cb, 'cb2')

        imgname = "view{0}-screen{1}-cb{2}x{3}-img{4}".format(view, screen, rows, cols, 2)
        if rows > 50 or cols > 50:
            img2 = self.acquireImage(imgname)
        else:
            img2 = self.acquireImage(imgname, width, height)
        self.putImage(img2, 'img2')

        return img1, img2


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


    def acquireImage(self, imgname, width=None, height=None, focus=False):
        img = img_string_to_cv(self.camera.acquireImage(imgname, focus))
        if img is None:
            raise ImageCaptureError("Failed to read image %s", imgname)
        if width is not None and height is not None:
            img = img_cv_resize(img, width, height, 'outer')
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
        width = self.screens[screen]['width']
        height = self.screens[screen]['height']
        if isinstance(img, str):
            with open(path, 'wb') as f: f.write(img)
            gui.sendCommand('displayImage', window=screen, image=cStringIO.StringIO(img))
        elif isinstance(img, np.ndarray):
            cv2.imwrite(path, img)
            imgr = img_cv_resize(img, width, height, 'inner')
            gui.sendCommand('displayImage', window=screen, image=img_cv_to_stringio(imgr))
        elif type(img).__name__ == 'StringI':
            with open(path, 'wb') as f: f.write(img.getvalue())
            gui.sendCommand('displayImage', window=screen, image=img)
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


