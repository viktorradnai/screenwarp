import cv2
import logging
import numpy as np

from draw import *
from util import *

logger = logging.getLogger(__name__)

class NotFoundError(Exception):
    pass

class PhotoGroup():
    photos = {}

    def addPhotos(self, img1, img2, cols, rows):
        key = "{0}x{1}".format(cols, rows)
        self.photos[key] = [img1, img2]

    def getPhotos(self, cols, rows):
        key = "{0}x{1}".format(cols, rows)
        return self.photos[key]


class ScreenView():
    def __init__(self, vname, sname, view, screen):
        self.currView = vname
        self.currScreen = sname
        self.slicegroups = {}
        self.coordMap = np.zeros((screen['width'], screen['height']))
        self.photos = PhotoGroup()

    def addSliceGroup(self, slicegroup, key):
        self.slicegroups[key] = slicegroup

    def gegSliceGroup(self, key):
        if not key in slicegroups:
            raise NotFoundError("Key {0} not found in slice group".format(key))
        return self.slicegroups[key]

    def addCoord(self, x, y, xt, yt, intensity):
        hits = self.coordMap[x][y][3]
        self.coordMap[x][y][0] = (self.coordMap[x][y][0] * hits + xt) / (hits + 1)
        self.coordMap[x][y][1] = (self.coordMap[x][y][1] * hits + yt) / (hits + 1)
        self.coordMap[x][y][2] = (self.coordMap[x][y][2] * hits + intensity) / (hits + 1)

    def getCoordMap(self):
        return self.coordMap

    def getPhotoGroup(self):
        return self.photos


class SliceGroup():
    cut = None
    def __init__(self, screen, width, height, rows, cols, contours):
        if rows == 1:
            self.cut = 'horizontal'
        elif cols == 1:
            self.cut = 'vertical'
        self.contours = contours
        slices = []
        for c in contours:
            s = ScreenSlice(screen, self.cut, width, height, c)
            slices.append(s)

        def sorter(x):
            if self.cut == 'horizontal':
                return x.getBoundingRect()[0]
            else:
                return x.getBoundingRect()[1]
        self.slices = sorted(slices, key=sorter)

        for s in self.slices:
            putImage(s.toMask(), "sslice-s{0}x{1}".format(rows, cols))

    def getSlices(self):
        return self.slices

    def getSlice(self, n):
        return self.slices[n]


class ScreenSlice():
    def __init__(self, screen, cut, width, height, contour):
        self.screen = screen
        if not cut in ('horizontal', 'vertical', 'cross'):
            err = "Cut must be either 'horizontal', 'vertical' or 'cross'"
            logger.error(err)
            # raise ParameterError(err)
        self.cut = cut
        self.width = width
        self.height = height
        self.contour = contour
        self.parent = None
        self.children = []

        self.mask = createMaskFromCntr(contour, width, height)

    def getCut(self):
        return self.cut

    """Returns a list in the form of (x, y, w, h)"""
    def getBoundingRect(self):
        return cv2.boundingRect(self.contour)

    def getCentroid(self):
        pass

    def addChild(self, child):
        self.children.append(child)

    def getChildren(self):
        return self.children

    def toMask(self, invert=False):
        if invert:
            imask = (255 - self.mask)
            return imask
        else:
            return self.mask

    def onEdge(self):
        return isMaskOnEdge(self.mask)



# Helper functions

def applyMask(img, mask):
    # Mask out everything other than the chessboard area
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_bw.shape != mask.shape:
        mask = img_cv_resize(mask, img_bw.shape[1], img_bw.shape[0])
    ret, img_bw = cv2.threshold(img_bw, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    return cv2.add(img_bw, mask)


def createMask(img1, img2):
    # Subtract two images showing inverted pattern. This blacks out non-display areas that only have diffuse light.
    imgdiff1 = cv2.subtract(img1, img2)
    putImage(imgdiff1, 'imgdiff1')
    imgdiff2 = cv2.subtract(img2, img1)
    putImage(imgdiff2, 'imgdiff2')

    # Add two diff images together to fill visible area, then threshold it to create mask
    imgdiff = cv2.add(imgdiff1, imgdiff2)
    putImage(imgdiff, 'imgdiff')
    img = cv2.cvtColor(imgdiff, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 13)
    ret, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    putImage(mask, 'mask')

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
    putImage(mask2, 'mask2')

    return mask2


def isMaskOnEdge(mask):
    edges = []
    if 0 in mask[0]:
        edges.append('top')
    if 0 in mask[-1]:
        edges.append('bottom')
    if 0 in mask[:, [0]]:
        edges.append('left')
    if 0 in mask[:, [-1]]:
        edges.append('right')

    onedge = bool(len(edges))
    if onedge:
        logger.warn("Mask reaches edge(s): %s", ", ".join(edges))
    return onedge, edges


def createMaskFromCntr(contour, width, height):
    contour = np.reshape(contour, (1, -1, 2))
    mask = np.full((height+20, width+20, 3), 255, np.uint8)
    cv2.fillPoly(mask, pts=(contour), color=(0,0,0))
    return mask



def findChessboard(img, cnrx, cnry, visualize=False):
    logger.info("Looking for %s x %s inside corners", cnrx, cnry)

    status, data = cv2.findChessboardCorners(img, (cnrx, cnry), flags=cv2.cv.CV_CALIB_CB_ADAPTIVE_THRESH)
    if status == False:
        logger.error("Failed to parse checkerboard pattern in image")
        raise NotFoundError("Failed to parse checkerboard pattern in image")

    logger.info("Checkerboard found")
    if visualize:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(vis, (cnrx, cnry), data, status)
        self.putImage(vis, 'cbvis')
    return data
