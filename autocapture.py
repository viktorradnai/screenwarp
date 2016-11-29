#!/usr/bin/python

import cv2
import sys
import time
import json
import logging
import argparse

import wx
import sys
import logging
import argparse
import cStringIO

import wand.image
import wand.color
import wand.drawing
import requests
import traceback
import numpy as np
from threading import *

logger = logging.getLogger(__name__)

#IMAGE_CAPTURE_URL = 'http://192.168.0.237:8080/photoaf.jpg'
IMAGE_CAPTURE_URL = 'http://192.168.43.194:8080/photoaf.jpg'


class ImageCaptureError(Exception):
    pass

class NotFoundError(Exception):
    pass

class Aborted(Exception):
    pass

class AndroidIPCamera:
    def acquireImage(self, imgname):
        logger.debug("Calling %s", IMAGE_CAPTURE_URL)
        res = requests.get(IMAGE_CAPTURE_URL)
        logger.debug(res)
        if not res:
            logger.error("Failed to get image from IP Webcam, no response received")
            return False
        if res.status_code != 200:
            logger.error("Failed to get image from IP Webcam, service returned status %s", res.status_code)
            return False

        return res.content


class FileCacheCamera:
    def acquireImage(self, imgname):
        path = 'cache/{0}'.format(imgname)
        logger.info("Loading image from %s", path)
        with open(path, 'rb') as f:
            return f.read()


# Define notification event for thread completion
EVT_COMMAND_ID = wx.NewId()

def EVT_COMMAND(win, func):
     """Define Result Event."""
     win.Connect(-1, -1, EVT_COMMAND_ID, func)


class CommandEvent(wx.PyEvent):
     """Simple event to carry arbitrary result data."""
     def __init__(self, data):
         """Init Result Event."""
         wx.PyEvent.__init__(self)
         self.SetEventType(EVT_COMMAND_ID)
         #logger.debug(data)
         self.data = data


class WindowFrame(wx.Frame):
    pass

class ScreenFrame(wx.Panel):

    def __init__(self, *args, **kwargs):
        # logger.debug(args)
        self.parent = args[0]
        super(ScreenFrame, self).__init__(*args[1:], **kwargs)
        self.panel = wx.Panel(self, wx.ID_ANY, pos=(0, 0), size=self.GetSize(), style=wx.NO_BORDER)
        self.panel.SetBackgroundColour("#000")
        self.panel.Bind(wx.EVT_KEY_DOWN, self.parent.onKeyPress)
        cursor = wx.StockCursor(wx.CURSOR_BLANK)
        self.SetCursor(cursor)


    def displayImage(self, image):
        logger.debug('displayImage')
        for c in self.panel.GetChildren():
            #logger.debug(c)
            c.Destroy()
        img = wx.ImageFromStream(image).ConvertToBitmap()
        wx.StaticBitmap(self.panel, -1, img, (0, 0), (img.GetWidth(), img.GetHeight()))
        self.panel.SetFocus()
        self.SetFocus()
        self.panel.Show()
        self.Show()

    def displayColor(self, color):
        logger.debug('displayColor')
        for c in self.panel.GetChildren():
            logger.debug(c)
            c.Destroy()
        self.panel.SetBackgroundColour(color)
        self.panel.Show()


class MainApp(wx.App):
    screens = {}
    windows = {} # WindowFrame objects
    frames = {}  # ScreenFrame objects

    def OnInit(self):
        self.cond = Condition()

        # Set up event handler for any worker thread results
        EVT_COMMAND(self, self.onCommand)

        args = self.parseCmdline()
        self.args = args

        if args.usecache:
            try:
                with open('cache/screens.json') as f:
                    displays = json.load(f)['displays']
            except Exception as e:
                logger.error("Caught exception, exiting")
                logger.error(traceback.format_exc())
        else:
            displays = self.getDisplays()

        """
            Open windows in each display but create a frame for each screen.
            A display may be made of more than one screen, eg. if using Xinerama
            or a hardware solution such as a Matrox TripleHead2Go
        """
        for dname in displays.keys():
            size = ( displays[dname]['width'], displays[dname]['height'] )
            # Hack to position windows sensibly in cache mode
            topleft = (0, 0) if args.usecache else ( displays[dname]['top'], displays[dname]['left'] )

            title = "Display "+dname
            logger.info('Creating window "%s" on display %s with size %s x %s offset %s %s', title, dname, size[0], size[1], topleft[0], topleft[1])
            window = WindowFrame(None, -1, title, topleft, size)
            if not args.usecache:
                screen.ShowFullScreen(True, style=wx.FULLSCREEN_ALL)
            self.windows[dname] = window

            screens = self.getScreens(dname, size[0], size[1])

            for sname in screens.keys():
                size = ( screens[sname]['width'], screens[sname]['height'] )
                logger.info('Creating frame for screen "%s" in window %s with size %s x %s offset %s %s', sname, title, size[0], size[1], topleft[0], topleft[1])
                #frame = ScreenFrame(self, window, -1, "Screen "+sname, screens[sname]['topleft'], size)
                frame = ScreenFrame(self, window, -1, screens[sname]['topleft'], size, style=wx.NO_BORDER)
                self.frames[sname] = frame

            self.screens.update(screens)
            window.Show()
        self.worker = WorkerThread(self, self.screens, args, self.cond)
        return True


    def getScreens(self, dname, width, height):
        screens = {}
        ar = height / (width / 16)
        if ar < 9:

            logger.debug("Multiscreen display detected, splitting")
            cols = int(9 / ar)
            width /= cols
            logger.debug("Cols: %s", cols)
            for i in range(cols):
                screens["{0}-{1}".format(dname, i)] = { 'topleft': (width*i, 0), 'width': width, 'height': height }
        elif ar > 12:
            rows = int(ar / 9)
            height /= rows
            logger.debug("Rows: %s", rows)
            for i in range(rows):
                screens["{0}.{1}".format(dname, i)] = { 'topleft': (0, height*i), 'width': width, 'height': height }
        else:
            screens[dname] = { 'topleft': (0, 0), 'width': width, 'height': height }

        return screens


    def getDisplays(self):
        displays = {}
        # Find distinct displays, split aggretates if necessary
        for i in range(wx.Display.GetCount()):
            display = wx.Display(i)
            geometry = display.GetGeometry()
            topleft = geometry.GetTopLeft()
            size = geometry.GetSize()
            logger.info("Display %s has size %s x %s at offset %s %s", i, size[0], size[1], topleft[0], topleft[1])
            displays["{0}".format(i)] = { 'topleft': topleft, 'width': size[0], 'height': size[1] }
            return displays


    def onKeyPress(self, event):
        keycode = event.GetKeyCode()
        print keycode
        if keycode == wx.WXK_SPACE:
            with self.cond:
                self.cond.notify()
        if keycode == wx.WXK_ESCAPE:
            self.Close()
            self.worker.abort()
        event.Skip()


    def onCommand(self, event):
        if not event.data:
            logger.error("No data received, exiting")
            self.Close()
        logger.debug("Received command")
        #logger.debug(event.data)
        action = event.data['action']
        if action == 'displayImage':
            self.frames[event.data['window']].displayImage(event.data['image'])
        elif action == 'displayColor':
            self.frames[event.data['window']].displayColor(event.data['color'])
        elif action == 'exit':
            self.Close()
            return
        else:
            logger.error("Unknown action %s", action)
            return

        event.Skip()


    def Close(self):
        for f in self.windows.values():
            f.Close()


    def parseCmdline(self):
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



class WorkerThread(Thread):
    def __init__(self, gui, windows, args, cond):
        Thread.__init__(self)
        self.gui = gui
        self.screens = windows
        self._want_abort = 0
        self.args = args
        self.cond = cond
        if args.usecache:
            self.camera = FileCacheCamera()
        else:
            self.camera = AndroidIPCamera()
        self.start()


    def run(self):
        rows = 40
        cols = 24
        cnrx = rows - 1
        cnry = cols - 1
        try:
            for wname in self.screens.keys():
                w = self.screens[wname]
                screenid = draw_text(wname, w['width'], w['height'])
                self.sendCommand('displayImage', window=wname, image=screenid)
            self.wait(1)

            for wname in self.screens.keys():
                w = self.screens[wname]
                self.sendCommand('displayColor', window=wname, color='#000')

            logger.debug(self.screens)
            for wname in self.screens.keys():
                w = self.screens[wname]

                if wname not in self.args.displays:
                    logger.info("Skipping display %s", wname)
                    continue

                logger.info("Processing display %s", wname)
                self.wait()

                cb = self.drawChessboard(w['width'], w['height'], rows, cols, '#fff', '#000')
                with open('cb1.png', 'wb') as f: f.write(cb.getvalue())
                self.sendCommand('displayImage', window=wname, image=cb)
                self.wait()

                img1 = self.acquireImage("window{0}-img{1}.png".format(wname, 1), w['width'], w['height'])
                self.sendCommand('displayImage', window=wname, image=img_cv_to_stringio(img1))
                self.wait()

                cb = self.drawChessboard(w['width'], w['height'], rows, cols, '#000', '#fff')
                with open('cb2.png', 'wb') as f: f.write(cb.getvalue())
                self.sendCommand('displayImage', window=wname, image=cb)
                self.wait()

                img2 = self.acquireImage("window{0}-img{1}.png".format(wname, 2), w['width'], w['height'])
                self.sendCommand('displayImage', window=wname, image=img_cv_to_stringio(img2))
                self.wait()


                img = cv2.subtract(img1, img2)
                self.sendCommand('displayImage', window=wname, image=img_cv_to_stringio(img))
                cv2.imwrite("window{0}-img{1}.png".format(wname, 3), img)
                imgdiff1 = img
                self.wait()

                img = cv2.subtract(img2, img1)
                self.sendCommand('displayImage', window=wname, image=img_cv_to_stringio(img))
                cv2.imwrite("window{0}-img{1}.png".format(wname, 4), img)
                imgdiff2 = img
                self.wait()

                imgdiff = cv2.add(imgdiff1, imgdiff2)
                self.sendCommand('displayImage', window=wname, image=img_cv_to_stringio(imgdiff))
                cv2.imwrite("window{0}-img{1}.png".format(wname, 'diff'), imgdiff)
                self.wait()

                img = cv2.cvtColor(imgdiff, cv2.COLOR_BGR2GRAY)
                img = cv2.medianBlur(img, 13)
                #blur = cv2.GaussianBlur(img,(5,5),0)
                ret, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                self.sendCommand('displayImage', window=wname, image=img_cv_to_stringio(mask))
                cv2.imwrite("window{0}-img{1}.png".format(wname, 'mask'), mask)
                self.wait()


                img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                ret, img1a = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                #img = cv2.add(img, mask)
                self.sendCommand('displayImage', window=wname, image=img_cv_to_stringio(img))
                cv2.imwrite("window{0}-img{1}.png".format(wname, 7), img1a)
                self.wait()

                try:
                    vis = self.findChessboard(img1a, cnrx, cnry)
                    self.sendCommand('displayImage', window=wname, image=img_cv_to_stringio(vis))
                    cv2.imwrite("window{0}-img{1}.png".format(wname, '-chess1'), vis)
                    self.wait()
                except NotFoundError:
                    pass

                img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                ret, img2a = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                #img = cv2.add(img, mask)
                self.sendCommand('displayImage', window=wname, image=img_cv_to_stringio(img2a))
                cv2.imwrite("window{0}-img{1}.png".format(wname, 8), img)
                self.wait()

                try:
                    vis = self.findChessboard(img2a, cnrx, cnry)
                    self.sendCommand('displayImage', window=wname, image=img_cv_to_stringio(vis))
                    cv2.imwrite("window{0}-img{1}.png".format(wname, '-chess2'), vis)
                    self.wait()
                except NotFoundError:
                    pass
                self.wait()

                points = self.findCorners(img)
                img = draw_points(points, img.shape[1], img.shape[0], 0, 0)
                with open("window{0}-img{1}.png".format(wname, 'corners'), 'wb') as f: f.write(img.getvalue())
                self.sendCommand('displayImage', window=wname, image=img)
                self.wait()

                bw = 10
                mask2 = cv2.copyMakeBorder(mask, top=bw, bottom=bw, left=bw, right=bw, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255] )
                #mask2 = mask.copy()

                (cnts, x) = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                logger.info("Found %s contours", len(cnts))
                cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[1:2]
                #cv2.drawContours(mask2, cnts, -1, (255,0,0), 2)

                mask2 = np.zeros(mask.shape, np.uint8)
                cv2.fillPoly(mask2, pts=cnts, color=(255,255,255))
                self.sendCommand('displayImage', window=wname, image=img_cv_to_stringio(mask2))
                cv2.imwrite("window{0}-img{1}.png".format(wname, '-contours'), mask2)
                self.wait(1)
                logger.debug("Finished screen %s", wname)
                self.sendCommand('displayColor', window=wname, color='#000')

                #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
                #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
                #img3 = cv2.fastNlMeansDenoising(img3,None,40,17,31)


        except Exception as e:
            logger.error("Caught exception, exiting")
            logger.error(traceback.format_exc())
        finally:
            self.sendCommand('exit')
            exit()


    def findCorners(self, img):
        blocksize = 2
        sobel = 3
        free = 0.04
        points = []

        #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = np.float32(img)
        dst = cv2.cornerHarris(gray, blocksize, sobel, free)

        logger.debug(dst.max())
        th = dst.max() * 0.005

        for i in range(len(dst)):
            for j in range(len(dst[i])):
                val = dst[i][j]
                if val > th:
                    # logger.debug("%s %s: %s", i, j, val)
                    points.append((j, i))

        return points


    def findChessboard(self, img, cnrx, cnry):
        logger.info("Looking for %s x %s inside corners", cnrx, cnry)

        status, data = cv2.findChessboardCorners(img, (cnrx, cnry), flags=cv2.cv.CV_CALIB_CB_ADAPTIVE_THRESH)
        if status == False:
            logger.error("Failed to parse checkerboard pattern in image")
            raise NotFoundError("Failed to parse checkerboard pattern in image")

        logger.info("Checkerboard found")
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(vis, (cnrx, cnry), data, status)
        return vis


    def acquireImage(self, imgname, width, height):
        img = img_string_to_cv(self.camera.acquireImage(imgname))
        if img is None:
            raise ImageCaptureError("Failed to read image %s", imgname)
        img = img_cv_resize(img, width, height, 'outer')
        cv2.imwrite(imgname, img)
        return img


    def wait(self, sleep=0):
        time.sleep(sleep)
        if not self.args.autorun:
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


    def drawChessboard(self, screen_width, screen_height, cols, rows, color1='#fff', color2='#000'):
        square_width = screen_width / cols
        square_height = screen_height / rows

        image = wand.image.Image(width=screen_width, height=screen_height, background=wand.color.Color(color1))
        with wand.drawing.Drawing() as draw:
            draw.fill_color = wand.color.Color(color2)
            draw.color = wand.color.Color(color2)
            for r in range(rows):
                for c in range(cols):
                    if not (c + r) % 2:
                        continue
                    x = square_width * c
                    y = square_height * r

                    # logger.debug("%s %s %s %s", x, y, square_width, square_height)
                    draw.rectangle(x, y, width=square_width-1, height=square_height-1)

            draw.draw(image)
            blob = image.make_blob('png')
        return cStringIO.StringIO(blob)


def draw_points(points, width, height, xoff, yoff):
    image = wand.image.Image(width=width, height=height, background=wand.color.Color('#fff'))

    with wand.drawing.Drawing() as draw:
        draw.fill_color = wand.color.Color('#f00')
        for p in range(len(points)):
            # draw.fill_color = wand.color.Color('#{0:x}{0:x}{0:x}'.format(r*2))
            x = points[p][0] + xoff
            y = points[p][1] + yoff
            draw_point(draw, x, y)

        draw.draw(image)
        blob = image.make_blob('png')
    return cStringIO.StringIO(blob)


def draw_grid(grid, width, height, xoff, yoff):
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
        blob = image.make_blob('png')
    return cStringIO.StringIO(blob)


def draw_point(draw, x, y):
    draw.point(x, y)
    draw.point(x+1, y)
    draw.point(x-1, y)
    draw.point(x, y+1)
    draw.point(x, y-1)


def draw_text(text, width, height):
    font_size = 400
    image = wand.image.Image(width=width, height=height, background=wand.color.Color('#fff'))
    with wand.drawing.Drawing() as draw:
        draw.fill_color = wand.color.Color('#000')
        draw.font_size = font_size
        draw.text_alignment = 'center'
        draw.text(width/2, height/2+font_size/2, text)

        draw.draw(image)
        blob = image.make_blob('png')
    return cStringIO.StringIO(blob)


def img_string_to_cv(string):
    nparr = np.fromstring(string, np.uint8)
    return cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)


def img_cv_to_stringio(img):
    nparr = cv2.imencode('.png', img)[1]
    return cStringIO.StringIO(nparr.tostring())


def img_cv_resize(img, width, height, style='exact'):
    h, w, channels = img.shape
    scale_h = float(height) / h
    scale_w = float(width) / w
    if style == 'inner':
        scale_h = scale_h if scale_h < scale_w else scale_w
        scale_w = scale_h
    elif style == 'outer':
        scale_h = scale_h if scale_h > scale_w else scale_w
        scale_w = scale_h

    return cv2.resize(img, None, fx=scale_w, fy=scale_h, interpolation=cv2.INTER_CUBIC)


if __name__ == '__main__':
    app = MainApp(0)
    app.MainLoop()
