import os
import wx
import cv2
import json
import errno
import logging
import cStringIO
from threading import *
from screenwarp.threads import WorkerThread

logger = logging.getLogger(__name__)


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
        #logger.debug('displayImage')
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
            #logger.debug(c)
            c.Destroy()
        self.panel.SetBackgroundColour(color)
        self.panel.Show()


class MainApp(wx.App):
    screens = {}
    windows = {} # WindowFrame objects
    frames = {}  # ScreenFrame objects

    def OnInit(self):
        viewdb = 'views.json'
        screendb = 'screens.json'
        self.cond = Condition()

        # Set up event handler for any worker thread results
        EVT_COMMAND(self, self.onCommand)

        self.args = args # from __builtin__

        with open(viewdb) as f:
            logger.info("Using camera view definitions from %s", viewdb)
            self.views = json.load(f)['views']

        if os.path.isfile(screendb):
            try:
                logger.warn("Overriding screen definitions from %s", screendb)
                with open(screendb) as f:
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
        for dname in displays:
            d = displays[dname]
            logger.debug(d)
            size = ( d['width'], d['height'] )
            # Hack to position windows sensibly in cache mode
            topleft = (0, 0) if args.usecache else ( d['top'], d['left'] )

            title = "Display "+dname
            logger.info('Creating window "%s" on display %s with size %s x %s offset %s %s', title, dname, size[0], size[1], topleft[0], topleft[1])
            window = WindowFrame(None, -1, title, topleft, size)
            if not args.usecache and ('fullscreen' not in d or d['fullscreen']):
                window.ShowFullScreen(True, style=wx.FULLSCREEN_ALL)
            else:
                # set the effective size of the display (not the size including title bar and borders)
                window.SetClientSize(size)

            self.windows[dname] = window
            screens = self.getScreens(dname, size[0], size[1])

            for sname in screens:
                size = ( screens[sname]['width'], screens[sname]['height'] )
                logger.info('Creating frame for screen "%s" in window %s with size %s x %s offset %s %s', sname, title, size[0], size[1], topleft[0], topleft[1])
                frame = ScreenFrame(self, window, -1, screens[sname]['topleft'], size, style=wx.NO_BORDER)
                self.frames[sname] = frame
            self.screens.update(screens)
            window.Show()
        self.worker = WorkerThread(self, self.views, self.screens, self.cond)
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
            displays["{0}".format(i)] = { 'top': topleft[0], 'left': topleft[1], 'width': size[0], 'height': size[1] }

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
        # logger.debug(event.data)
        action = event.data['action']
        logger.debug("Received command %s", action)
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


    def GetScreenPosition(self, wname):
        if wname in self.windows:
            return self.windows[wname].GetScreenPosition()
        else: return None


    def Close(self):
        self.worker.abort()
        self.worker.join()
        for f in self.windows.values():
            f.Close()

