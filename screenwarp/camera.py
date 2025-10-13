import time
import logging
import requests
import subprocess

logger = logging.getLogger(__name__)

class AndroidIPCamera:
    def __init__(self, url):
        self.url = url

    def acquireImage(self, imgname, focus=False):
        logger.debug("Using camera at %s", self.url)
        if focus:
            # take focused photo, lock focus and exposure
            requests.get(self.url + '/settings/focusmode?set=auto', timeout=2)
            requests.get(self.url + '/settings/focus?set=on', timeout=2)
            requests.get(self.url + '/settings/exposure_lock?set=on', timeout=2)
            requests.get(self.url + '/settings/whitebalance_lock?set=on', timeout=2)
            res = requests.get(self.url + '/photo.jpg', timeout=2)
            requests.get(self.url + '/settings/focus?set=off', timeout=2)

        else:
            time.sleep(0.5) # Allow display to update
            res = requests.get(self.url + '/photo.jpg', timeout=2)

        logger.debug(res)
        if not res:
            logger.error("Failed to get image from IP Webcam, no response received")
            return False
        if res.status_code != 200:
            logger.error("Failed to get image from IP Webcam, service returned status %s", res.status_code)
            return False

        path = 'cache/{0}.jpg'.format(imgname)
        with open(path, 'wb') as f:
            f.write(res.content)
        return res.content


class CommandLineCamera:
    def __init__(self, command):
        logger.debug("Using command %s to take photos", command)
        self.command = command

    def acquireImage(self, imgname, focus=False):
        path = 'cache/{0}.jpg'.format(imgname)
        command = self.command % (path)
        logger.info("Capturing image %s using command line %s", path, command)
        subprocess.call(command, shell=True)
        logger.debug("Loading image from %s", path)
        with open(path, 'rb') as f:
            return f.read()


class FileCacheCamera:
    def acquireImage(self, imgname, focus=False):
        path = 'cache/{0}.jpg'.format(imgname)
        logger.info("Loading image from %s", path)
        with open(path, 'rb') as f:
            return f.read()


