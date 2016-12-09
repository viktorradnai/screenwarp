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
            # focus to infinity, lock focus and exposure
            requests.get(self.url + '/settings/focusmode?set=auto')
            requests.get(self.url + '/settings/focus?set=on')
            requests.get(self.url + '/settings/exposure_lock?set=on')
            requests.get(self.url + '/settings/whitebalance_lock?set=on')
            res = requests.get(self.url + '/photo.jpg')
            requests.get(self.url + '/settings/focus?set=on')
            requests.get(self.url + '/settings/focus?set=off')

        else:
            time.sleep(1) # Allow display to update
            res = requests.get(self.url + '/photo.jpg')

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


