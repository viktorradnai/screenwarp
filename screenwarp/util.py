import os
import cv2
import errno
import logging
import cStringIO
import numpy as np

logger = logging.getLogger(__name__)

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


def factors(n):
    factors = set()
    for i in range(1, n//2 + 1):
        if n % i == 0: factors.add(i)
    return factors


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
