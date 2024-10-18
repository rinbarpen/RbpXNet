import cv2
from typing import Optional, Tuple
from enum import Enum
from queue import Queue

from utils.wrapper import thread
class VideoFrameCapture:
    def __init__(self, url: str, **kwargs):
        self.url = url
        self.vcapture = cv2.VideoCapture()

    def __del__(self):
        self.vcapture.release()
        del self

    @thread(name='VideoFrameCapture')
    def capture(self, f=None):
        if not self.vcapture.isOpened():
            if not self.vcapture.open(self.url):
                raise ValueError(f"Cannot open {self.url} video stream")

        while self.vcapture.isOpened():
            r, frame = self.vcapture.read()
            if not r:
                yield None

            yield f(frame) if f else frame

        # raise ValueError(f'video stream had been closed')

    def reset(self, url: Optional[str]=None):
        if url:
            self.vcapture.release()
            if self.vcapture.open(url):
                self.url = url
        else:
            self.vcapture.open(self.url)

    def release(self):
        self.vcapture.release()
