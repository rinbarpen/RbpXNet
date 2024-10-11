import cv2

class VideoFrameCapture:
    def __init__(self, url: str, **kwargs):
        self.url = url
        self.vcapture = cv2.VideoCapture()

    def __del__(self):
        self.vcapture.release()
        del self

    def capture(self, f):
        if not self.vcapture.isOpened():
            if not self.vcapture.open(self.url):
                raise ValueError(f"Cannot open {self.url} video stream")

        while True:
            r, frame = self.vcapture.read()
            if not r:
                yield None

            yield f(frame) if f else frame
