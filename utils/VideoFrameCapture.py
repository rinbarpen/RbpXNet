import cv2

class VideoFrameCapture:
    def __init__(self, filename: str, **kwargs):
        self.filename = filename
        self.vcapture = cv2.VideoCapture(filename)

    def __del__(self):
        self.vcapture.release()
        del self

    def capture(self, f):
        if not self.vcapture.isOpened():
            if not self.vcapture.open(self.filename):
                raise ValueError(f"Cannot open {self.filename} video stream")

        while True:
            r, frame = self.vcapture.read()
            if not r:
                yield None

            yield f(frame) if f else frame
