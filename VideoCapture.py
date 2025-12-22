# heavily inspired from https://stackoverflow.com/questions/43665208/how-to-get-the-latest-frame-from-capture-device-camera-in-opencv

from jetcam.csi_camera import CSICamera
import queue
import threading
import time

class VideoCapture:
    """A bufferless video streaming object. Wraps Nvidia's jetcam CSICamera object for a CSI Camera connected to the Jetson Orin Nano

    """

    def __init__(self, **kwargs):
        self.cap = CSICamera(**kwargs)
        self.q = queue.Queue()
        #self.t.start()
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True

        self.running_instances = 0

    def start_stream(self):
        self.running_instances += 1
        if not self.t.is_alive():
            self.t = threading.Thread(target=self._reader)
            self.t.daemon = True
            self.t.start()
        print("VideoCapture: currently serving", self.running_instances)
        #print("VideoCamera: started _reader thread")

    def stop_stream(self, force=False):
        if force:
            self.running_instances = 0
        else:
            self.running_instances = max(self.running_instances - 1, 0)
        print("VideoCapture: updated running instances to", self.running_instances)
        time.sleep(0.1)

        #self.t = threading.Thread(target=self._reader)
        #self.t.daemon = True

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while self.running_instances > 0:#not self.stop:
            frame = self.cap.read()
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)
        #print("VideoCamera: exiting _reader thread")

    def read(self):
        return self.q.get()