# heavily inspired from https://stackoverflow.com/questions/43665208/how-to-get-the-latest-frame-from-capture-device-camera-in-opencv

from jetcam.csi_camera import CSICamera
import queue
import threading
import time

class VideoCapture:
    """A bufferless video streaming class. 
    
    Wraps Nvidia's jetcam CSICamera object for a CSI Camera connected to the Jetson Orin Nano.

    Attributes:
        running_instances (int): number of instances reading from this object in parallel

    """

    def __init__(self, **kwargs):
        """Create a VideoCapture object

        Args:
            **kwargs: Arguments passed to jetcam.csi_camera.CSICamera
        """
        self.cap = CSICamera(**kwargs)
        self.q = queue.Queue()
        #self.t.start()
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True

        self.running_instances = 0

    def start_stream(self):
        """Start the parallel thread reading the camera to empty the buffer queue

        If the thread is already running, it just increases the number of running_instances (tracker, no other implications)
        """
        self.running_instances += 1
        if not self.t.is_alive():
            self.t = threading.Thread(target=self._reader)
            self.t.daemon = True
            self.t.start()
        print("VideoCapture: currently serving", self.running_instances)
        #print("VideoCamera: started _reader thread")

    def stop_stream(self, force=False):
        """Stops the stream

        This halts the reading thread.

        Args:
            force (bool, optional): soft stop would only decrease the number of running_instances by one (if that is zero, the thread terminates) or just terminate the thread.
        """
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
        """Read frames as soon as they are available, keeping only most recent one

        This runs in a parallel thread
        """
        while self.running_instances > 0:#not self.stop:
            frame = self.cap.read()
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)
        print("VideoCapture: exiting _reader thread")

    def read(self):
        """Get the most current frame
        """
        return self.q.get()