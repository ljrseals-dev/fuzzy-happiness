"""Depth sources. Each .read() returns float32 meters [H,W] (0 = invalid) or None."""
import time
import numpy as np
from common import Intrinsics


class FreenectV2Source:
    """Kinect v2 via libfreenect2 + pylibfreenect2 (build libfreenect2 with CUDA)."""
    def __init__(self):
        from pylibfreenect2 import Freenect2, SyncMultiFrameListener, FrameType
        try:
            from pylibfreenect2 import CudaPacketPipeline as Pipe
        except ImportError:
            from pylibfreenect2 import OpenGLPacketPipeline as Pipe
        self.fn = Freenect2()
        self.dev = self.fn.openDevice(self.fn.getDeviceSerialNumber(0), pipeline=Pipe())
        self.lst = SyncMultiFrameListener(FrameType.Depth)
        self.dev.setIrAndDepthFrameListener(self.lst)
        self.dev.start()
        p = self.dev.getIrCameraParams()
        self.intr = Intrinsics(p.fx, p.fy, p.cx, p.cy, 512, 424)

    def read(self):
        frames = self.lst.waitForNewFrame()
        d = frames["depth"].asarray().astype(np.float32) / 1000.0   # mm -> m
        self.lst.release(frames)
        return d

    def close(self):
        self.dev.stop()
        self.dev.close()


class AzureKinectSource:
    """Azure Kinect via pyk4a (needs the Azure Kinect Sensor SDK)."""
    def __init__(self):
        from pyk4a import PyK4A, Config, DepthMode, ColorResolution, CalibrationType
        self.k4a = PyK4A(Config(depth_mode=DepthMode.NFOV_UNBINNED,
                                color_resolution=ColorResolution.OFF))
        self.k4a.start()
        K = self.k4a.calibration.get_camera_matrix(CalibrationType.DEPTH)
        self.intr = Intrinsics(K[0, 0], K[1, 1], K[0, 2], K[1, 2], 640, 576)

    def read(self):
        cap = self.k4a.get_capture()
        return None if cap.depth is None else cap.depth.astype(np.float32) / 1000.0

    def close(self):
        self.k4a.stop()


class SyntheticSource:
    """Sensor-less source for throughput tests. fps=0 -> as fast as possible."""
    def __init__(self, fps=0, seed=0):
        from common import KINECT_V2
        from synth import sample_frame
        self.intr, self.fps, self.f = KINECT_V2, fps, sample_frame
        self.rng, self.t = np.random.default_rng(seed), time.time()

    def read(self):
        if self.fps:
            time.sleep(max(0, self.t + 1 / self.fps - time.time()))
            self.t = time.time()
        return self.f(self.rng)[0]

    def close(self):
        pass
