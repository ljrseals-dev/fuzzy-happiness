"""INT8 entropy calibrator. Feed 200-500 *representative real* depth frames (.npy/.png);
falls back to synthetic frames if no directory is given."""
import glob
import os
import numpy as np
import tensorrt as trt
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
from data import load_depth
from preprocess import preprocess_depth
from synth import sample_frame


class DepthCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_dir=None, n=300, bs=8, cache="calib.cache"):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npy")) +
                            glob.glob(os.path.join(data_dir, "*.png"))) if data_dir else []
        self.n, self.bs, self.cache, self.i = n, bs, cache, 0
        self.rng = np.random.default_rng(0)
        self.dev = cuda.mem_alloc(bs * 256 * 256 * 4)

    def get_batch_size(self):
        return self.bs

    def get_batch(self, names):
        if self.i >= self.n:
            return None
        frames = [preprocess_depth(load_depth(self.files[(self.i + k) % len(self.files)])) if self.files
                  else preprocess_depth(sample_frame(self.rng)[0]) for k in range(self.bs)]
        cuda.memcpy_htod(self.dev, np.ascontiguousarray(np.stack(frames), np.float32))
        self.i += self.bs
        return [int(self.dev)]

    def read_calibration_cache(self):
        return open(self.cache, "rb").read() if os.path.exists(self.cache) else None

    def write_calibration_cache(self, c):
        open(self.cache, "wb").write(c)
