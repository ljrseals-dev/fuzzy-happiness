"""TensorRT runner (TRT 8.6 binding API + pycuda). Create AND use it in the same thread:
CUDA contexts are thread-bound."""
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda


class TRTRunner:
    def __init__(self, engine_path, device=0):
        cuda.init()
        self.ctx = cuda.Device(device).make_context()
        rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = rt.deserialize_cuda_engine(open(engine_path, "rb").read())
        self.ec = self.engine.create_execution_context()
        self.ec.set_binding_shape(0, (1, 1, 256, 256))
        self.h_in = cuda.pagelocked_empty((1, 1, 256, 256), np.float32)
        self.h_out = cuda.pagelocked_empty(tuple(self.ec.get_binding_shape(1)), np.float32)
        self.d_in, self.d_out = cuda.mem_alloc(self.h_in.nbytes), cuda.mem_alloc(self.h_out.nbytes)
        self.stream = cuda.Stream()

    def infer(self, x):                       # x: [1,256,256] or [1,1,256,256] float32
        self.h_in[...] = x.reshape(1, 1, 256, 256)
        cuda.memcpy_htod_async(self.d_in, self.h_in, self.stream)
        self.ec.execute_async_v2([int(self.d_in), int(self.d_out)], self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_out, self.d_out, self.stream)
        self.stream.synchronize()
        return self.h_out[0].copy()           # [17,64,64]

    def close(self):
        self.ctx.pop()
