"""pipeline.py — 4-stage threaded pipeline with per-stage latency profiler.
  python pipeline.py --source synth --engine student_fp16.plan --seconds 20"""
import argparse
import collections
import queue
import threading
import time
import numpy as np
from preprocess import preprocess_depth
from postprocess import decode_skeleton
from trt_runner import TRTRunner


class LatestQueue:
    """Bounded queue that drops the oldest item when full (keeps latency bounded)."""
    def __init__(self, n=2):
        self.q = queue.Queue(n)

    def put(self, x):
        while True:
            try:
                self.q.put_nowait(x)
                return
            except queue.Full:
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass

    def get(self, timeout=1.0):
        return self.q.get(timeout=timeout)


class Profiler:
    def __init__(self):
        self.d = collections.defaultdict(lambda: collections.deque(maxlen=300))
        self.lock = threading.Lock()

    def rec(self, k, ms):
        with self.lock:
            self.d[k].append(ms)

    def report(self):
        with self.lock:
            return {k: f"mean {np.mean(v):.2f} ms  p95 {np.percentile(v, 95):.2f} ms" for k, v in self.d.items()}


class PosePipeline:
    def __init__(self, source, engine, on_skeleton=None):
        self.src, self.engine, self.cb = source, engine, on_skeleton or (lambda s: None)
        self.q1, self.q2, self.q3 = LatestQueue(), LatestQueue(), LatestQueue()
        self.run, self.prof, self.threads, self.done = True, Profiler(), [], 0

    def _capture(self):
        i = 0
        while self.run:
            t = time.perf_counter()
            d = self.src.read()
            if d is None:
                continue
            self.prof.rec("capture", (time.perf_counter() - t) * 1e3)
            self.q1.put((i, d, t))
            i += 1

    def _pre(self):
        while self.run:
            try:
                i, d, t0 = self.q1.get()
            except queue.Empty:
                continue
            t = time.perf_counter()
            x = preprocess_depth(d)
            self.prof.rec("preprocess", (time.perf_counter() - t) * 1e3)
            self.q2.put((i, d, x, t0))

    def _infer(self):
        r = TRTRunner(self.engine)      # created in this thread: CUDA contexts are thread-bound
        try:
            while self.run:
                try:
                    i, d, x, t0 = self.q2.get()
                except queue.Empty:
                    continue
                t = time.perf_counter()
                hm = r.infer(x)
                self.prof.rec("inference", (time.perf_counter() - t) * 1e3)
                self.q3.put((i, d, hm, t0))
        finally:
            r.close()

    def _post(self):
        while self.run:
            try:
                i, d, hm, t0 = self.q3.get()
            except queue.Empty:
                continue
            t = time.perf_counter()
            sk = decode_skeleton(hm, d, self.src.intr, i)
            self.prof.rec("postprocess", (time.perf_counter() - t) * 1e3)
            self.prof.rec("end_to_end", (time.perf_counter() - t0) * 1e3)
            self.done += 1
            self.cb(sk)

    def start(self):
        for f in (self._capture, self._pre, self._infer, self._post):
            th = threading.Thread(target=f, daemon=True)
            th.start()
            self.threads.append(th)

    def stop(self):
        self.run = False
        for th in self.threads:
            th.join(timeout=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synth", "v2", "azure"], default="synth")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--seconds", type=int, default=20)
    a = ap.parse_args()
    import sources
    src = {"synth": sources.SyntheticSource, "v2": sources.FreenectV2Source,
           "azure": sources.AzureKinectSource}[a.source]()
    p = PosePipeline(src, a.engine)
    p.start()
    t0 = time.time()
    while time.time() - t0 < a.seconds:
        time.sleep(2)
        print(f"FPS {p.done / (time.time() - t0):.1f}", p.prof.report(), flush=True)
    p.stop()
    src.close()
