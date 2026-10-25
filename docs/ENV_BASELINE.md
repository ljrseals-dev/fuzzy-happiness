# Environment baseline

This file records the environment of the machine that last ran `scripts/capture_env.py`.
Compare your own `env_report.json` against this when debugging GPU gate failures.

**Important:** The baseline below is from a **CPU-only sandbox** used for scaffold development.
It is **not** a known-working GPU baseline. Replace this file after Phase A succeeds on a real GPU box.

```json
{
  "timestamp_utc": "2026-10-25T12:52:33.241450+00:00",
  "hostname": "hds-73herq74vd30",
  "platform": "Linux-6.12.8+-x86_64-with-glibc2.39",
  "python": "3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0]",
  "pip_list_hash": "7d1c2415a125ed79",
  "pip_package_count": 179,
  "torch": {
    "version": "2.12.0+cu130",
    "cuda_built": "13.0",
    "cuda_available": false,
    "device_count": 0
  },
  "onnxruntime": {
    "version": "1.26.0",
    "providers": [
      "AzureExecutionProvider",
      "CPUExecutionProvider"
    ]
  },
  "tensorrt": {
    "error": "No module named 'tensorrt'"
  },
  "opencv": {
    "error": "No module named 'cv2'"
  },
  "nvidia_smi": {
    "error": "[Errno 2] No such file or directory: 'nvidia-smi'"
  },
  "nvcc": {
    "error": "[Errno 2] No such file or directory: 'nvcc'"
  },
  "trtexec": {
    "error": "[Errno 2] No such file or directory: 'trtexec'"
  }
}
```
