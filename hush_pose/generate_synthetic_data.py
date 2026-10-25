#!/usr/bin/env python3
"""Generate synthetic depth + pose labels.

--renderer stick  : always available (cv2 stick figure)
--renderer smpl   : requires smplx + pybullet + SMPL model files

Usage:
  python generate_synthetic_data.py --out data/synth --n 2000
  python generate_synthetic_data.py --out data/synth --n 5 --renderer smpl --require-smpl
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from common import KINECT_V2, JOINT_NAMES
from synth import sample_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--renderer", choices=["stick", "smpl"], default="stick")
    ap.add_argument("--require-smpl", action="store_true",
                    help="With --renderer smpl: exit 1 if SMPL deps/model missing (no silent fallback)")
    ap.add_argument("--smpl-model-path", default=None,
                    help="SMPL model directory (default: $SMPL_MODEL_PATH)")
    ap.add_argument("--amass-path", default=None,
                    help="NOT IMPLEMENTED (exits with an error); see smpl_renderer.render_frame(motion_sample=...)")
    a = ap.parse_args()
    if a.amass_path:
        sys.exit("--amass-path is not implemented yet: AMASS is Z-up SMPL-H and needs a verified frame conversion. "
                 "Pass your own motion_sample dicts to smpl_renderer.render_frame() instead.")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    render = None
    renderer_name = "stick"

    if a.renderer == "smpl":
        try:
            from smpl_renderer import _check_deps, load_smpl, render_frame
            missing = _check_deps()
            if missing:
                raise RuntimeError("Missing packages: " + ", ".join(missing))
            model = load_smpl(a.smpl_model_path)
            def render_smpl(rng):
                return render_frame(rng, KINECT_V2, model, motion_sample=None)
            render = render_smpl
            renderer_name = "smpl"
        except Exception as e:
            msg = (
                f"[generate_synthetic_data] SMPL renderer unavailable: {e}\n"
                "Required:\n"
                "  - pip install smplx pybullet\n"
                "  - SMPL neutral model from https://smpl.is.tue.mpg.de/\n"
                "  - export SMPL_MODEL_PATH=/path/to/smpl/models\n"
            )
            if a.require_smpl:
                print(msg, file=sys.stderr)
                sys.exit(1)
            print("WARNING: " + msg + "Falling back to stick renderer.", file=sys.stderr)
            render = None

    if render is None:
        def render_stick(rng):
            return sample_frame(rng)
        render = render_stick
        renderer_name = "stick"

    for i in range(a.n):
        depth, xyz, uv = render(rng)
        np.save(out / f"{i:06d}_depth.npy", depth.astype(np.float32))
        np.save(out / f"{i:06d}_xyz.npy", xyz.astype(np.float32))
        np.save(out / f"{i:06d}_uv.npy", uv.astype(np.float32))
        if (i + 1) % 100 == 0 or i == 0:
            print(f"wrote {i+1}/{a.n}  renderer={renderer_name}", flush=True)

    meta = {
        "n": a.n,
        "intrinsics": {
            "fx": KINECT_V2.fx, "fy": KINECT_V2.fy,
            "cx": KINECT_V2.cx, "cy": KINECT_V2.cy,
            "width": KINECT_V2.width, "height": KINECT_V2.height,
        },
        "joint_names": JOINT_NAMES,
        "renderer": renderer_name,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done. {a.n} frames written to {out} (renderer={renderer_name})")


if __name__ == "__main__":
    main()
