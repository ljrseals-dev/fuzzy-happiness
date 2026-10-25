"""SMPL mesh depth renderer via PyBullet (CPU-compatible).

Public entry point:
    render_frame(rng, intr, smpl_model, motion_sample=None)
        -> (depth_m, joints_xyz, joints_uv)

Requires:
  - smplx  (pip install smplx)
  - pybullet (pip install pybullet)
  - SMPL neutral model files under SMPL_MODEL_PATH (or --smpl-model-path)
  - Optional: AMASS motion via human_body_prior (if motion_sample is provided)

Without a real AMASS sequence, pose parameters are sampled randomly within a
plausible range (documented random pose, not mocap).
"""
from __future__ import annotations
import os
import tempfile
from pathlib import Path
import numpy as np

from common import NUM_JOINTS, SMPL24_TO_17, Intrinsics


def _check_deps():
    missing = []
    try:
        import smplx  # noqa: F401
    except ImportError:
        missing.append("smplx (pip install smplx)")
    try:
        import pybullet  # noqa: F401
    except ImportError:
        missing.append("pybullet (pip install pybullet)")
    return missing


def load_smpl(model_path: str | None = None):
    """Load SMPL neutral model. Raises RuntimeError with a clear message on failure."""
    import smplx
    import torch
    path = model_path or os.environ.get("SMPL_MODEL_PATH")
    if not path:
        raise RuntimeError(
            "SMPL_MODEL_PATH is not set and --smpl-model-path was not given.\n"
            "Download the SMPL neutral model from https://smpl.is.tue.mpg.de/ "
            "and set SMPL_MODEL_PATH to the directory containing SMPL_NEUTRAL.pkl "
            "(or the smplx-layout models/smpl/ folder)."
        )
    path = str(path)
    if not os.path.isdir(path) and not os.path.isfile(path):
        raise RuntimeError(f"SMPL model path does not exist: {path}")
    model = smplx.create(
        model_path=path,
        model_type="smpl",
        gender="neutral",
        use_face_contour=False,
        batch_size=1,
    )
    model.eval()
    return model


def _sample_pose(rng: np.random.Generator):
    """Random but plausible SMPL pose (radians). Not real mocap."""
    # global_orient: small yaw/pitch
    global_orient = rng.normal(0, 0.3, size=(1, 3)).astype(np.float32)
    # body_pose: 23 joints * 3
    body_pose = rng.normal(0, 0.25, size=(1, 69)).astype(np.float32)
    # damp extreme joint angles a bit
    body_pose = np.clip(body_pose, -1.2, 1.2)
    betas = rng.normal(0, 0.8, size=(1, 10)).astype(np.float32)
    return global_orient, body_pose, betas


def place_subject(verts, joints24, rng, intr: Intrinsics):
    """Translate a camera-frame (y-down) SMPL body so it stands on the floor in front of a sensor mounted
    0.9-1.5 m above the floor and fits in the field of view. Returns (verts, joints24)."""
    z_off = float(rng.uniform(2.0, 3.2))
    cam_h = float(rng.uniform(0.9, 1.5))                     # sensor height above floor
    feet_y = float(np.percentile(verts[:, 1], 98))           # lowest body points (y grows downward)
    t = np.array([float(rng.uniform(-0.3, 0.3)), cam_h - feet_y, z_off], np.float32)
    verts, joints24 = verts + t, joints24 + t
    # keep the whole body vertically inside the image (tall subject close to the sensor)
    v_top = intr.cy + intr.fy * float(verts[:, 1].min()) / max(float(verts[:, 2].min()), 0.3)
    if v_top < 5.0:
        push = float(np.clip(intr.fy * (-verts[:, 1].min()) / (intr.cy - 5.0) - verts[:, 2].min(), 0.0, 2.0))
        verts = verts + np.array([0, 0, push], np.float32)
        joints24 = joints24 + np.array([0, 0, push], np.float32)
    return verts, joints24


def head_top_point(verts, head_joint, radius=0.18):
    """Highest mesh point (smallest y) within `radius` m horizontally of the head joint. A raised hand is
    excluded unless it is within `radius` of the head."""
    d = np.hypot(verts[:, 0] - head_joint[0], verts[:, 2] - head_joint[2])
    cand = verts[d < radius]
    return cand[np.argmin(cand[:, 1])] if len(cand) else head_joint


def render_frame(rng, intr: Intrinsics, smpl_model, motion_sample=None):
    """Render one depth frame + 17 joints from an SMPL mesh.

    Parameters
    ----------
    rng : np.random.Generator
    intr : common.Intrinsics
    smpl_model : smplx SMPL model (from load_smpl)
    motion_sample : optional dict with keys global_orient, body_pose, betas (torch/np)
                    If None, a random plausible pose is sampled.

    Returns
    -------
    depth_m : float32 [H,W] meters, 0 = invalid
    joints_xyz : float32 [17,3] camera frame
    joints_uv : float32 [17,2] raw pixel coordinates
    """
    import torch
    import pybullet as p
    import pybullet_data

    if motion_sample is None:
        go, bp, betas = _sample_pose(rng)
        go_t = torch.from_numpy(go)
        bp_t = torch.from_numpy(bp)
        betas_t = torch.from_numpy(betas)
    else:
        go_t = torch.as_tensor(motion_sample["global_orient"]).float().view(1, 3)
        bp_t = torch.as_tensor(motion_sample["body_pose"]).float().view(1, 69)
        betas_t = torch.as_tensor(motion_sample.get("betas", np.zeros((1, 10)))).float().view(1, 10)

    with torch.no_grad():
        out = smpl_model(
            global_orient=go_t,
            body_pose=bp_t,
            betas=betas_t,
        )
    verts = out.vertices[0].cpu().numpy().astype(np.float32)   # [V,3]
    joints24 = out.joints[0, :24].cpu().numpy().astype(np.float32)

    # Place subject ~2.5 m in front of camera, standing on ground
    # SMPL is Y-up; our camera looks along +Z with Y down (Kinect convention).
    # Rotate SMPL (Y-up) -> camera (Y-down, Z-forward): 180 deg about X then face camera.
    R = np.array([[1, 0, 0],
                  [0, -1, 0],
                  [0, 0, -1]], np.float32)
    verts = verts @ R.T
    joints24 = joints24 @ R.T
    verts, joints24 = place_subject(verts, joints24, rng, intr)

    # Map 24 -> 17
    joints_xyz = np.zeros((NUM_JOINTS, 3), np.float32)
    for i, sidx in enumerate(SMPL24_TO_17):
        joints_xyz[i] = joints24[sidx]
    joints_xyz[0] = head_top_point(verts, joints24[15])      # head_top: top of the skull, not the head joint

    # PyBullet scene
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.loadURDF("plane.urdf")

    # Write temporary OBJ for the mesh
    with tempfile.TemporaryDirectory() as td:
        obj_path = os.path.join(td, "body.obj")
        with open(obj_path, "w") as f:
            for v in verts:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            # SMPL faces from the model
            faces = smpl_model.faces
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        col_id = p.createCollisionShape(p.GEOM_MESH, fileName=obj_path)
        vis_id = p.createVisualShape(p.GEOM_MESH, fileName=obj_path)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id,
                          baseVisualShapeIndex=vis_id)

        # Camera from intrinsics
        w, h = intr.width, intr.height
        fov_y = 2.0 * np.degrees(np.arctan(h / (2.0 * intr.fy)))
        near, far = 0.1, 6.0
        view = p.computeViewMatrix(
            cameraEyePosition=[0, 0, 0],
            cameraTargetPosition=[0, 0, 1],
            cameraUpVector=[0, -1, 0],
        )
        proj = p.computeProjectionMatrixFOV(
            fov=fov_y, aspect=w / float(h), nearVal=near, farVal=far
        )
        _, _, _, depth_buf, _ = p.getCameraImage(
            width=w, height=h,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER,
        )
    p.disconnect()

    # Convert Bullet depth buffer to meters
    # depth_buf is [0,1] nonlinear; recover Z:
    depth_buf = np.reshape(depth_buf, (h, w)).astype(np.float64)
    z = far * near / (far - (far - near) * depth_buf)
    z = z.astype(np.float32)
    # Invalidate background (far plane)
    z[depth_buf >= 0.999] = 0.0
    z[(z < 0.3) | (z > 5.5)] = 0.0

    # Project joints
    joints_uv = np.stack([
        intr.fx * joints_xyz[:, 0] / joints_xyz[:, 2] + intr.cx,
        intr.fy * joints_xyz[:, 1] / joints_xyz[:, 2] + intr.cy,
    ], 1).astype(np.float32)

    return z, joints_xyz, joints_uv
