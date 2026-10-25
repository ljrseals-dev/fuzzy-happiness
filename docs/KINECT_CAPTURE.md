# Kinect capture protocol

Run only after Gate 6 passes. Target session: ~30 minutes of capture.

## Layout produced by `scripts/capture_session.py`

```
capture_YYYYMMDD_HHMMSS/
├── calib/          # INT8 calibration (≥500 frames)
├── unlabeled/      # MAE fine-tune (≥1000 frames)
├── posed/          # labeling seed (≥30 held poses)
├── meta.json       # device serial, firmware, intrinsics, depth mode
├── session.log
└── README.txt
```

## Protocol

| Set | Duration | Content | Purpose |
|-----|----------|---------|---------|
| calib | 3 min | 1 person, varied poses & distances (0.5–4.5 m); some empty; some clutter | INT8 calibration |
| unlabeled | 5 min | continuous natural movement, single person | MAE fine-tune |
| posed | 5 min | held poses: T-pose, arms-out, squat, arms-up, side T-pose, sitting; 5 s × 3 reps, front + side | semi-auto labeling seed |

Reject frames where the live pipeline fails badly; log the reason in `session.log`.

## Minimum viable counts

- calib ≥ 500 frames  
- unlabeled ≥ 1000 frames  
- posed ≥ 30 distinct pose samples  

If short, schedule another session.
