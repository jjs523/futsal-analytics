"""Video -> tracks pipeline.

    per camera, per segment : detect players on every `stride`-th frame -> foot pixels      (detect.py)
    per match, at the end   : pixels -> pitch (homography) -> sync -> fuse cameras -> track on the pitch
                              -> stats -> tracks.json                                         (run.py, pitch_tracker.py)

Detections are stored in pixels so calibration can be (re)done after upload.

Box-level tracker (v2; meant to replace the foot-point fusion step, not wired into run.py yet):
    boxes.py      Detections -> CamBoxes per camera on the common frame grid; teams from upper-body colour
    align.py      cross-camera self-alignment from players seen by both cameras
    reid.py       ReID embeddings for detect_video(embedder=...)
    tracklets.py  pure per-camera tracklets, then cross-view pairing
    identity.py   team vote, covariances, cannot-links, clean-crop appearance, gap filling for the linkers
"""
