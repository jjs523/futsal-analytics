"""Video -> tracks pipeline.

    per camera, per segment : detect players on every `stride`-th frame -> foot pixels      (detect.py)
    per match, at the end   : pixels -> pitch (homography) -> sync -> fuse cameras -> track on the pitch
                              -> stats -> tracks.json                                         (run.py, pitch_tracker.py)

Detections are stored in pixels so calibration can be (re)done after upload.
"""
