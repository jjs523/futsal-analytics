"""Video -> tracks pipeline.

    per camera, per segment : detect players on every `stride`-th frame -> foot pixels      (detect.py)
    per match, at the end   : pixels -> pitch (homography) -> sync -> fuse cameras -> track on the pitch
                              -> stats -> tracks.json                                         (run.py, pitch_tracker.py)

Detections are stored in pixels so calibration can be (re)done after upload.

Box-level tracker (v2; run.build_tracks(ids="v2"), the default whenever every camera has ReID embeddings;
ids="v2gta" swaps the closed-set linker for the open-set one):
    boxes.py      Detections -> CamBoxes per camera on the common frame grid; teams from upper-body colour
    align.py      cross-camera self-alignment from players seen by both cameras
    reid.py       ReID embeddings for detect_video(embedder=...)
    tracklets.py  pure per-camera tracklets, then cross-view pairing
    identity.py   team vote, covariances, cannot-links, clean-crop appearance, gap filling for the linkers
    link_closed.py closed-set identities (5 + 1 slots per team, one MILP; windowed and chained for a full match)
    sources.py    ids="v2gta": long pure fused tracklets (per-camera re-link around occlusions, pairing,
                  fused re-link, short pieces absorbed)
    link_gta.py   ids="v2gta": open-set agglomerative linking of those tracklets (windowed for a full match)
    refine.py     covariance-aware RTS smoothing of the final positions (analytics only)
"""
