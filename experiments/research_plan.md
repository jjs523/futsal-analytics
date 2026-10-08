# Tracker plan for the futsal-analytics cached data (10 Hz, 6 minutes, 2 cameras, no labels)

The plan is checked against the existing harness:
- `C:\Users\user\dev\futsal-analytics\experiments\harness.py` provides `Data`/`CamData`, `metrics()` and `audit_sheet()`.
- `C:\Users\user\dev\futsal-analytics\experiments\trackers.py` registers the `baseline` and `baseline_motion` variants.
- `C:\Users\user\dev\futsal-analytics\experiments\cache\cam{1,2}.npz` holds `xyxy`, `conf`, `reid` (512-d), `color` (30-d) and `fid`/`t`, with 3602 grid frames.
- `CamData` already provides `team` ('Y', 'N' or ''), `yellow`, `xy`, `sigma` and `k`.

Each variant below is one `@register` function. Variants 1 to 4 form a cumulative ladder, so each step's gain can be attributed. Variants 5 to 7 are alternatives or later phases.

---

## Shared building blocks (write once, reused by every variant)

**B1. Tracklet team vote**
- Weight each box label by box height, using only boxes with h ≥ 40 px and conf ≥ 0.4.
- Label the tracklet `Y` if the weighted yellow share is ≥ 0.7, `N` if ≤ 0.3, otherwise `U` (unknown).
- A `U` tracklet can link to either team, but pays a cost penalty.
- Goalkeepers: if a tracklet's mean colour histogram is an outlier for both teams, and it spends more than 70% of its time within 6 m of a goal line, label it `GK_A` or `GK_B` and treat it as its own one-person class.
- Split a tracklet when the team vote flips persistently: at least 15 consecutive labelled samples (1.5 s) on each side, each with a share of 0.8 or more.

**B2. Anisotropic pitch covariance**
- Undistort the foot point, then compute the Jacobian J of H by finite differences.
- R = J·diag((0.03·w)², (0.05·h)²)·Jᵀ + (0.15 m)²·I, where the last term absorbs homography error.
- This replaces the current isotropic `sigma` (DETECTOR_JITTER_PX times metres per pixel).
- Fuse the two cameras' positions by inverse-covariance weighting.
- Gate with Mahalanobis distance d² < 9.21 (χ², 2 degrees of freedom, 99%).

**B3. Reachability gate between tracklet end A and start B**
- ‖p_B − p_A‖ ≤ v_max·Δt + slack + 3·√λ_max(R_A + R_B).
- Try v_max ∈ {7, 8, 9} m/s and slack ∈ {0.5, 1.0, 1.5} m.
- If Δt > 8 s, a link also requires appearance distance below a stricter threshold.

**B4. Robust tracklet appearance**
- Use only "clean" crops: h ≥ 60 px, conf ≥ 0.5, IoU < 0.1 with any other box in the same camera, at least 2% of frame width from the image border, and not in an ambiguous frame.
- For a fused tracklet, take crops from the camera where the box is larger.
- Store the L2-normalised mean plus up to 10 medoids.
- d_app = min(1 − cos(means), 20th percentile of pairwise medoid distances).

**B5. Hard cannot-links**
- Temporal overlap of 2 or more samples, in either camera.
- Team mismatch (Y vs N; U is exempt).
- Failing the B3 reachability gate.

**B6. Gap filling, after identities are fixed**
- Gaps up to 3 s: cubic Hermite interpolation using the endpoint velocities, then Savitzky-Golay smoothing (window 7, order 2).
- Longer gaps: mark as "occluded, identity known". Do not interpolate.

---

## Ranked variants

The rank is by expected final quality. The build order is 1 → 2 → 3 → 4 → 5, then 6 and 7.

### Rank 1: V4 "pertrack_xview_closedset_ssl" (target pipeline)

**Pipeline**
1. Per-camera image-space tracklets, built with the custom conservative builder from V3.
2. Purity splitting:
   - Same-team proximity cut: two same-team boxes whose fused positions come within 1.2 m, or whose same-camera IoU exceeds 0.3, cut both tracklets for that window.
   - DBSCAN splitter: cosine metric on clean crops of tracklets longer than 5 s; eps 0.5–0.6, min_samples 5, at most 3 clusters; outliers go to the nearest cluster.
   - Team-flip split (B1).
3. Cross-view tracklet pairing (from V3), plus a cut wherever a tracklet's partner in the other camera changes.
4. Per-match self-supervised embedding head trained on the cached 512-d OSNet features, so no video decoding is needed:
   - MLP 512 → 256 → 128 with InfoNCE or batch-hard triplet loss.
   - Positives: clean crops of the same pure fused tracklet at least 1 s apart, including the cross-view partner's crops.
   - Negatives: crops of tracklets that coexist in time, with same-team coexisting tracklets as hard negatives (sampling weight 3x).
   - 20–50 epochs; a few minutes on the GPU.
5. Closed-set identity assignment per team (from V2) using the new embedding.
6. Gap filling (B6).

**Parameters to try:** temperature 0.07 or 0.1; head size 64 or 128; whether to include cross-view positives; DBSCAN eps 0.5 or 0.6.

**Failure modes**
- If an impure tracklet survives step 2, the training labels contain noise, and the head learns the swap. Mitigation: train only on tracklets of 2 s or more that pass the DBSCAN single-cluster test.
- The head overfits to pose or camera rather than identity. Mitigation: hold out by time when measuring the AUC proxy.
- Far-side cam2-only tracklets have no clean crops. They are linked by motion and team only.

**Why it may win:** it combines the structural fix (fuse tracklets, not frames), purity before linking, the known-N constraint, and the only cue that can separate teammates in the same bibs (shorts, socks, build) learned for this specific match. This is the Kalisteo, idtracker.ai and LMGP recipe adapted to two calibrated views.

### Rank 2: V3 "pertrack_xview_closedset" (structural change, no learned head)

**Pipeline**
1. **Per-camera tracklets at 10 Hz in image space (custom, about 150 lines)**
   - Cost = min(gated appearance, EIoU distance), Deep-EIoU style.
   - Expansion E stepped over {0.7, 0.85, 1.0} as successive Hungarian passes.
   - Appearance gate: 1 − cos < 0.3 on the raw OSNet feature.
   - An extra pitch-plane veto: the Mahalanobis distance from the last position, with variance grown by (v_max·Δt)², must be under 9.21.
   - Cut rules:
     - another box in the same camera overlaps the track's box with IoU > 0.5;
     - the best-vs-second-best cost margin is below 0.1;
     - there is a gap of more than 3 samples.
   - Do not use a Kalman filter. At 10 Hz, constant velocity overshoots on direction changes.
2. **Cross-view pairing**
   - Candidate pairs: cam1/cam2 tracklets with at least 10 shared samples.
   - Cost: median Mahalanobis d² over shared samples, with fused covariance R1 + R2.
   - Accept if median d² < 9.21, at least 70% of shared samples are inside the gate, teams are compatible, and the pair is mutual best (Hungarian over each overlap window).
   - Unpaired tracklets stay single-view.
3. **Closed-set assignment per team (from V2)** on the fused or single-view tracklets.
4. **Gap filling (B6).**

**Parameters to try:** EIoU schedule; IoU cut threshold 0.4 or 0.5; minimum pairing overlap 1 s or 2 s; gate fraction 0.6 or 0.8.

**Failure modes**
- Along the camera diagonal, the two cameras' error ellipses are aligned, so two same-team players 1–2 m apart can both pass the pairing gate. Mitigation: prefer the pairing with the lowest summed cost; leave ties unpaired.
- Lens distortion at the frame edges biases pairings. Mitigation: undistort in B2, and give edge boxes 1.5x the covariance.
- At 10 Hz the image-space IoU of sprinting players collapses. Expansion and appearance partly compensate; V7 fixes it properly.

**Why it may win:** in this pipeline, a fusion error in one frame cannot create an identity swap. In each camera image, crossing players stay apart sideways, and cross-view evidence is averaged over 10 or more samples. LMGP's ablation measured roughly 19 IDF1 points from the geometric cross-view pre-clustering step alone.

### Rank 3: V2 "fused_closedset" (cheapest test of the known-N constraint)

**Pipeline**
1. Keep the current `fused_frames` and `pitch_tracker.tracklets`, with `ambiguity` lowered to 0.3 for purer tracklets.
2. Add the B1 team-flip split.
3. Replace `associate` with a closed-set assignment, solved separately for team Y and team N:
   - **Identity slots:** K = 5 slots per team, plus 1 high-cost "spare" slot for substitutes or noise. Tracklets shorter than 1 s are left out of the solve and attached at the end by nearest reachable neighbour.
   - **Prototype initialisation:** use the longest window in which 5 same-team tracklets of 2 s or more coexist. Those 5 are distinct people, which removes label symmetry.
   - **MILP with scipy.optimize.milp (HiGHS):**
     - Binary variables x[t,k].
     - Each tracklet gets at most one slot: Σ_k x[t,k] ≤ 1.
     - Two tracklets that overlap in time cannot share a slot: x[t,k] + x[u,k] ≤ 1.
     - Each linked pair must pass the B3 gate. Implement this either as a successor-chain formulation, or as pairwise exclusion for unreachable pairs with |Δt| < 10 s.
     - Objective: minimise Σ len_t·(d_app(t, proto_k) − λ_cov). The coverage reward λ_cov ∈ {0.3, 0.5}.
   - **EM refinement:** re-estimate the prototypes from the assigned tracklets, 3–5 rounds.
   - **Alternative solver, for comparison:** `networkx.min_cost_flow` per team with flow fixed to 5. Each tracklet is an in/out node pair with capacity 1 and a −len reward. Link-edge cost = motion Mahalanobis + w·d_app. Source and sink edges are allowed only at the clip edges or in the bench zone.
4. Gap filling (B6).

**Parameters to try:** v_max and slack (B3); λ_cov; spare-slot cost; with and without the team constraint (this isolates its gain).

**Failure modes**
- Fusion errors upstream are not fixed, so impure tracklets get one label and the error spreads.
- The 5-coexisting initial window may not exist if detection drops a player. Fallback: seed with the 5 longest tracklets that are pairwise non-overlapping-compatible.
- Substitutions: if a sixth person shows up, the spare slot absorbs them. Watch whether the spare-slot assignment is unstable.

**Why it may win:** the output can never exceed 5 + 5 concurrent IDs, and it has no impossible teleports. Fragments become label errors on short tracklets instead of new IDs.

### Rank 4: V1 "fused_gta_constrained" (open-set; isolates the linker)

**Pipeline**
1. The current fused tracklets.
2. GTA-style connector: average-linkage agglomerative clustering on d_app (B4, raw OSNet instead of colour histograms), under the B5 cannot-links and the B3 metre gate.
3. Stop when the closest pair is above merge_thr ∈ {0.3, 0.4, 0.5}.
4. Optionally run the DBSCAN splitter first.

**Failure modes**
- A distance threshold cannot hit exactly 10 IDs.
- Teammates are entangled in raw OSNet space, so a low threshold under-merges and a high one merges teammates.

**Why run it:** it is a quick A/B showing how much the colour histograms and the missing team constraint cost the current `associate`. Keep it as V2's pre-merge stage, using only d < 0.25.

### Rank 5: V5 "boxmot_percam_xview" (off-the-shelf tracklet source for the V3/V4 back end)

**Pipeline**
1. Per camera, run boxmot 12 `HybridSort` (ReID variant) and `DeepOcSort` as the tracklet builder, with:
   - w_association_emb 1.25, aw_param 1.0, alpha_fixed_emb 0.95;
   - det_thresh 0.4, max_age 20 (2 s at 10 Hz);
   - the cached embeddings passed in through `embs=` and a dummy zero image, so no video is decoded.
2. Then the V3/V4 back end.

**Gotchas**
- boxmot frame-based defaults assume 30 fps. Rescale max_age, min_hits and similar settings by 1/3.
- BotSort and StrongSort always run CMC. Stub `tracker.cmc.apply` so it returns an identity 2x3 warp; `cmc_method=None` crashes.
- Verify that `embs=` is honoured, so the ReID model is not loaded. This is a known-uncertain point.

**Failure modes:** the Kalman filter overshoots on cuts at 10 Hz, and these trackers make long, impure tracks rather than cutting when unsure. Lower the match thresholds and rely on the V4 splitter.

**Why run it:** it shows whether the custom builder beats tuned library trackers. If the boxmot output is comparable, keep it for maintainability and the eventual phone port.

### Rank 6: V6 "v4_finetuned_osnet" (heavier version of V4's embedding step)

**Pipeline**
1. Replace the MLP head with full fine-tuning of `osnet_ain_x1_0_msmt17` on crops re-read from the video, using the same positives and negatives as V4.
2. Settings: AdamW, lr 1e-4, 10 epochs, batch P = 16 tracklets × K = 4 crops, crops ≥ 50 px tall.
3. Also run an A/B of input weights: MSMT17 OSNet, the SportsMOT OSNet (`sports_model.pth.tar-60`, if obtainable), and `clip_market1501`.
4. All judged with the AUC proxy described under metrics.

**Cost:** the crops must be decoded from video (about 44k crop boxes per camera). That is minutes of work.

**Failure modes:** overfitting to 10 identities, and leakage between training and evaluation tracklets. Use the time-split protocol.

**Why it may win over V4:** the MLP head can only re-weight the frozen 512-d feature. Full fine-tuning can learn shoes and socks that the pedestrian-trained feature discards.

### Rank 7: V7 "v4_at_30hz_eiou" (phase 2: needs a cache rebuild)

**Pipeline**
1. Rebuild the cache at 30 Hz: every cam2 frame and every second cam1 frame, about 10.8k frames per camera for 6 minutes, with detection plus ReID.
2. Track per camera with Deep-EIoU-style association: E 0.7 → 0.8 → 0.9, appearance gate 0.25, EIoU gate 0.5, buffer 60 frames, high/low score split at 0.6.
3. Downsample tracklets to 10 Hz for the V4 back end.

**Why it may win:** a sprinting player's box IoU between samples rises from about 0.1 at 10 Hz to about 0.55 at 30 Hz. This most likely removes the residual cuts and swaps from the 10 Hz builder.

**Failure modes:** GPU and disk time; 3x more data to audit.

**Do it after V3/V4 if:** the audit still shows swaps at fast cuts in the 10 Hz per-camera tracklets.

**Deferred:** McByte/SAM crossing-window masks, CAMELTrack and SportsSUSHI. These need labels, a Linux environment or heavy compute. Revisit once ground truth exists.

---

## Label-free evaluation

**Protocol rules**
- Tune on minutes 0–3 and report on minutes 3–6 with the parameters frozen.
- Report every metric separately for team Y and team N.

**Keep from `harness.metrics()`:** events_per_min, teleports_per_min, appearance_breaks_per_min, team_impurity, dup_pairs, double_used_boxes, box_coverage, frames_in_top10.
- Caution: `ids` and `ids_over_half` are fixed by construction in V2–V7. Do not use them to rank closed-set variants.

**Metrics to add**
1. **Count consistency:** share of frames with exactly 5 Y and 5 N tracks; share of frames with more than 5 of one team (this must be 0 under the closed-set constraint); share of frames with fewer than 5 (coverage holes).
2. **Physics violations:**
   - per minute: implied speed above 9 m/s, or acceleration above 6 m/s² sustained for 0.5 s or more, after smoothing;
   - jerk outliers at link points: compare speed before and after each link.
3. **Cross-view consistency:**
   - share of samples where a track's cam1 and cam2 boxes are within the Mahalanobis gate (d² < 9.21);
   - partner switches per minute: the cam2 tracklet paired to a track's cam1 tracklet changes.
4. **Within-track ReID purity:** number of DBSCAN clusters per track on clean crops (should be 1); intra-track dispersion (mean cosine distance to the track medoid).
5. **Between-track separability:** for same-team track pairs, the gap between the identity prototypes (larger is better); the silhouette of clean crops grouped by final ID.
6. **Crossing swap score:**
   - For every same-team encounter under 1.2 m, compare each ID's clean appearance 1–3 s before and 1–3 s after.
   - A swap is flagged when cos(A_pre, B_post) + cos(B_pre, A_post) > cos(A_pre, A_post) + cos(B_pre, B_post) + 0.05.
   - Report the flagged fraction of encounters.
7. **ReID proxy AUC, for choosing embeddings:**
   - Positive pairs: same pure tracklet, at least 2 s apart.
   - Negative pairs: coexisting same-team tracklets.
   - Compute on the held-out time half.
8. **Stability and agreement:**
   - Pairwise IDF1 between variants, and between each variant and itself under ±20% parameter changes.
   - Time-reversal agreement: run on the reversed sequence and measure IDF1 against the forward output.
   - Single-camera agreement: in regions both cameras cover well, the cam1-only and cam2-only solutions should agree with the fused one.
   - Low agreement marks unstable decisions to audit.

---

## Visual audit protocol

The aim is to estimate identity purity with a few minutes of human time per variant.

1. **Per-ID sheets:** `audit_sheet(n=24)` for each final ID, cropped from the larger-view camera.
   - The rater marks every crop whose person differs from the sheet's majority.
   - Purity estimate = 1 − foreign crops / total crops.
2. **Event sheets (new):**
   - For a stratified sample of 30 crossings per variant, plus every flagged appearance break and every swap-score flag:
     - one row per involved ID showing −2, −1, +1 and +2 s crops;
     - a pitch mini-map of both trajectories.
   - The rater marks each event OK, SWAP or UNSURE.
   - Estimated swaps per minute = SWAP rate × crossings per minute.
3. **Blinding:** shuffle sheets across variants, hide variant names, and use the same sampled events for every variant (sample from the union of events).
4. **Anchor labels, about 10–15 minutes, done once:**
   - In trackview, click each of the 10 players at 5 instants spread over the 6 minutes, and give each a name.
   - This gives 50 anchors and about 1225 anchor pairs.
   - Pairwise identity precision = share of anchor pairs with the same predicted ID that have the same name.
   - Pairwise identity recall = share of same-name anchor pairs that got the same ID.
   - This is a cheap stand-in for IDF1. The same anchors can later seed the closed-set prototypes (the Maglo-style few-shot variant) and start a 60–90 s dense ground truth for TrackEval in pitch coordinates (HOTA/IDF1 with a 1 m match radius).
5. **Decision rule**
   - Promote a variant only if all of these hold on the held-out 3 minutes:
     - anchor pairwise precision does not drop;
     - crossing SWAP rate falls;
     - events_per_min and partner switches fall;
     - the count-consistency share rises.
   - Reject any variant whose team_impurity or teleports increase.