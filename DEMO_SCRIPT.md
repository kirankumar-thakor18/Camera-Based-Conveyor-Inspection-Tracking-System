# 10-minute screen-recorded demo – suggested run sheet
Record with OBS / QuickTime / Win+G. Keep terminal font large. Pre-run everything once so outputs exist; re-run live only the short commands.

| Time | Show | Say (1–2 sentences) |
|---|---|---|
| 0:00–0:45 | Repo tree, your raw video playing | Setup: static camera, boxes on a table/belt, classical CV only. |
| 0:45–2:30 | Module A: `A_segmentation_grid.png` (zoom on a frame with touching boxes) | Which method separated touching boxes and why (watershed / distance transform). Mention the from-scratch split & merge. |
| 2:30–4:00 | Module B: run `calibrate.py` (RMS error, K), open `B_undistort_before_after.png`, run `measure_box.py` | Explain K, distortion, P=K[R\|t] decomposition, and the 4-model table. |
| 4:00–5:30 | Module C: run `optical_flow.py`, open the 3 images | Dense vs sparse flow, aperture problem, RANSAC inliers/outliers, the cm/s number. |
| 5:30–7:00 | Module D: run `kalman_tracker.py`, show console predict/measure/update table, plot, overlay video incl. occlusion | Explain the state model and why prediction continues during dropout. |
| 7:00–8:15 | Module E: run `recognize.py`, show accuracy table + `E_eigenboxes.png` | Compare alignment vs eigenspace vs Hu under rotation / lighting. |
| 8:15–9:45 | `python pipeline.py …` and `pipeline_out.mp4` + printed table | Chain A→D→B→C→E; read one row: ID, size cm, speed cm/s, type. |
| 9:45–10:00 | Wrap-up | Limitations: static camera, height assumption, small dataset. |