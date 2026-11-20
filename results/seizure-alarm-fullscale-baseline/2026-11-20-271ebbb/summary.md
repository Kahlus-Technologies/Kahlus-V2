# seizure-alarm baseline at full scale (exploratory, CPU only)

Method: the repo's gradient-boosting baseline with a split-conformal threshold, graded by the repo's grader (10 s latency, 60 s quiet period). Rows are alpha 0.2, the 80% target. Data are development patients only.

| Run | Hours | Seizures | Seizures caught | Worst patient caught | Median false alarms per hour |
|---|---|---|---|---|---|
| chbmit | 269.1 | 159 | 0.76 | 0.36 | 43.1 |
| siena | 105.6 | 40 | 0.89 | 0.28 | 23.8 |
| chbmit-to-siena | 105.6 | 40 | 0.76 | 0.23 | 26.4 |
| seizeit2 | 9206.3 | 755 | 0.80 | 0.00 | 32.7 |
| chbmit-to-seizeit2 | 5085.1 | 479 | 0.82 | 0.00 | 43.4 |

The CHB-MIT row reproduces `results/baseline` of seizure-alarm (0.76 caught, 43 false alarms per hour). No network was trained here; the deep detectors are the seizure-s1 lane's work.
Per-alpha numbers are in `result.json`. The run that needed a rerun is explained there. No participant data is in this folder.

Check before citing: the within-dataset SeizeIT2 run reports 9,206.3 h and 755 seizures, but the cross-site run on the same development patients reports 5,085.1 h and 479 seizures (the figure in the research plan). I did not find why. The SeizeIT2 rows are not comparable until this is resolved.
