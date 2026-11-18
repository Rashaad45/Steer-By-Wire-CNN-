# End-to-end CNN steering for an aBAJA autonomous vehicle (PilotNet)

Camera image → convolutional neural network → steering. PyTorch for training, a MATLAB
navigation and vehicle simulator for closed-loop testing, and ONNX to connect the two.

```
python/                      training, evaluation, export (run in VS Code)
  pilotnet/model.py          PilotNet + optional command-conditioned heads (FOLLOW/LEFT/RIGHT)
  pilotnet/data.py           readers: SullyChen, Udacity simulator, aBAJA simulator; block split
  pilotnet/augment.py        flip (+label), shift (+label), brightness, shadow, pitch, blur, noise
  pilotnet/vbp.py            VisualBackProp saliency (Bojarski et al., ICRA 2018)
  pilotnet/metrics.py        RMSE/MAE/R², smoothness, NVIDIA autonomy metric
  train.py  evaluate.py  export_onnx.py  pilotnet_infer.py  make_reference.py
  sim/                       Python reference copy of the MATLAB simulator (bit-identical renders)
  tests/test_pipeline.py     unit tests (pytest)
matlab/
  main_abaja_demo.m          one-click demo: course → A* route → expert run → CNN closed loop
  abaja_config.m             every constant (mirrors python/sim/config.py)
  build_course.m hermite_chain.m astar_graph.m build_route.m        navigation layer
  build_texture.m camera_precompute.m render_camera.m               camera simulator
  bicycle_step.m pure_pursuit.m nearest_on_route.m cross_track.m    vehicle + expert
  generate_dataset.m run_closed_loop.m                              data + closed-loop scoring
  cnn_load.m cnn_predict.m pilotnet_native_load.m pilotnet_native_forward.m
  abaja_render_check.m  reference/                                  MATLAB == Python check
```

## 1. Setup (Windows / VS Code)

```bash
cd python
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r ../requirements.txt
pytest -q tests                   # 9 tests should pass
```
A CUDA GPU is optional. CPU training of the simulator model takes about 20–40 min.

## 2. Datasets

| name | where | put it in | notes |
|---|---|---|---|
| SullyChen 2017 (~45.5k frames) or 2018 (~63k) | github.com/SullyChen/driving-datasets (Google Drive links in README) | `data/driving_dataset/` (folder holding `data.txt` + jpgs) | real road driving; label = steering-wheel angle |
| Udacity simulator recordings | github.com/udacity/self-driving-car-sim (Term 1 simulator, “Training mode”) | `data/udacity/` (`driving_log.csv` + `IMG/`) | 3 cameras; ±0.2 side-camera correction |
| aBAJA simulator | generated here | `data/sim/` | off-road track with a fork, LEFT/RIGHT commands |

## 3. Train → evaluate → export (Python)

```bash
# a) aBAJA simulator data (needed for the MATLAB closed loop)
python -m sim.gen_dataset --out ../data/sim --cycles 6
python train.py --dataset sim --root ../data/sim --branches 3 --balance --cache --out runs/sim
python evaluate.py --ckpt runs/sim/best.pt --root ../data/sim
python export_onnx.py --ckpt runs/sim/best.pt           # -> pilotnet.onnx + pilotnet_weights.mat
python make_reference.py --onnx runs/sim/pilotnet.onnx  # reference images for the MATLAB check
python -m sim.closed_loop --onnx runs/sim/pilotnet.onnx --out runs/sim/closed_loop

# b) real-world benchmark (open loop only)
python train.py --dataset sully --root ../data/driving_dataset --cache --out runs/sully
python evaluate.py --ckpt runs/sully/best.pt --root ../data/driving_dataset
python train.py --dataset udacity --root ../data/udacity --balance --out runs/udacity
```
Outputs: `runs/<name>/best.pt`, `history.csv`, `eval/metrics.json`, `eval/*.png`
(time series, scatter, VisualBackProp).

## 4. MATLAB simulation

A trained conditional model is already in `python/runs/sim/` (`pilotnet.onnx`, `pilotnet_weights.mat`, `best.pt`, plus `results/`), so the MATLAB demo runs before you train anything.
Its simulator results: 0 interventions over 1,049 m, 4/4 forks, mean |CTE| 0.044 m.


Open `matlab/main_abaja_demo.m`, check `ONNX_PATH`, and run it. It:
1. builds the road graph, plans the mission with A*, and assigns FOLLOW/LEFT/RIGHT commands;
2. runs `abaja_render_check` (MATLAB images and CNN outputs must match Python);
3. drives the pure-pursuit expert, then the CNN, and prints the autonomy, cross-track error and fork-compliance table.

`BACKEND='native'` runs the network in plain MATLAB (no toolboxes, matches ONNX to ~1e-8).
`'python'` uses ONNX Runtime through `pyenv`, and `'matlab'` uses the Deep Learning Toolbox importer.
Base MATLAB R2019b or newer is enough. The code also runs in GNU Octave 7+.

## 5. Conventions
* Simulator label: `target = kappa / kappa_max`, kappa = path curvature [1/m], **+ = left**,
  kappa_max = tan(30°)/1.4 m = 0.412 1/m. Steering: `delta = atan(L·kappa)`.
* Commands: 0 FOLLOW, 1 LEFT, 2 RIGHT (active from 18 m before to 8 m after a fork).
* Datasets are split by contiguous 400–500-frame blocks, not by single frames. Adjacent frames are
  near-duplicates, so a per-frame random split leaks training data into the test set.

## Safety
SullyChen's README says these datasets must never be used to drive a real car. The same goes for any
model trained here. On the BAJA vehicle, test only on a closed course with a remote e-stop, a
speed limit and a safety driver or chase operator.
