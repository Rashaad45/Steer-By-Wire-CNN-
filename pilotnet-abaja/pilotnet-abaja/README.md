# End-to-end CNN steering for an aBAJA autonomous vehicle (PilotNet)

Camera image → convolutional neural network → steering. PyTorch for training, a MATLAB
navigation and vehicle simulator for closed-loop testing, and ONNX to connect the two.
The whole project is **two files**:

```
pilotnet_all.py        ALL Python: PilotNet + FOLLOW/LEFT/RIGHT heads, data readers, augmentation,
                       metrics, VisualBackProp, simulator (course, A*, texture, renderer, bicycle,
                       pure pursuit), dataset generator, training, evaluation, ONNX + .mat export,
                       MATLAB reference writer, closed loop, ONNX inference class, unit tests,
                       dataset download + conversion
matlab/abaja_sim.m     ALL MATLAB (one function file, every former .m is a local function)
matlab/reference/      reference renders + CNN outputs (MATLAB == Python check)
runs/sim/              trained conditional model: best.pt, pilotnet.onnx, pilotnet_weights.mat, results/
datasets/              downloaded / generated data, see datasets/DATASETS.md
legacy/                the previous multi-file version (python/ package, 20 .m files), kept unchanged
requirements.txt
```
The single files are a verbatim merge of the old code. Byte-identical renders, CNN outputs, `labels.csv`
and closed-loop metrics were checked against the old version before it was moved to `legacy/`.

## 1. Setup (Windows 11, PowerShell, VS Code)

```powershell
cd "C:\Project\Steer by wire\pilotnet-abaja\pilotnet-abaja"
python -m venv .venv                     # optional; the system Python 3.11 also works
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python pilotnet_all.py selftest          # all unit tests, prints PASS/FAIL per test
```
Relative paths given to any command are resolved from the folder holding `pilotnet_all.py`, so the commands
work from any working directory. A CUDA GPU is optional. CPU training of the simulator model takes about 20–40 min.

## 2. Commands (`python pilotnet_all.py <command> -h` lists every flag)

| command | what it does | was |
|---|---|---|
| `selftest [-k name]` | unit tests (old `tests/test_pipeline.py` + renderer/CNN reference, closed loop, converters) | `pytest tests` |
| `download --what {sully,udacity_sim,udacity_sample,comma,comma2k19,tartandrive,rellis,all} [--dest datasets] [--yes]` | fetch datasets (>5 GB needs `--yes`; disk space is checked) | – |
| `gen --out datasets/sim --cycles 6 --seed 0 ...` | simulator dataset | `python -m sim.gen_dataset` |
| `train --dataset {sim,sully,udacity} --root ... --out runs/...` | training (same flags/defaults as before) | `train.py` |
| `evaluate --ckpt runs/x/best.pt --root ... [--workers 2]` | open-loop metrics, plots, VisualBackProp | `evaluate.py` |
| `export --ckpt runs/x/best.pt` | `pilotnet.onnx`, `pilotnet_meta.json`, `pilotnet_weights.mat` | `export_onnx.py` |
| `reference --onnx runs/sim/pilotnet.onnx` | rewrite `matlab/reference/` | `make_reference.py` |
| `closedloop --onnx ... [--policy expert] [--texture-seed N] [--gain G] [--speed V] [--tag T]` | closed-loop simulation | `python -m sim.closed_loop` |
| `convert --source {comma,comma2k19,tartandrive} --src ... --dst ...` | convert a real-world log into `labels.csv + images/` | – |

**Windows DataLoader workers.** `train` (default `--workers 4`) and `evaluate` (default `--workers 2`) use spawned
worker processes. All code runs under `if __name__ == "__main__":`, so this works on Windows. If a worker still
fails (low virtual memory "paging file is too small", antivirus, broken pipe), the error message tells you to re-run
with `--workers 0`.

## 3. Datasets
See **`datasets/DATASETS.md`** for links, sizes, licences, citations and label conventions.

| `--dataset` | folder | notes |
|---|---|---|
| `sim` | `datasets/sim/` (`gen`) | off-road track with a fork, LEFT/RIGHT commands, target = κ/κmax (+ = left) |
| `sully` | `datasets/sullychen/driving_dataset/` | real road driving; label = steering-wheel angle → radians |
| `udacity` | `datasets/udacity/` (you record it) or `datasets/udacity_sample/data/` | 3 cameras, ±0.2 side-camera correction |
| `sim` | `datasets/<x>_csv/` (`convert`) | comma / comma2k19 / TartanDrive converted; `command` = 0, train with `--branches 1` |

## 4. Typical run (simulator → MATLAB)

```powershell
python pilotnet_all.py gen --out datasets\sim --cycles 6
python pilotnet_all.py train --dataset sim --root datasets\sim --branches 3 --balance --cache --out runs\sim
python pilotnet_all.py evaluate --ckpt runs\sim\best.pt --root datasets\sim
python pilotnet_all.py export --ckpt runs\sim\best.pt
python pilotnet_all.py reference --onnx runs\sim\pilotnet.onnx
python pilotnet_all.py closedloop --onnx runs\sim\pilotnet.onnx --out runs\sim\results
```
Outputs: `runs/<name>/best.pt`, `history.csv`, `split.json`, `eval/metrics.json`, `eval/*.png`.
A trained conditional model is already in `runs/sim/`. Its simulator results: 0 interventions over 1,049 m,
4/4 forks, mean |CTE| 0.044 m.

## 5. MATLAB simulation (`matlab/abaja_sim.m`)

```matlab
cd matlab
abaja_sim('demo')                                    % course -> A* -> render check -> expert -> CNN table
abaja_sim('demo', '..\runs\sim\pilotnet.onnx', 'native', 2)
ok  = abaja_sim('check')                             % renders + CNN equal to Python
abaja_sim('route')                                   % plot course + A* route S->A->S->B->S
res = abaja_sim('closedloop', 'expert', [], 1)
res = abaja_sim('closedloop', 'cnn', '..\runs\sim\pilotnet.onnx', 2, struct('speed', 6, 'startOffset', 1.5))
abaja_sim('generate', '..\datasets\sim_matlab', 6, 0)
```
CNN backends (`opts.backend` or the 2nd `check` argument):
* `'native'` (default): plain MATLAB, no toolbox, reads `pilotnet_weights.mat` next to the `.onnx`, matches ONNX to ~1e-7.
* `'python'`: ONNX Runtime through `pyenv`, e.g. `pyenv('Version','C:\Program Files\Python311\python.exe')`.
* `'matlab'`: Deep Learning Toolbox importer.

Base MATLAB R2019b or newer is enough (no Image Processing / Statistics toolbox calls). The file is written to also
run in GNU Octave 7+: no nested functions, `struct2table` and `yline` are avoided, and there is no video in Octave.

## 6. Conventions
* Simulator label: `target = kappa / kappa_max`, kappa = path curvature [1/m], **+ = left**,
  kappa_max = tan(30°)/1.4 m = 0.412 1/m. Steering: `delta = atan(L·kappa)`.
* Commands: 0 FOLLOW, 1 LEFT, 2 RIGHT (active from 18 m before to 8 m after a fork).
* Datasets are split by contiguous 400–500-frame blocks, not by single frames. Adjacent frames are
  near-duplicates, so a per-frame random split leaks training data into the test set.
* `convert` outputs keep each source's own steering sign and scale (documented in `dataset_meta.json` and
  `datasets/DATASETS.md`). Check the sign against your actuator before using such a model on the vehicle.

## Safety and data licences
* **SullyChen's README says these datasets must never be used to pilot a real car:**
  *"Absolutely, under NO circumstance, should one ever pilot a car using computer vision software trained on these
  datasets"*. They are for research and statistics only. Use Sully models for offline benchmarking only.
* comma.ai research and RELLIS-3D are CC BY-NC-SA 3.0 (non-commercial, share-alike). comma2k19, TartanDrive and the
  Udacity simulator are MIT. Cite them as listed in `datasets/DATASETS.md`.
* On the BAJA vehicle, test any model trained here only on a closed course, with a remote e-stop, a speed limit and a
  safety driver or chase operator.
