# ConvexAdam 运行环境 (ENV.md)

本文件记录在 WSL Ubuntu 中搭建 ConvexAdam 运行环境的可复现步骤。
默认工具链规约见上级 `../CLAUDE.md` §5（Python 3.12 / pip 优先 / CUDA 13.2）。

## 机器（实测）
- WSL distro: Ubuntu 24.04.2 LTS
- GPU: NVIDIA RTX 4090 (24GB)；驱动 CUDA 13.3（向下兼容 13.2）
- 系统 python: `python3` = 3.12.3（无 pip / 无 ensurepip，且 sudo 需密码）

## 一次性：建立独立 venv 并自举 pip（免 sudo）
系统 python 没装 pip、`ensurepip` 也缺，且 sudo 需密码。用 `--without-pip` 建 venv，
再 `get-pip.py` 在 venv 内自举：

```bash
mkdir -p ~/.venvs
python3 -m venv --without-pip ~/.venvs/convexadam
. ~/.venvs/convexadam/bin/activate
wget -qO /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py
python /tmp/get-pip.py
```

## 安装依赖
ConvexAdam 依赖 torch（`convex_adam_pt` 的 MIND/凸优化用 torch），按 CUDA 13.2 取官方轮子：

```bash
. ~/.venvs/convexadam/bin/activate
# torch + CUDA 13.2 运行库
python -m pip install torch --index-url https://download.pytorch.org/whl/cu132/
# ConvexAdam 本体（会补装 nibabel/numpy/scikit-learn/SimpleITK；torch 已满足不会被重装）
python -m pip install convexAdam
# 可视化
python -m pip install matplotlib
```

实测版本：
- torch 2.13.0+cu132  → `torch.cuda.is_available()` = True，device = RTX 4090
- convexAdam 0.2.0, SimpleITK 2.5.5, numpy 2.5.1, scikit-learn 1.9.0, scipy 1.18.0, nibabel 5.4.2
- matplotlib 3.11.0, pydicom 3.0.2（CHAOS DICOM 转换用）

## 运行
仓库在 WSL 内路径：`/mnt/e/Hans_Files/t1-t2-all-in-one/ConvexAdam`

```bash
. ~/.venvs/convexadam/bin/activate
cd /mnt/e/Hans_Files/t1-t2-all-in-one/ConvexAdam

# 1) 跑自带各向异性测试（用 bundled t2w/adc 数据，验证安装+GPU）
PYTHONPATH=tests python tests/test_convex_adam_mind_aniso.py

# 2) 可视化 + 量化（配准前后 NCC/MSE 对比，输出到 experiments/figures/）
python experiments/validate_known_transform.py
```

## CHAOS 真实数据实验（自己实现的干净管线）

```bash
. ~/.venvs/convexadam/bin/activate
cd /mnt/e/Hans_Files/t1-t2-all-in-one/ConvexAdam

# 1) CHAOS DICOM -> RAS NIfTI（自写转换器，从只读的 ../chaos-raw 读，写到 chaos_data/）
python experiments/convert_chaos.py --split Train --cases 1 2

# 2) nib<->sitk 几何互转自检（往返精确）
PYTHONPATH=experiments python experiments/_roundtrip_check.py

# 3) 真实 CHAOS 上的 ConvexAdam 验证（已知形变恢复 + T1->T2 跨模态）
PYTHONPATH=experiments python experiments/validate_chaos.py
```

脚本（均在 `experiments/`）：
- `convert_chaos.py` — DICOM→RAS NIfTI（IPP 投影排序、相邻 IPP 求切片方向与 z-spacing、`as_closest_canonical` 保 RAS）。
- `imaging_utils.py` — nib↔SimpleITK 互转（往返精确）、ISO 重采样+裁剪、形变生成(仿射+弹性)+后向 warp、NCC/MSE/ROI-NCC、解剖学朝向可视化。
- `validate_chaos.py` — 两个实验：1a identity 自检、1b 已知形变恢复（T2SPIR）、2 真实 T1→T2 跨模态。

case 1 结果（2mm iso, crop 160×160×110, RTX 4090）：
- 1a identity：NCC=0.999（管线正确）
- 1b 形变恢复（|phi|~4vx）：**NCC 0.933 → 0.989**（ROI 0.928 → 0.997），nMSE 降 6×
- 2 T1→T2 跨模态：NCC 基本不变（0.458→0.439）——**原始强度 NCC 是弱指标**，公正评价需 CHAOS 器官标签 Dice（后续）

图输出在 `experiments/runs/`（已 gitignore，脚本可复现）。

## 备注
- 测试输出落在 `tests/output/10000/`（已被本仓 `.gitignore` 忽略，不入库）。
- 实验脚本与图放在 `experiments/`（纳入版本控制，提交到本 fork）。
