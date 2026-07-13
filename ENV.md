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
- matplotlib 3.11.0

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

## 备注
- 测试输出落在 `tests/output/10000/`（已被本仓 `.gitignore` 忽略，不入库）。
- 实验脚本与图放在 `experiments/`（纳入版本控制，提交到本 fork）。
