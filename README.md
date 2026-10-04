# 01_UAV_PathPlanning

基于 ESDF（符号距离场）的无人机三维路径规划实验代码，实现并对比四种规划算法：

| 算法 | 说明 |
|---|---|
| **FGDA*** | FMM 引导的方向状态 A*（本项目提出方法，详见 `FDGA_method.md`） |
| **FMM** | Fast Marching Method 最快到达场回溯 |
| **A*** | 自适应网格分辨率的三维 A*（`ds = clamp(d_sg/250, 5, 15)` m） |
| **RRT*** | 基于 OMPL 的最优 RRT（多次随机运行取中位代表性结果） |

所有算法共用统一的代价模型（ESDF 净空代价 + 高度惩罚 `λ_h = 0.01/m`）与统一后处理流程（ESDF 净空投影 + 段间加密 + 5 轮高斯平滑），规划超时统一为 60 秒。

## 1. 环境要求

- C++17 编译器（GCC ≥ 9 / Clang ≥ 10）
- CMake ≥ 3.10
- **GDAL**（读取 DEM GeoTIFF 与起止点 GeoJSON）
- **ITK 5.4**（构建/加载 ESDF）
- **OMPL ≥ 1.6**（RRT*）
- Eigen3
- Python 3（结果分析：`numpy` `matplotlib` `rasterio` 等）

推荐使用仓库内置的 Docker 环境运行（见下节），可免去手动安装 ITK/OMPL。

## 2. Docker 环境（推荐）

`docker/` 目录包含完整的开发容器定义：

```
docker/
├── Dockerfile            # Ubuntu 22.04 + GDAL + ITK 5.4 + OMPL 2.0.1 + Python 科学栈
├── docker-compose.yml    # 容器编排（workspace 目录挂载到 /workspace）
└── devcontainer.json     # VSCode Dev Container 配置
```

启动方式：

```bash
# 进入 docker 目录构建并启动（首次构建需编译 ITK/OMPL，耗时较长）
cd docker
docker compose run dev        # 或 docker compose up -d && docker compose exec dev bash

# VSCode 用户：直接用 Dev Containers 打开 .devcontainer 即可
```

容器内约定路径为 `/workspace/01_uav_path_planning`（对应本仓库克隆位置）。

## 3. 数据准备

由于 GitHub 文件大小限制，仓库仅包含起止点数据，其余大体积数据需自行放置到 `data/` 目录：

| 文件 | 是否必需 | 说明 |
|---|---|---|
| `data/points_3d.geojson` | ✅ 已含 | 起止点数据（GeoJSON Point 要素） |
| `data/SF_Downtown.tif` | ✅ 需自备 | DEM 地形（旧金山城区，GeoTIFF） |
| `data/SF_Downtown_sdf.mhd` + `.raw` | 可选 | ITK 格式 ESDF 缓存（约 6.9 GB）；**缺失时程序首次运行会基于 DEM 自动构建并保存**，生成后可复用 |

> 注意：`src/main.cpp` 中 DEM / ESDF 路径为硬编码默认值（第 698、719、757 行），指向容器内 `/workspace/01_uav_path_planning/data/`。非容器环境运行请修改默认值，或按相同目录结构组织数据。

## 4. 编译

```bash
cd /workspace/01_uav_path_planning
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target uav_planner -j$(nproc)
```

## 5. 运行

### 5.1 基本用法

```bash
# 默认依次运行 FGDA*、FMM、A*、RRT* 四算法（raw + smoothed 两种后处理由参数指定）
./build/uav_planner --postprocess-mode smoothed --output-dir data/my_run

# 查看全部参数
./build/uav_planner --help
```

每次运行输出 `path_comparison.csv`（各算法路径长度、规划时间、曲率、最大转向角、最小 ESDF 净空、综合评分等指标）及可视化数据文件。

### 5.2 常用命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--postprocess-mode raw\|smoothed` | smoothed | 后处理模式（smoothed 额外执行 5 轮高斯平滑） |
| `--fgda-only` / `--fmm-only` / `--astar-only` | 关闭 | 单算法模式（默认四算法全部运行） |
| `--fgda-curvature-weight <val>` | 0.1 | FGDA* 曲率权重 |
| `--fgda-traversal-weight <val>` | 1.0 | FGDA* 遍历代价权重 |
| `--fgda-clearance-weight <val>` | 0.2 | FGDA* 净空权重 |
| `--fgda-heuristic-weight <val>` | 1.0 | FGDA* 启发权重 |
| `--fgda-max-turn-angle <deg>` | 60 | FGDA* 最大转向角约束（-1 禁用；下限 35.26°，由 26 邻域几何决定） |
| `--fgda-corridor-radius <cells>` | 5 | FGDA* 搜索走廊半径 |
| `--fgda-position-state` | 关闭 | 使用位置状态（非方向状态）的 FMM 启发 A* |
| `--rrt-runs <n>` | 10 | RRT* 随机重复次数（取路径长度中位数者为代表） |
| `--rrt-seed <n>` | 42 | RRT* 基础种子（第 k 次运行用 seed+k） |
| `--rrt-iters <n>` | 150 | RRT* 迭代次数 |
| `--planner-label <label>` | — | 单算法实验的 CSV 标签 |
| `--high-curvature-threshold <deg/m>` | — | 高曲率阈值（统计指标） |
| `--export-fmm-3d` | 关闭 | 导出完整 FMM 三维到达时间场 |
| `--export-fgda-corridor` | 关闭 | 导出 FGDA* 引导路径、走廊与到达代价场 |
| `--output-dir <path>` | — | 结果输出目录 |

### 5.3 单算法示例

```bash
# 仅运行 FGDA*，raw 后处理，扫描曲率权重
./build/uav_planner --fgda-only --postprocess-mode raw \
    --fgda-curvature-weight 0.2 --planner-label "w_kappa=0.2" \
    --output-dir data/w_kappa/wk-0.2
```

## 6. 实验脚本

`scripts/` 目录封装了批量实验入口（在容器内以 `sh` 运行）。**克隆仓库后请先设置：**

```bash
export WORKSPACE_ROOT=/workspace/01_uav_path_planning   # 仓库根目录
```

### 6.1 主对比实验（`run_compare4.sh`）

四算法 ×（raw / smoothed）后处理全流程 + Python 统计分析：

```bash
# compare4 分析依赖独立 Python 虚拟环境（首次使用先创建）
python3 -m venv /opt/venvs/compare4
/opt/venvs/compare4/bin/pip install numpy matplotlib

sh scripts/run_compare4.sh          # 自动编译 + 运行 + 分析
```

结果输出到 `data/compare4/<RUN_ID>/`，包含 `experiment_parameters.txt`、两组 `path_comparison.csv` 及分析图表。

常用环境变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `RUN_ID` | 时间戳 | 实验批次目录名 |
| `SKIP_BUILD` | 0 | 设为 1 跳过编译 |
| `ANALYSIS_ONLY` | 0 | 设为 1 仅对已有批次重新分析 |
| `FGDA_MAX_TURN_ANGLE` | 60 | FGDA* 最大转向角 |
| `FGDA_CURVATURE_WEIGHT` 等 | 0.1 等 | 透传对应命令行参数 |
| `RRT_RUNS` / `RRT_SEED` / `RRT_ITERS` | 10 / 42 / 150 | RRT* 配置 |

### 6.2 消融实验（`run_b0_b3.sh`）

FGDA* 真实机制消融（B0–B3，含曲率权重隔离），结果输出到 `data/ablation/<RUN_ID>/`：

```bash
sh scripts/run_b0_b3.sh
```

其参数与主实验 FGDA* 配置保持一致，仅隔离目标机制。`run_g0_g3.sh` / `run_g0_g3.bat` 已停用，自动转交本脚本。

### 6.3 曲率权重扫描（`run_w_kappa_scan.sh`）

```bash
sh scripts/run_w_kappa_scan.sh    # 默认扫描 0.0 0.05 0.1 0.2 0.3
W_KAPPA_VALUES="0.05 0.1 0.3" sh scripts/run_w_kappa_scan.sh   # 自定义扫描点
```

### 6.4 综合分析（`run_comprehensive_analysis.sh`）

依次调度消融实验与主对比实验的批量分析入口，支持 `ANALYSIS_ONLY=1` 复用已有结果。

> `run_lookahead_scan.sh` 已停用（方向前瞻参数已从代码中移除）。

## 7. 目录结构

```
├── CMakeLists.txt          # 构建配置（GDAL + ITK + OMPL）
├── FDGA_method.md          # FGDA* 方法说明文档
├── run_compare4.sh         # 主对比实验入口（等价于 scripts/ 中调用）
├── src/
│   ├── main.cpp            # 主程序：ESDF 加载/构建 + 四算法运行 + 指标输出
│   ├── analyze_compare4.py # compare4 统计分析与图表
│   ├── vis*.py             # 各类可视化脚本
│   └── ...
├── scripts/                # 批量实验脚本
├── docker/                 # 开发容器配置
└── data/                   # 数据目录（仅含起止点 GeoJSON，其余自备）
```

## 8. 关键实验设定

- 安全余量 5 m：后处理 ESDF 投影沿 +∇ESDF 迭代推进直至净空 ≥ 5 m（≤ 60 步）
- 统一后处理：ESDF 净空 + 段间加密 → 5 轮（1 轮 5 点高斯平滑 + ESDF 再净空）
- 路径评估采用 1 m 步长密集采样进行安全检测
- 统计比较采用带 Holm 校正的 Wilcoxon 配对检验
- RRT* 具有随机性，结论需基于多次运行（≥ 5 次）
