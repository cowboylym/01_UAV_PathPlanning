#!/bin/sh
set -eu

ROOT=${WORKSPACE_ROOT:-/workspace/01_uav_path_planning}
VENV=${VENV:-/opt/venvs/compare4}
[ -f "$VENV/bin/activate" ] || { echo "[错误] 找不到 Python 虚拟环境: $VENV" >&2; exit 1; }
. "$VENV/bin/activate"

BIN=${BIN:-$ROOT/build/uav_planner}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RESULT_ROOT=${RESULT_ROOT:-$ROOT/data/compare4/$RUN_ID}
FGDA_CURVATURE_WEIGHT=${FGDA_CURVATURE_WEIGHT:-0.1}
FGDA_TRAVERSAL_WEIGHT=${FGDA_TRAVERSAL_WEIGHT:-1.0}
FGDA_CLEARANCE_WEIGHT=${FGDA_CLEARANCE_WEIGHT:-0.2}
FGDA_HEURISTIC_WEIGHT=${FGDA_HEURISTIC_WEIGHT:-1.0}
FGDA_MAX_TURN_ANGLE=${FGDA_MAX_TURN_ANGLE:-60}
FGDA_CORRIDOR_RADIUS=${FGDA_CORRIDOR_RADIUS:-5}
RRT_RUNS=${RRT_RUNS:-10}
RRT_SEED=${RRT_SEED:-42}
RRT_ITERS=${RRT_ITERS:-150}
BUILD_JOBS=${BUILD_JOBS:-2}
SKIP_BUILD=${SKIP_BUILD:-0}
ANALYSIS_ONLY=${ANALYSIS_ONLY:-0}

require_file() {
    [ -s "$1" ] || { echo "[错误] 缺少实验结果或文件为空: $1" >&2; exit 1; }
}

if [ "$ANALYSIS_ONLY" != 1 ]; then
    if [ -e "$RESULT_ROOT" ]; then
        echo "[错误] 实验目录已存在，拒绝覆盖: $RESULT_ROOT" >&2
        exit 1
    fi
    mkdir -p "$RESULT_ROOT/raw" "$RESULT_ROOT/smoothed"
    if [ "$SKIP_BUILD" != 1 ]; then
        cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release
        cmake --build "$ROOT/build" --target uav_planner -j "$BUILD_JOBS"
    fi
    [ -x "$BIN" ] || { echo "[错误] 找不到可执行程序: $BIN" >&2; exit 1; }
    cat > "$RESULT_ROOT/experiment_parameters.txt" <<EOF
run_id=$RUN_ID
algorithms=FGDA*,FMM,A*,RRT*
postprocess_modes=raw,smoothed
postprocess_definition=both modes use common ESDF projection; smoothed additionally uses 5 Gaussian rounds
fgda_curvature_weight=$FGDA_CURVATURE_WEIGHT
fgda_traversal_weight=$FGDA_TRAVERSAL_WEIGHT
fgda_clearance_weight=$FGDA_CLEARANCE_WEIGHT
fgda_heuristic_weight=$FGDA_HEURISTIC_WEIGHT
fgda_max_turn_angle_deg=$FGDA_MAX_TURN_ANGLE
fgda_corridor_radius_cells=$FGDA_CORRIDOR_RADIUS
rrt_runs=$RRT_RUNS
rrt_seed=$RRT_SEED
rrt_seed_rule=run k uses rrt_seed+k
rrt_iters=$RRT_ITERS
rrt_representative=successful run with median 3D path length
planner_timeout_seconds=60
EOF
    for mode in raw smoothed; do
        "$BIN" --postprocess-mode "$mode" \
            --fgda-curvature-weight "$FGDA_CURVATURE_WEIGHT" \
            --fgda-traversal-weight "$FGDA_TRAVERSAL_WEIGHT" \
            --fgda-clearance-weight "$FGDA_CLEARANCE_WEIGHT" \
            --fgda-heuristic-weight "$FGDA_HEURISTIC_WEIGHT" \
            --fgda-max-turn-angle "$FGDA_MAX_TURN_ANGLE" \
            --fgda-corridor-radius "$FGDA_CORRIDOR_RADIUS" \
            --rrt-runs "$RRT_RUNS" --rrt-seed "$RRT_SEED" --rrt-iters "$RRT_ITERS" \
            --output-dir "$RESULT_ROOT/$mode"
        require_file "$RESULT_ROOT/$mode/path_comparison.csv"
    done
else
    [ -d "$RESULT_ROOT" ] || { echo "[错误] 找不到待分析批次: $RESULT_ROOT" >&2; exit 1; }
fi

require_file "$RESULT_ROOT/raw/path_comparison.csv"
require_file "$RESULT_ROOT/smoothed/path_comparison.csv"
python3 "$ROOT/src/analyze_compare4.py" "$RESULT_ROOT"
echo "四算法实验分析完成: $RESULT_ROOT"
