#!/bin/sh
set -eu

ROOT=${WORKSPACE_ROOT:-/workspace}
BIN=${BIN:-$ROOT/build/uav_planner}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RESULT_ROOT=${RESULT_ROOT:-$ROOT/data/ablation/$RUN_ID}
W_KAPPA=${W_KAPPA:-0.1}
FGDA_MAX_TURN_ANGLE=${FGDA_MAX_TURN_ANGLE:-60}
BUILD_JOBS=${BUILD_JOBS:-2}
SKIP_BUILD=${SKIP_BUILD:-0}
ANALYSIS_ONLY=${ANALYSIS_ONLY:-0}

require_file() {
    [ -s "$1" ] || { echo "[错误] 缺少结果或文件为空: $1" >&2; exit 1; }
}

if [ "$ANALYSIS_ONLY" != 1 ]; then
    if [ -e "$RESULT_ROOT" ]; then
        echo "[错误] 实验目录已存在，拒绝覆盖: $RESULT_ROOT" >&2
        exit 1
    fi
    mkdir -p "$RESULT_ROOT"
    if [ "$SKIP_BUILD" != 1 ]; then
        cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release
        cmake --build "$ROOT/build" --target uav_planner -j "$BUILD_JOBS"
    fi
    [ -x "$BIN" ] || { echo "[错误] 找不到可执行程序: $BIN" >&2; exit 1; }

    cat > "$RESULT_ROOT/experiment_parameters.txt" <<EOF
run_id=$RUN_ID
postprocess_mode=raw
B0=standard A* with Euclidean heuristic, position state, no curvature
B1=FMM arrival-time heuristic, position-state A*, no curvature
B2=FMM arrival-time heuristic, direction-state A*, curvature weight 0
B3=full FGDA*, curvature weight $W_KAPPA
fgda_max_turn_angle_deg=$FGDA_MAX_TURN_ANGLE
EOF

    mkdir -p "$RESULT_ROOT/B0" "$RESULT_ROOT/B1" "$RESULT_ROOT/B2" "$RESULT_ROOT/B3"
    "$BIN" --astar-only --planner-label B0 --postprocess-mode raw --output-dir "$RESULT_ROOT/B0"
    "$BIN" --fgda-only --fgda-position-state --fgda-curvature-weight 0 --fgda-max-turn-angle "$FGDA_MAX_TURN_ANGLE" --planner-label B1 --postprocess-mode raw --output-dir "$RESULT_ROOT/B1"
    "$BIN" --fgda-only --fgda-curvature-weight 0 --fgda-max-turn-angle "$FGDA_MAX_TURN_ANGLE" --planner-label B2 --postprocess-mode raw --output-dir "$RESULT_ROOT/B2"
    "$BIN" --fgda-only --fgda-curvature-weight "$W_KAPPA" --fgda-max-turn-angle "$FGDA_MAX_TURN_ANGLE" --planner-label B3 --postprocess-mode raw --output-dir "$RESULT_ROOT/B3"
fi

for group in B0 B1 B2 B3; do
    require_file "$RESULT_ROOT/$group/path_comparison.csv"
done
python3 "$ROOT/src/Analysis.py" --exp ablation --batch-dir "$RESULT_ROOT"
echo "B0-B3 消融实验分析完成: $RESULT_ROOT"
