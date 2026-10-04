#!/bin/sh
set -eu

ROOT=${WORKSPACE_ROOT:-/workspace}
BIN=${BIN:-$ROOT/build/uav_planner}
W_KAPPA=${W_KAPPA:-0.1}
W_KAPPA_VALUES=${W_KAPPA_VALUES:-"0.0 0.05 $W_KAPPA 0.2 0.3"}
SKIP_EXISTING=${SKIP_EXISTING:-0}
RESULT_ROOT=${W_KAPPA_ROOT:-$ROOT/data/w_kappa}

printf '%s\n' '[提示] 扫描 FGDA* 曲率权重；全部使用 raw 后处理。'
for weight in $W_KAPPA_VALUES; do
    out="$RESULT_ROOT/wk-$weight"
    csv="$out/path_comparison.csv"
    if [ -f "$csv" ]; then
        if [ "$SKIP_EXISTING" = 1 ]; then
            echo "[跳过] w_kappa=$weight: $csv 已存在"
            continue
        fi
        echo "[错误] 已有结果，拒绝覆盖: $csv" >&2
        exit 1
    fi
    mkdir -p "$out"
    echo "[运行] FGDA* w_kappa=$weight -> $out"
    "$BIN" --fgda-only --planner-label "w_kappa=$weight" \
        --postprocess-mode raw --fgda-curvature-weight "$weight" \
        --output-dir "$out"
done
