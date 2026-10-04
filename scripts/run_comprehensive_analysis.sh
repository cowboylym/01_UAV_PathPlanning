#!/bin/sh
set -eu

ROOT=${WORKSPACE_ROOT:-/workspace}
SKIP_BUILD=${SKIP_BUILD:-0}
ANALYSIS_ONLY=${ANALYSIS_ONLY:-0}
BUILD_JOBS=${BUILD_JOBS:-2}
B0_B3_RESULT_ROOT=${B0_B3_RESULT_ROOT:-}
COMPARE4_RESULT_ROOT=${COMPARE4_RESULT_ROOT:-}

run_batch() {
    script=$1
    result_root=$2
    skip_build=$3
    if [ -n "$result_root" ]; then
        env WORKSPACE_ROOT="$ROOT" \
            RESULT_ROOT="$result_root" \
            SKIP_BUILD="$skip_build" \
            ANALYSIS_ONLY="$ANALYSIS_ONLY" \
            BUILD_JOBS="$BUILD_JOBS" \
            /bin/sh "$ROOT/$script"
    else
        env WORKSPACE_ROOT="$ROOT" \
            SKIP_BUILD="$skip_build" \
            ANALYSIS_ONLY="$ANALYSIS_ONLY" \
            BUILD_JOBS="$BUILD_JOBS" \
            /bin/sh "$ROOT/$script"
    fi
}

if [ "$ANALYSIS_ONLY" = 1 ] && \
   { [ -z "$B0_B3_RESULT_ROOT" ] || [ -z "$COMPARE4_RESULT_ROOT" ]; }; then
    echo '[错误] ANALYSIS_ONLY=1 时必须同时设置 B0_B3_RESULT_ROOT 和 COMPARE4_RESULT_ROOT' >&2
    exit 1
fi

run_batch run_b0_b3.sh "$B0_B3_RESULT_ROOT" "$SKIP_BUILD"
run_batch run_compare4.sh "$COMPARE4_RESULT_ROOT" 1
