#!/bin/sh
set -eu
printf '%s\n' '[停用] G0-G3 已替换为真实机制消融 B0-B3，转交 run_b0_b3.sh。' >&2
exec /bin/sh "${WORKSPACE_ROOT:-/workspace}/run_b0_b3.sh" "$@"
