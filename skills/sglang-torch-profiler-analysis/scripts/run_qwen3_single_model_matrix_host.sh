#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_qwen3_single_model_matrix_host.sh \
    --model-id qwen3_14b \
    --model Qwen/Qwen3-14B \
    --root /data/bbuf/validate/unified_llm_profiler_skill/runs/20260422_h100_qwen3_matrix_v2 \
    --sglang-gpu 0 \
    --vllm-gpu 1 \
    --trt-formal-gpu 3 \
    --trt-mapping-gpu 5 \
    --sglang-port 30098 \
    --vllm-formal-port 31098 \
    --vllm-mapping-port 31099 \
    --trt-formal-prefill-port 32098 \
    --trt-formal-decode-port 32099 \
    --trt-mapping-prefill-port 32198 \
    --trt-mapping-decode-port 32199

This script is intended to run on the H100 host. It:
1. captures SGLang live profiling and writes `analysis_sglang.txt`
2. captures vLLM formal + eager mapping traces and writes `analysis_vllm.txt`
3. captures TensorRT-LLM formal + graph-off mapping traces and writes `analysis_trtllm.txt`

Environment:
  Export `HF_TOKEN` and `HUGGINGFACE_HUB_TOKEN` before running.
EOF
}

MODEL_ID=""
MODEL=""
ROOT=""
SGLANG_GPU=""
VLLM_GPU=""
TRT_FORMAL_GPU=""
TRT_MAPPING_GPU=""
SGLANG_PORT=""
VLLM_FORMAL_PORT=""
VLLM_MAPPING_PORT=""
TRT_FORMAL_PREFILL_PORT=""
TRT_FORMAL_DECODE_PORT=""
TRT_MAPPING_PREFILL_PORT=""
TRT_MAPPING_DECODE_PORT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-id) MODEL_ID="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --root) ROOT="$2"; shift 2 ;;
    --sglang-gpu) SGLANG_GPU="$2"; shift 2 ;;
    --vllm-gpu) VLLM_GPU="$2"; shift 2 ;;
    --trt-formal-gpu) TRT_FORMAL_GPU="$2"; shift 2 ;;
    --trt-mapping-gpu) TRT_MAPPING_GPU="$2"; shift 2 ;;
    --sglang-port) SGLANG_PORT="$2"; shift 2 ;;
    --vllm-formal-port) VLLM_FORMAL_PORT="$2"; shift 2 ;;
    --vllm-mapping-port) VLLM_MAPPING_PORT="$2"; shift 2 ;;
    --trt-formal-prefill-port) TRT_FORMAL_PREFILL_PORT="$2"; shift 2 ;;
    --trt-formal-decode-port) TRT_FORMAL_DECODE_PORT="$2"; shift 2 ;;
    --trt-mapping-prefill-port) TRT_MAPPING_PREFILL_PORT="$2"; shift 2 ;;
    --trt-mapping-decode-port) TRT_MAPPING_DECODE_PORT="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN before running." >&2
  exit 2
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
fi
if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

for value in \
  MODEL_ID MODEL ROOT \
  SGLANG_GPU VLLM_GPU TRT_FORMAL_GPU TRT_MAPPING_GPU \
  SGLANG_PORT VLLM_FORMAL_PORT VLLM_MAPPING_PORT \
  TRT_FORMAL_PREFILL_PORT TRT_FORMAL_DECODE_PORT \
  TRT_MAPPING_PREFILL_PORT TRT_MAPPING_DECODE_PORT; do
  if [[ -z "${!value}" ]]; then
    echo "Missing required argument: $value" >&2
    usage >&2
    exit 2
  fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="$ROOT/$MODEL_ID"
SGLANG_LOG="$MODEL_ROOT/sglang_server.log"
SGLANG_PROFILE_ROOT="$MODEL_ROOT/sglang_profile_live"
SGLANG_ANALYSIS="$MODEL_ROOT/analysis_sglang.txt"
VLLM_FORMAL_DIR="$MODEL_ROOT/vllm_formal"
VLLM_MAPPING_DIR="$MODEL_ROOT/vllm_mapping"
VLLM_ANALYSIS="$MODEL_ROOT/analysis_vllm.txt"
TRT_FORMAL_DIR="$MODEL_ROOT/trtllm_formal"
TRT_MAPPING_DIR="$MODEL_ROOT/trtllm_mapping"
TRT_ANALYSIS="$MODEL_ROOT/analysis_trtllm.txt"

docker exec sglang_bbuf bash -lc "mkdir -p '$MODEL_ROOT' '$SGLANG_PROFILE_ROOT'"

cleanup() {
  docker exec sglang_bbuf bash -lc "pkill -f 'sglang.launch_server.*--port $SGLANG_PORT' >/dev/null 2>&1 || true" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[1/6] SGLang server + probe + live triage"
docker exec -d sglang_bbuf bash -lc "cd /data/bbuf/repos/sglang && export HF_TOKEN='$HF_TOKEN' HUGGINGFACE_HUB_TOKEN='$HUGGINGFACE_HUB_TOKEN' && CUDA_VISIBLE_DEVICES=$SGLANG_GPU PYTHONPATH=python python3 -m sglang.launch_server --model-path '$MODEL' --port $SGLANG_PORT --tp-size 1 --mem-fraction-static 0.85 --trust-remote-code > '$SGLANG_LOG' 2>&1"
docker exec sglang_bbuf bash -lc "for i in \$(seq 1 180); do curl -sf http://127.0.0.1:$SGLANG_PORT/v1/models >/dev/null && exit 0; sleep 2; done; exit 1"
docker exec sglang_bbuf bash -lc "python3 - <<'PY'
import json, urllib.request
payload = {
  'text': '用两句话解释 CUDA graph 和 eager mode 的区别。',
  'sampling_params': {'temperature': 0.0, 'max_new_tokens': 12},
  'stream': False,
}
req = urllib.request.Request(
    'http://127.0.0.1:$SGLANG_PORT/generate',
    data=json.dumps(payload).encode(),
    headers={'Content-Type': 'application/json'},
)
with urllib.request.urlopen(req, timeout=600) as resp:
    body = json.loads(resp.read().decode())
print(body.get('text', '')[:240])
PY"
docker exec sglang_bbuf bash -lc "cd '$SCRIPT_DIR' && python3 analyze_llm_torch_profile.py --framework sglang --url http://127.0.0.1:$SGLANG_PORT --output-dir '$SGLANG_PROFILE_ROOT' --num-steps 5 --probe-requests 1 --profile-by-stage > '$SGLANG_ANALYSIS'"
docker exec sglang_bbuf bash -lc "pkill -f 'sglang.launch_server.*--port $SGLANG_PORT' >/dev/null 2>&1 || true"

echo "[2/6] vLLM formal"
HF_TOKEN="$HF_TOKEN" HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  "$SCRIPT_DIR/run_vllm_torch_profile_host.sh" \
  --model "$MODEL" \
  --run-dir "$VLLM_FORMAL_DIR" \
  --port "$VLLM_FORMAL_PORT" \
  --gpu "$VLLM_GPU" \
  --trust-remote-code

echo "[3/6] vLLM mapping"
HF_TOKEN="$HF_TOKEN" HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  "$SCRIPT_DIR/run_vllm_torch_profile_host.sh" \
  --model "$MODEL" \
  --run-dir "$VLLM_MAPPING_DIR" \
  --port "$VLLM_MAPPING_PORT" \
  --gpu "$VLLM_GPU" \
  --trust-remote-code \
  --enforce-eager

echo "[4/6] vLLM mapping-formal analysis"
docker exec sglang_bbuf bash -lc "cd '$SCRIPT_DIR' && python3 analyze_llm_torch_profile.py --framework vllm --mapping-input '$VLLM_MAPPING_DIR' --formal-input '$VLLM_FORMAL_DIR' > '$VLLM_ANALYSIS'"

echo "[5/6] TensorRT-LLM formal + mapping captures"
HF_TOKEN="$HF_TOKEN" HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  "$SCRIPT_DIR/run_trtllm_pytorch_profile_host.sh" \
  --model "$MODEL" \
  --run-dir "$TRT_FORMAL_DIR" \
  --stage prefill \
  --port "$TRT_FORMAL_PREFILL_PORT" \
  --gpu "$TRT_FORMAL_GPU" \
  --override-py-executor /data/bbuf/validate/unified_llm_profiler_skill/overrides/trtllm/py_executor_with_stack.py \
  --trust-remote-code
HF_TOKEN="$HF_TOKEN" HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  "$SCRIPT_DIR/run_trtllm_pytorch_profile_host.sh" \
  --model "$MODEL" \
  --run-dir "$TRT_FORMAL_DIR" \
  --stage decode \
  --port "$TRT_FORMAL_DECODE_PORT" \
  --gpu "$TRT_FORMAL_GPU" \
  --override-py-executor /data/bbuf/validate/unified_llm_profiler_skill/overrides/trtllm/py_executor_with_stack.py \
  --trust-remote-code
HF_TOKEN="$HF_TOKEN" HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  "$SCRIPT_DIR/run_trtllm_pytorch_profile_host.sh" \
  --model "$MODEL" \
  --run-dir "$TRT_MAPPING_DIR" \
  --stage prefill \
  --port "$TRT_MAPPING_PREFILL_PORT" \
  --gpu "$TRT_MAPPING_GPU" \
  --override-py-executor /data/bbuf/validate/unified_llm_profiler_skill/overrides/trtllm/py_executor_with_stack.py \
  --disable-cudagraph \
  --trust-remote-code
HF_TOKEN="$HF_TOKEN" HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  "$SCRIPT_DIR/run_trtllm_pytorch_profile_host.sh" \
  --model "$MODEL" \
  --run-dir "$TRT_MAPPING_DIR" \
  --stage decode \
  --port "$TRT_MAPPING_DECODE_PORT" \
  --gpu "$TRT_MAPPING_GPU" \
  --override-py-executor /data/bbuf/validate/unified_llm_profiler_skill/overrides/trtllm/py_executor_with_stack.py \
  --disable-cudagraph \
  --trust-remote-code

echo "[6/6] TensorRT-LLM mapping-formal analysis"
docker exec sglang_bbuf bash -lc "cd '$SCRIPT_DIR' && python3 analyze_llm_torch_profile.py --framework trtllm --mapping-input '$TRT_MAPPING_DIR' --formal-input '$TRT_FORMAL_DIR' > '$TRT_ANALYSIS'"

echo "MODEL_ROOT=$MODEL_ROOT"
echo "ANALYSIS_SGLANG=$SGLANG_ANALYSIS"
echo "ANALYSIS_VLLM=$VLLM_ANALYSIS"
echo "ANALYSIS_TRTLLM=$TRT_ANALYSIS"
