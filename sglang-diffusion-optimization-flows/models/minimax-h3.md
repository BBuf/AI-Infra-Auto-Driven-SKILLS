# MiniMax-H3 优化 flow（第一优先级）

源码基线包含合并 PR `sgl-project/sglang#33275`。根 checkpoint 同时服务 T2VA、
FL2VA、Ref2VA；用 `--model-variant=fl2va|ref2va` 选分区，不要下载或传子目录。
先读 [`../common/execution-contract.md`](../common/execution-contract.md)。

## 下载与 baseline

```bash
# Hugging Face（官方或国内镜像二选一）
MODEL_ID=MiniMaxAI/MiniMax-H3
export HF_ENDPOINT=https://hf-mirror.com  # 国内环境可选
MODEL_DIR=$(hf download "$MODEL_ID")      # preset 与手工命令都复用 HF cache

# ModelScope 是源码已登记的别名：
# MODEL_DIR=/data/models/MiniMax-H3
# modelscope download --model MiniMax/MiniMax-H3 --local_dir "$MODEL_DIR"
# export SGLANG_USE_MODELSCOPE=true

export CUDA_VISIBLE_DEVICES=$(python3 "$ENV_PY" print-idle-gpus --count 4)
PYTHONPATH=python python3 "$BENCH_PY" \
  --model minimax-h3-t2va --label baseline --output-dir "$BENCH_DIR"
```

preset 固定 seed 1101、1344x768 resolved canvas、5 秒/124 帧/24 fps、50 个联合
音视频 step、4 GPU TP2 + Ulysses2、`performance-mode=speed`。它自动在
`$BENCH_DIR/generated_configs/` 写入 task/conditions/target/flow shifts。

该 preset 来自 companion skills PR `sgl-project/sglang#33282`。若尚未合并，先保存
以下内容到 `$BENCH_DIR/generated_configs/minimax-h3-t2va.json`：

```json
{
  "task": "t2va",
  "conditions": [],
  "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0},
  "num_inference_steps": 50,
  "flow_shift": 12.0,
  "audio_flow_shift": 3.0
}
```

再用下载后的根目录执行：

```bash
sglang generate --backend sglang --model-path "$MODEL_DIR" \
  --model-variant fl2va \
  --config "$BENCH_DIR/generated_configs/minimax-h3-t2va.json" \
  --prompt "At night, while their owner sleeps in a bedroom, three cats march in loudly playing tiny brass instruments, then abruptly file out." \
  --seed 1101 --num-gpus 4 --tp-size 2 --ulysses-degree 2 \
  --performance-mode speed --enable-torch-compile=false --save-output --warmup \
  --perf-dump-path "$BENCH_DIR/minimax-h3-t2va-baseline.json"
```

关键合同：eager BF16/FP32 是唯一 lossless reference；当前 `torch.compile` 会改变
H3 数值输出。禁止 Ring、CFG parallel、SageAttention；released video VAE 只允许
overlapping tiled decode，spatial/spatial-shard/patch mode 会被拒绝。

## FL2VA / Ref2VA smoke

T2VA baseline 通过后复制 JSON，只改 `task`、`conditions` 和 variant。所有 `uri`
必须是可读的绝对 `file:///...` 或稳定 HTTPS 地址。

FL2VA 仍加载 `--model-variant=fl2va`：

```json
{
  "task": "fl2va",
  "conditions": [
    {"type": "image", "uri": "file:///ABS/first.png", "role": "keyframe", "frame_index": 0},
    {"type": "image", "uri": "file:///ABS/last.png", "role": "keyframe", "frame_index": -1}
  ],
  "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0},
  "num_inference_steps": 50, "flow_shift": 12.0, "audio_flow_shift": 3.0
}
```

Ref2VA 必须改为 `--model-variant=ref2va`；最小条件为
`{"type":"image","uri":"file:///ABS/reference.png","role":"reference"}`。扩展
覆盖 video、audio、video_audio reference 及 `start_time_seconds`。命令复用上面的
`sglang generate`，只替换 `--model-variant` 和 `--config`。released duration 范围是
4–15 秒；任务/partition 不匹配必须 fail closed。

## 拆分并行执行

在固定 baseline 后并行研究，但一次只合并一个可归因改动：

1. DiT：执行 [`../components/dit-attention.md`](../components/dit-attention.md)，
   重点检查源码已有 indexed scale/shift/gate、head-dim 128 fused QK norm + RoPE
   （RoPE dim 96、NeoX、norm-before-RoPE）、packed Ulysses QKV、`usp_merge_heads`、
   2-rank IPC A2A、batched TP AdaLN 和 final projection dead-row removal。
2. Video VAE：执行
   [`../components/vae-minimax-h3-video.md`](../components/vae-minimax-h3-video.md)。
3. Audio VAE：执行 [`../components/vae-audio.md`](../components/vae-audio.md)，只选
   `MiniMaxH3AudioVAE` 子项；从聚合 decoding stage 中剥离 audio/video/mux。
4. Qwen3-VL 条件 encoder：执行 [`../components/encoders.md`](../components/encoders.md)。
5. 4/8 卡 topology：执行
   [`../components/distributed-runtime.md`](../components/distributed-runtime.md)。
6. scheduler/mux：执行
   [`../components/schedulers-postprocess.md`](../components/schedulers-postprocess.md)，
   固定 `MiniMaxH3EulerAncestralEta0SchedulerAdapter` 与 H.264/AAC delivery contract。

建议先后顺序：DiT stage profile → H3 fast-path guard → video VAE → audio VAE →
encoder/通信。Cache-DiT high/medium/low 与 online FP8 是近似质量支线，不能与 eager
lossless 结果混表。

## 验收与结束

- 固定 prompt/seed/target/steps/shifts/variant/topology；分别验 T2VA、FL2VA、Ref2VA。
- video：帧数/时间戳对齐，逐帧 PSNR/SSIM 及 worst frame，无 seam/flicker。
- audio：32 kHz stereo 的 channel/sample count、waveform error、对齐频谱，并验 A/V
  duration/sync；输出仍是 24fps H.264 + AAC stereo。
- kernel fast path 跑相应 H3 exactness/unit tests；E2E 容差不能掩盖 exact test 失败。
- 4×H100/H200 首测 TP2+Ulysses2；另测 Ulysses4。B200/B300 可测 8×Ulysses8；
  4×B200 FSDP 是容量路线。每个 topology 单独报告 latency/memory/scaling。

只有 joint video/audio correctness、目标组件与 E2E 性能都过门槛才 `accepted`。
