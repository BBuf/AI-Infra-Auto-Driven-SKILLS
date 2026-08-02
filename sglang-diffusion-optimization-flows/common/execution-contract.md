# 共享执行协议

所有模型 flow 都继承本协议。模型文件只覆盖这里明确写出的差异。

## 0. 固定实验合同

开始前记录：

- SGLang commit、patch diff、CUDA/PyTorch/FlashAttention/FlashInfer 版本；
- GPU 型号、数量、拓扑和空闲显存；
- checkpoint revision 或本地 snapshot hash；
- model、prompt、输入图、seed、分辨率、帧数、步数、dtype、并行参数；
- compile/offload/cache/quant/attention backend 是否开启。

基线与候选必须使用同一合同。一次只改变一个变量。

## 1. 环境与 native backend gate

在 SGLang diffusion 容器内执行：

```bash
ENV_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/diffusion_skill_env.py
BENCH_PY=python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py
ROOT=$(python3 "$ENV_PY" print-root)
cd "$ROOT"
python3 "$ENV_PY" check-write-access

export FLASHINFER_DISABLE_VERSION_CHECK=1
# 不在公共协议里固定卡数。模型 flow 会按 preset/config 选择 1/2/4/8 张空闲卡；
# 如果手工跑命令，再显式设置 CUDA_VISIBLE_DEVICES。

ASSET_DIR=$(python3 "$ENV_PY" print-assets-dir --mkdir)
BENCH_DIR=$(python3 "$ENV_PY" print-output-dir --kind benchmarks --mkdir)
PROFILE_DIR=$(python3 "$ENV_PY" print-output-dir --kind profiles --mkdir)
```

FLUX 等 gated repo 先 `export HF_TOKEN=...`。任何 native 基准命令都显式使用
`--backend sglang` 或 benchmark helper。日志中出现以下任一字符串就立即停止，
丢弃这次数字与 trace：

```text
Falling back to diffusers backend
Using diffusers backend
Loaded diffusers pipeline
```

## 2. 下载 checkpoint 和输入资产

优先下载固定 snapshot，之后用本地路径跑，避免网络抖动与远端更新污染实验。

```bash
MODEL_ID=<huggingface-repo>

# benchmark helper 的 preset 内置 registry ID；预填 HF cache，helper 会直接复用
MODEL_DIR=$(hf download "$MODEL_ID")

# 国内 HF 镜像；gated repo 仍需 HF_TOKEN
export HF_ENDPOINT=https://hf-mirror.com
MODEL_DIR=$(hf download "$MODEL_ID")

# 需要稳定本地目录或离线搬运时：
MODEL_DIR=/data/models/<stable-local-name>
hf download "$MODEL_ID" --local-dir "$MODEL_DIR"
# 此时用模型文件的手工 sglang generate 命令并传 "$MODEL_DIR"；preset 本身仍用 ID。
```

ModelScope 只在确认存在对应仓库后使用，不要假设 HF ID 一定同名：

```bash
MODELSCOPE_MODEL_ID=<verified-modelscope-id>
modelscope download --model "$MODELSCOPE_MODEL_ID" --local_dir "$MODEL_DIR"
```

ModelScope 下载的本地目录同样传给手工 `sglang generate`。只有源码登记了相应
ModelScope alias 的路径才设置 `SGLANG_USE_MODELSCOPE=true` 直接按 ID 加载。

常用输入图：

```bash
wget -O "${ASSET_DIR}/cat.png" \
  https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png
wget -O "${ASSET_DIR}/mova_single_person.jpg" \
  https://github.com/OpenMOSS/MOVA/raw/main/assets/single_person.jpg
```

下载后保存 `config.json`/`model_index.json`、safetensors 文件列表、总字节数和
revision。禁止在 baseline 与 candidate 之间静默更新 checkpoint。

## 3. 建立 baseline

有 preset 时优先使用它：

```bash
PYTHONPATH=python python3 "$BENCH_PY" --validate-nightly-alignment
PYTHONPATH=python python3 "$BENCH_PY" \
  --model <preset> --label baseline --output-dir "$BENCH_DIR"
```

没有 preset 时用模型文件中的 `sglang generate` 命令。至少保留：

- perf dump JSON；
- 一张图/一段视频/一段音频/一个 mesh 作为固定 reference；
- denoise、VAE encode/decode、text encoder 等 stage latency；
- mean E2E latency、peak GPU memory；
- 完整命令与日志。

除模型文件明确例外外，`torch.compile` 是默认优化下界：正式 baseline 使用
`--enable-torch-compile --warmup`。为理解图边界，可以额外跑一次
`--no-torch-compile`，但不能把它混入正式前后对比。

MiniMax-H3 是当前例外：源码合同明确 compile 会改变数值输出，因此正式正确性
reference 必须是 eager。compile 结果只能作为独立实验，不能冒充 lossless baseline。

## 4. 先定位组件，再定位 kernel

先收集一次全阶段 trace：

```bash
export SGLANG_DIFFUSION_TORCH_PROFILER_DIR="${PROFILE_DIR}/torch"
sglang generate --backend sglang --model-path "$MODEL_DIR" \
  <固定 workload 参数> \
  --enable-torch-compile --warmup --profile --profile-all-stages
```

按下列顺序归因：

1. E2E stage：text/image encoder、denoise、VAE/audio/mesh decode、后处理、通信；
2. 模型层：attention、MLP、norm/modulation、conv/resblock、scheduler；
3. CUDA kernel：总时长、调用次数、单次时长和 shape；
4. 数据移动：layout conversion、pad/reshape/pack/scatter、H2D、all-to-all。

如果组件低于 E2E 的 3%，先记录为阴性结论；除非改动几乎零风险，否则不要立项。

## 5. 先排查已有 fast path

新 kernel 前必须检查当前源码已有实现：

- fused scale/shift/gate 与 Qwen modulation；
- fused QK norm、QK norm + RoPE；
- Z-Image bf16 Triton norm/tanh residual；
- LTX split RoPE、residual-gate add；
- Hunyuan/LTX upsampler GroupNorm + SiLU；
- varlen USP pack/scatter；
- FLUX Nunchaku GELU MLP 与 NVFP4 packed QKV；
- SANA packed QKV/KV；
- MiniMax-H3 indexed scale/shift/gate、head-dim 128 fused QK norm + RoPE；
- MiniMax-H3 packed Ulysses QKV/USP relayout、2-rank IPC A2A、batched TP AdaLN；
- Ulysses/USP、CFG parallel、async all-to-all、compile compute/comm reorder。

已有 fast path 没触发时，先修 backend、checkpoint mapping、shape guard、dtype、
contiguity 或配置，不要再写一份同类 kernel。

## 6. 两条候选路线

### A. 数学/语义 rewrite 与 fuse

优先级：消除完整中间张量的等价 rewrite > 跨算子 fuse > kernel 内少量访存。
重点检查：

- upsample + conv 是否可等价为 transposed conv；
- 单帧部署是否能把 causal Conv3d 退化为 Conv2d；
- 线性投影能否通过 attention 的代数性质折叠；
- bias/residual 是否能直接进入 norm 统计；
- QKV/KV/MLP projection 是否可以加载期合并；
- tiled/streaming/CFG 语义是否允许更强的专用 dispatch。

所有 rewrite 都必须有公式或等价说明、fail-closed guard、原实现 fallback、
安装/回滚路径和 checkpoint/shape 指纹。

### B. 真正的 kernel 盲区

只有 profile 证明热点且已有 fast path 不适用时才进入：

1. 用 KernelWiki 查询同架构、同 dtype、同 shape 的上游实现和 PR；
2. 建独立 harness，真实 tensor/shape，编译带 `-lineinfo`；
3. 每次 NCU run 使用独立 `profile/<run>/` 目录；
4. 收集 `--set full`（含 PM sampling）与
   `--set source --section SourceCounters`；
5. 用具体指标判定 occupancy、memory、stall、tensor core、tail、load balance；
6. 只实现证据支持的最高收益改动。

在 B200/SM100 上，KernelWiki 用于查询 tcgen05/TMEM/CLC/TMA/PDL、FA4、
CuTe DSL、Triton 与已合并 PR。NCU 负责诊断，不允许用“看起来 memory-bound”
代替 counter 数字。

## 7. 正确性门槛

不要要求 bit-identical，也不要用“PSNR 必须严格不降/MSE 必须严格不增”卡死
BF16 舍入顺序变化。默认非量化候选采用以下合理容差；模型文件可收紧：

- component cosine similarity：VAE `>= 0.999`，DiT `>= 0.995`，encoder `>= 0.98`；
- decoder 输出归一到 `[0, 1]` 后，相对 baseline 的 MSE `<= 1e-4`；
- ImageNet-val 或固定重建集：PSNR 下降 `<= 0.10 dB`，SSIM 下降 `<= 0.002`；
- 端到端固定 seed：无 NaN/Inf、无明显 artifact，图像/视频/音频/mesh 都可打开；
- 音视频模型必须分别校验视频帧与 waveform/sample-rate/duration，不能只看 mp4 可播放；
- 100 次短 soak 无随机 fallback、累积 cache 污染或显存持续增长。

`1e-5` 量级的 MSE 本身不是放弃高价值 fuse 的理由。若变化超过默认容差，
必须扩大样本、定位误差来源，并显式把它归类为 lossy；不能悄悄放宽门槛。
量化、Cache-DiT、稀疏/近似 attention 和减少采样步数使用单独 quality budget，
不得与 lossless baseline 混在一张速度结论里。

## 8. 性能门槛

- component microbenchmark：至少 20 次 warmup + 100 次计时；
- 昂贵 E2E：至少 3 个独立进程重复，报告 mean 和 median；
- 同步 GPU，排除 compile/warmup，保存原始样本；
- 报告 denoise、目标组件、mean E2E、峰值显存和多 GPU scaling；
- 默认要求目标组件或 mean E2E 至少提升 3%，且大于运行方差；
- 优化一个组件导致 E2E、显存或其他 stage 明显回退时，不接受局部胜利。

前后 perf dump：

```bash
python3 python/sglang/multimodal_gen/benchmarks/compare_perf.py \
  "${BENCH_DIR}/baseline.json" "${BENCH_DIR}/candidate.json"
```

## 9. 一轮的结束条件

每轮只允许以下三种结论：

- `accepted`：正确性和性能都通过，保留 patch；
- `rejected`：证据表明无收益/有回归，完整回滚并记录阴性结果；
- `next-profile`：信息不足，回到 stage/kernel profile，不继续猜。

最终交付包含 source/checkpoint lock、命令、diff、perf dumps、trace/NCU report、
正确性表、速度表、已知限制、fallback/rollback 测试和下一候选。不要只提交一张
“更快”的截图。
