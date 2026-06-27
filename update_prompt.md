# AI-Infra-Auto-Driven-SKILLS Refresh Prompt

Use this prompt when refreshing PR 72 or a successor PR for
`BBuf/AI-Infra-Auto-Driven-SKILLS`.

目标：把仓库里的 skills、model PR history、open PR watch、benchmark/profiler
说明和真实验证证据更新到当前日期，避免过期 SHA、幻觉式框架支持、以及未验证的
benchmark/profiler 结论。完成后提交并推送到当前 PR 分支。

## 固定机器

Use the B200 machine:

```bash
ssh -i ~/.ssh/id_ed25519 bbuf@216.114.73.196
# or
radix shell cirrascale-gpuc5a6
```

Main container:

```bash
sudo docker exec -it sglang_bbuf bash
```

MiniMax-M3-only container:

```bash
sudo docker start sglang_m3
sudo docker exec -it sglang_m3 bash
```

每做完一个模型验证，都要 kill server、确认 `nvidia-smi` 显存回到 0，再做下一个。
MiniMax-M3 验证结束后如果是本轮启动的 `sglang_m3`，要停止容器。

## 必做步骤

1. 在本地确认分支、PR、dirty 状态，不要覆盖用户未要求回退的改动。
2. 读取相关 skill 说明：`model-pr-history-knowledge`、`llm-serving-auto-benchmark`、
   `llm-torch-profiler-analysis`、以及当前 B200/Radix 机器 skill。
3. 用 live upstream state 更新证据锚点：

   ```bash
   git ls-remote https://github.com/sgl-project/sglang.git refs/heads/main
   git ls-remote https://github.com/vllm-project/vllm.git refs/heads/main
   git ls-remote https://github.com/NVIDIA/TensorRT-LLM.git refs/heads/main
   git ls-remote https://github.com/lightseekorg/tokenspeed.git refs/heads/main
   ```

4. 刷新 SGLang/vLLM merged PR history：

   ```bash
   SGLANG_PR_HISTORY_ROOT=/tmp/model-history-upstreams/sglang \
   VLLM_PR_HISTORY_ROOT=/tmp/model-history-upstreams/vllm \
   MODEL_PR_HISTORY_CACHE=/tmp/model_pr_history_git_trace_cache_v5.json \
   python3 tools/rebuild_model_pr_history_from_git.py
   ```

   如果 SGLang/vLLM head 又变化，先 fetch/fast-forward mirror，再重跑。

5. 刷新 open PR watch：

   ```bash
   python3 tools/check_open_pr_watch.py --format markdown \
     --output model-pr-optimization-history/open-pr-watch.md
   ```

   如果 GitHub rate limit 触发，脚本应走匿名 REST fallback；如果所有 repo 都失败，
   不允许写空报告。

6. 对手写 TensorRT-LLM/TokenSpeed history 页做 head refresh，但不要把未人工读 diff
   的 open PR 或新 merged PR 写成已审计卡片。
7. 做 targeted stale scan，至少覆盖旧日期标题和旧 head：

   ```bash
   rg -n '2026-..-.. Latest Source Scan|2026-..-.. 最新源码扫描|OLD_SHA_1|OLD_SHA_2' \
     model-pr-optimization-history skills README.md tests tools
   ```

8. 在 B200 `sglang_bbuf` 里建立 artifact root，例如：

   ```bash
   ART=/data/bbuf/ai_infra_skills_pr72_$(date +%Y%m%d)
   mkdir -p "$ART"/{help,serving,profiler,logs,minimax_m3}
   ```

   记录 Python、torch、CUDA、SGLang 版本，保存 server/benchmark `--help`。
   如果 vLLM/TensorRT-LLM/TokenSpeed CLI 缺失，只记录 environment gap，
   不要说该框架不支持。

9. 用 5 个不同规模模型做 SGLang serving smoke，每次单独启动/benchmark/kill：

   - `Qwen/Qwen2.5-0.5B-Instruct`
   - `Qwen/Qwen2.5-1.5B-Instruct`
   - `Qwen/Qwen2.5-3B-Instruct`
   - `Qwen/Qwen2.5-7B-Instruct`
   - `Qwen/Qwen3-8B`

   固定单卡 B200、小 workload 即可，目标是验证 skill 命令、OpenAI path、benchmark
   path 和清理流程，不要把它包装成公平跨框架性能结论。

10. MiniMax-M3 只在 `sglang_m3` 容器验证。优先跑仓库里和 MiniMax-M3 loader/cache/JIT
    相关的 unit/JIT smoke；只有环境已经满足官方 2x4 B200 disagg recipe 时，才做完整
    serving。
11. 跑 `llm-torch-profiler-analysis` live capture，至少对一个小 SGLang 模型抓
    prefill/decode 分离 trace，并确认输出 kernel、overlap、fuse 三张表。
12. 跑一个很小的 `ncu --set basic` CUDA matmul smoke，保存 `.ncu-rep` 和 log。
13. 把当前 repo 快照同步到远端临时目录，远端跑：

    ```bash
    python3 -m pytest -q tests/test_model_pr_dossier_quality.py \
      tests/test_open_pr_watch.py tests/test_llm_serving_cookbook_configs.py
    python3 skills/llm-serving-auto-benchmark/scripts/validate_cookbook_configs.py \
      --help-dir "$ART/help" \
      skills/llm-serving-auto-benchmark/configs/cookbook-llm/*.yaml
    ```

14. 本地跑全量 pytest、py_compile/stale scan、必要的 query smoke。
15. 更新 README、相关 skill 文档和 `model-pr-optimization-history/README.md`，
    只写真实验证过的结论，明确环境 gap。
16. 清理远端临时 repo、kill 任何 server、确认 8 张 B200 都回到 0 MiB。停止本轮启动的
    `sglang_m3`。
17. 提交、推送当前分支，更新 PR 72 描述或评论，等待 GitHub checks。若 CI 失败，
    读取日志并修到通过。

## 完成标准

- 没有旧 head/旧 scan 标题残留。
- `open-pr-watch.md` 不是 rate-limit 造成的空报告。
- 5 个 SGLang 模型 smoke 都有 artifact，且每个模型后 GPU 清干净。
- MiniMax-M3 有独立容器验证记录。
- profiler live capture 和 ncu smoke 都有 artifact。
- 本地和远端测试都通过。
- PR 已推送，CI 状态已核对。
