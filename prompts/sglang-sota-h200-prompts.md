# H200 prompts

## 1 GPU

### nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
```text
使用 sglang-sota-humanize-loop skill。
model_id: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 1x NVIDIA H200
minimum_gpu_count: 1
precision_quantization: FP8
initial_deployment: SGLang TP=1
要求: 远端使用 ion8-h200 或 ion9-h200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 启动 Humanize/RLCR loop 时，必须先 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: Humanize/RLCR loop 必须在本地 Codex 会话启动后启动；远端机器只作为执行和验证环境，不要在远端机器上启动或托管这个 loop。
要求: 每次 benchmark/profile 前必须确认这 1 张 H200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果那时没有足够的空闲 H200 继续完成当前任务，就等待半小时；如果半小时内资源仍然不足，再停止并报告 blocker。
要求: 只使用 1 张 H200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 1 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；重点看 MoE、attention/backend、decode throughput。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/h200_nemotron3_nano_30b_a3b_fp8_sota_humanize
```

### google/gemma-4-26B-A4B-it
```text
使用 sglang-sota-humanize-loop skill。
model_id: google/gemma-4-26B-A4B-it
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 1x NVIDIA H200
minimum_gpu_count: 1
precision_quantization: BF16
initial_deployment: SGLang TP=1
要求: 远端使用 ion8-h200 或 ion9-h200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 启动 Humanize/RLCR loop 时，必须先 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: Humanize/RLCR loop 必须在本地 Codex 会话启动后启动；远端机器只作为执行和验证环境，不要在远端机器上启动或托管这个 loop。
要求: 每次 benchmark/profile 前必须确认这 1 张 H200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果那时没有足够的空闲 H200 继续完成当前任务，就等待半小时；如果半小时内资源仍然不足，再停止并报告 blocker。
要求: 只使用 1 张 H200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 1 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/h200_gemma4_26b_a4b_it_sota_humanize
```

## 2 GPU

### Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
```text
使用 sglang-sota-humanize-loop skill。
model_id: Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 2x NVIDIA H200
minimum_gpu_count: 2
precision_quantization: FP8
initial_deployment: SGLang TP=2
要求: 远端使用 ion8-h200 或 ion9-h200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 启动 Humanize/RLCR loop 时，必须先 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: Humanize/RLCR loop 必须在本地 Codex 会话启动后启动；远端机器只作为执行和验证环境，不要在远端机器上启动或托管这个 loop。
要求: 每次 benchmark/profile 前必须确认这 2 张 H200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果那时没有足够的空闲 H200 继续完成当前任务，就等待半小时；如果半小时内资源仍然不足，再停止并报告 blocker。
要求: 只使用 2 张 H200；不要测试 4 卡或 8 卡部署。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 2 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；重点看 hybrid attention、Mamba/GDN、radix cache、target verify、CUDA graph。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/h200_qwen3_next_80b_a3b_instruct_fp8_sota_humanize
```

### Qwen/Qwen3-Next-80B-A3B-Thinking-FP8
```text
使用 sglang-sota-humanize-loop skill。
model_id: Qwen/Qwen3-Next-80B-A3B-Thinking-FP8
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 2x NVIDIA H200
minimum_gpu_count: 2
precision_quantization: FP8
initial_deployment: SGLang TP=2
要求: 远端使用 ion8-h200 或 ion9-h200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 启动 Humanize/RLCR loop 时，必须先 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: Humanize/RLCR loop 必须在本地 Codex 会话启动后启动；远端机器只作为执行和验证环境，不要在远端机器上启动或托管这个 loop。
要求: 每次 benchmark/profile 前必须确认这 2 张 H200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果那时没有足够的空闲 H200 继续完成当前任务，就等待半小时；如果半小时内资源仍然不足，再停止并报告 blocker。
要求: 只使用 2 张 H200；不要测试 4 卡或 8 卡部署。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 2 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；重点看 thinking workload 下的 Mamba/GDN、decode latency、target verify。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/h200_qwen3_next_80b_a3b_thinking_fp8_sota_humanize
```

### google/gemma-4-31B-it
```text
使用 sglang-sota-humanize-loop skill。
model_id: google/gemma-4-31B-it
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 2x NVIDIA H200
minimum_gpu_count: 2
precision_quantization: BF16
initial_deployment: SGLang TP=2
要求: 远端使用 ion8-h200 或 ion9-h200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 启动 Humanize/RLCR loop 时，必须先 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: Humanize/RLCR loop 必须在本地 Codex 会话启动后启动；远端机器只作为执行和验证环境，不要在远端机器上启动或托管这个 loop。
要求: 每次 benchmark/profile 前必须确认这 2 张 H200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果那时没有足够的空闲 H200 继续完成当前任务，就等待半小时；如果半小时内资源仍然不足，再停止并报告 blocker。
要求: 只使用 2 张 H200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 2 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/h200_gemma4_31b_it_sota_humanize
```

## 4 GPU

### nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
```text
使用 sglang-sota-humanize-loop skill。
model_id: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 4x NVIDIA H200
minimum_gpu_count: 4
precision_quantization: BF16
initial_deployment: SGLang TP=4
要求: 远端使用 ion8-h200 或 ion9-h200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 启动 Humanize/RLCR loop 时，必须先 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: Humanize/RLCR loop 必须在本地 Codex 会话启动后启动；远端机器只作为执行和验证环境，不要在远端机器上启动或托管这个 loop。
要求: 每次 benchmark/profile 前必须确认这 4 张 H200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果那时没有足够的空闲 H200 继续完成当前任务，就等待半小时；如果半小时内资源仍然不足，再停止并报告 blocker。
要求: 只使用 4 张 H200；不要测试 8 卡部署，除非 4 卡实测 OOM 并记录 artifact。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 4 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；重点看 MoE、attention/backend、memory pressure、decode throughput。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/h200_nemotron3_super_120b_a12b_bf16_sota_humanize
```

### stepfun-ai/Step-3.5-Flash-FP8
```text
使用 sglang-sota-humanize-loop skill。
model_id: stepfun-ai/Step-3.5-Flash-FP8
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 4x NVIDIA H200
minimum_gpu_count: 4
precision_quantization: FP8
initial_deployment: SGLang TP=4
要求: 远端使用 ion8-h200 或 ion9-h200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 启动 Humanize/RLCR loop 时，必须先 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: Humanize/RLCR loop 必须在本地 Codex 会话启动后启动；远端机器只作为执行和验证环境，不要在远端机器上启动或托管这个 loop。
要求: 每次 benchmark/profile 前必须确认这 4 张 H200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果那时没有足够的空闲 H200 继续完成当前任务，就等待半小时；如果半小时内资源仍然不足，再停止并报告 blocker。
要求: 只使用 4 张 H200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 4 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/h200_step35_flash_fp8_sota_humanize
```

### meta-llama/Llama-3.1-70B-Instruct
```text
使用 sglang-sota-humanize-loop skill。
model_id: meta-llama/Llama-3.1-70B-Instruct
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 4x NVIDIA H200
minimum_gpu_count: 4
precision_quantization: BF16
initial_deployment: SGLang TP=4
要求: 远端使用 ion8-h200 或 ion9-h200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 启动 Humanize/RLCR loop 时，必须先 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: Humanize/RLCR loop 必须在本地 Codex 会话启动后启动；远端机器只作为执行和验证环境，不要在远端机器上启动或托管这个 loop。
要求: 每次 benchmark/profile 前必须确认这 4 张 H200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果那时没有足够的空闲 H200 继续完成当前任务，就等待半小时；如果半小时内资源仍然不足，再停止并报告 blocker。
要求: 只使用 4 张 H200；不要测试 TP8。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 4 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/h200_llama31_70b_instruct_sota_humanize
```

### mistralai/Mistral-Large-Instruct-2407
```text
使用 sglang-sota-humanize-loop skill。
model_id: mistralai/Mistral-Large-Instruct-2407
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 4x NVIDIA H200
minimum_gpu_count: 4
precision_quantization: BF16
initial_deployment: SGLang TP=4
要求: 远端使用 ion8-h200 或 ion9-h200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 启动 Humanize/RLCR loop 时，必须先 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: Humanize/RLCR loop 必须在本地 Codex 会话启动后启动；远端机器只作为执行和验证环境，不要在远端机器上启动或托管这个 loop。
要求: 每次 benchmark/profile 前必须确认这 4 张 H200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果那时没有足够的空闲 H200 继续完成当前任务，就等待半小时；如果半小时内资源仍然不足，再停止并报告 blocker。
要求: 只使用 4 张 H200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 4 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/h200_mistral_large_instruct_2407_sota_humanize
```

## 8 GPU

### deepseek-ai/DeepSeek-V3.2
```text
使用 sglang-sota-humanize-loop skill。
model_id: deepseek-ai/DeepSeek-V3.2
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA H200
minimum_gpu_count: 8
precision_quantization: FP8
initial_deployment: SGLang TP=8；DP/EP/MTP 只能在同样 8 卡预算内搜索
要求: 远端使用 ion8-h200 或 ion9-h200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 启动 Humanize/RLCR loop 时，必须先 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: Humanize/RLCR loop 必须在本地 Codex 会话启动后启动；远端机器只作为执行和验证环境，不要在远端机器上启动或托管这个 loop。
要求: 每次 benchmark/profile 前必须确认这 8 张 H200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果那时没有足够的空闲 H200 继续完成当前任务，就等待半小时；如果半小时内资源仍然不足，再停止并报告 blocker。
要求: 只使用 8 张 H200；不要测试多机或更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；重点看 MLA、MoE/EP、DP attention、speculative decoding、overlap、memory/cache。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/h200_deepseek_v32_sota_humanize
```

### deepseek-ai/DeepSeek-R1-0528
```text
使用 sglang-sota-humanize-loop skill。
model_id: deepseek-ai/DeepSeek-R1-0528
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA H200
minimum_gpu_count: 8
precision_quantization: FP8
initial_deployment: SGLang TP=8；DP/EP/MTP 只能在同样 8 卡预算内搜索
要求: 远端使用 ion8-h200 或 ion9-h200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 启动 Humanize/RLCR loop 时，必须先 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: Humanize/RLCR loop 必须在本地 Codex 会话启动后启动；远端机器只作为执行和验证环境，不要在远端机器上启动或托管这个 loop。
要求: 每次 benchmark/profile 前必须确认这 8 张 H200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果那时没有足够的空闲 H200 继续完成当前任务，就等待半小时；如果半小时内资源仍然不足，再停止并报告 blocker。
要求: 只使用 8 张 H200；不要测试多机或更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；重点看 MLA、MoE/EP、DP attention、speculative decoding、overlap、memory/cache。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/h200_deepseek_r1_0528_sota_humanize
```

### Qwen/Qwen3.5-397B-A17B-FP8
```text
使用 sglang-sota-humanize-loop skill。
model_id: Qwen/Qwen3.5-397B-A17B-FP8
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA H200
minimum_gpu_count: 8
precision_quantization: FP8
initial_deployment: SGLang TP=8, EP=8；MTP 只能在同样 8 卡预算内搜索
要求: 远端使用 ion8-h200 或 ion9-h200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 启动 Humanize/RLCR loop 时，必须先 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: Humanize/RLCR loop 必须在本地 Codex 会话启动后启动；远端机器只作为执行和验证环境，不要在远端机器上启动或托管这个 loop。
要求: 每次 benchmark/profile 前必须确认这 8 张 H200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果那时没有足够的空闲 H200 继续完成当前任务，就等待半小时；如果半小时内资源仍然不足，再停止并报告 blocker。
要求: 只使用 8 张 H200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；重点看 MoE routing、EP、attention backend、MTP/speculative、long-context prefill/decode。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/h200_qwen35_397b_a17b_fp8_sota_humanize
```

### moonshotai/Kimi-K2-Instruct
```text
使用 sglang-sota-humanize-loop skill。
model_id: moonshotai/Kimi-K2-Instruct
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA H200
minimum_gpu_count: 8
precision_quantization: FP8
initial_deployment: SGLang TP=8；DP/EP 只能在同样 8 卡预算内搜索
要求: 远端使用 ion8-h200 或 ion9-h200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 启动 Humanize/RLCR loop 时，必须先 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: Humanize/RLCR loop 必须在本地 Codex 会话启动后启动；远端机器只作为执行和验证环境，不要在远端机器上启动或托管这个 loop。
要求: 每次 benchmark/profile 前必须确认这 8 张 H200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果那时没有足够的空闲 H200 继续完成当前任务，就等待半小时；如果半小时内资源仍然不足，再停止并报告 blocker。
要求: 只使用 8 张 H200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；重点看 MLA/MoE/DP-attention、long-context decode、framework parity。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/h200_kimi_k2_instruct_sota_humanize
```

### moonshotai/Kimi-K2.5
```text
使用 sglang-sota-humanize-loop skill。
model_id: moonshotai/Kimi-K2.5
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA H200
minimum_gpu_count: 8
precision_quantization: FP8
initial_deployment: SGLang TP=8；DP 只能在同样 8 卡预算内搜索
要求: 远端使用 ion8-h200 或 ion9-h200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 启动 Humanize/RLCR loop 时，必须先 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: Humanize/RLCR loop 必须在本地 Codex 会话启动后启动；远端机器只作为执行和验证环境，不要在远端机器上启动或托管这个 loop。
要求: 每次 benchmark/profile 前必须确认这 8 张 H200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果那时没有足够的空闲 H200 继续完成当前任务，就等待半小时；如果半小时内资源仍然不足，再停止并报告 blocker。
要求: 只使用 8 张 H200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；重点看 MLA/MoE/DP-attention、long-context decode、framework parity。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/h200_kimi_k25_sota_humanize
```

### moonshotai/Kimi-K2.6
```text
使用 sglang-sota-humanize-loop skill。
model_id: moonshotai/Kimi-K2.6
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA H200
minimum_gpu_count: 8
precision_quantization: INT4
initial_deployment: SGLang TP=8, --trust-remote-code；DP 只能在同样 8 卡预算内搜索
要求: 远端使用 ion8-h200 或 ion9-h200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 启动 Humanize/RLCR loop 时，必须先 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: Humanize/RLCR loop 必须在本地 Codex 会话启动后启动；远端机器只作为执行和验证环境，不要在远端机器上启动或托管这个 loop。
要求: 每次 benchmark/profile 前必须确认这 8 张 H200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果那时没有足够的空闲 H200 继续完成当前任务，就等待半小时；如果半小时内资源仍然不足，再停止并报告 blocker。
要求: 只使用 8 张 H200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；保留 trust-remote-code、multimodal、thinking、tool-calling 验证证据。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/h200_kimi_k26_sota_humanize
```

### MiniMaxAI/MiniMax-M2.7
```text
使用 sglang-sota-humanize-loop skill。
model_id: MiniMaxAI/MiniMax-M2.7
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA H200
minimum_gpu_count: 8
precision_quantization: FP8
initial_deployment: SGLang TP=8
要求: 远端使用 ion8-h200 或 ion9-h200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 启动 Humanize/RLCR loop 时，必须先 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: Humanize/RLCR loop 必须在本地 Codex 会话启动后启动；远端机器只作为执行和验证环境，不要在远端机器上启动或托管这个 loop。
要求: 每次 benchmark/profile 前必须确认这 8 张 H200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果那时没有足够的空闲 H200 继续完成当前任务，就等待半小时；如果半小时内资源仍然不足，再停止并报告 blocker。
要求: 只使用 8 张 H200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；重点看 MoE、attention、memory pressure、decode throughput。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/h200_minimax_m27_sota_humanize
```

### zai-org/GLM-4.6-FP8
```text
使用 sglang-sota-humanize-loop skill。
model_id: zai-org/GLM-4.6-FP8
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA H200
minimum_gpu_count: 8
precision_quantization: FP8
initial_deployment: SGLang TP=8；DP/EP/MTP 只能在同样 8 卡预算内搜索
要求: 远端使用 ion8-h200 或 ion9-h200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 启动 Humanize/RLCR loop 时，必须先 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: Humanize/RLCR loop 必须在本地 Codex 会话启动后启动；远端机器只作为执行和验证环境，不要在远端机器上启动或托管这个 loop。
要求: 每次 benchmark/profile 前必须确认这 8 张 H200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果那时没有足够的空闲 H200 继续完成当前任务，就等待半小时；如果半小时内资源仍然不足，再停止并报告 blocker。
要求: 只使用 8 张 H200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/h200_glm46_fp8_sota_humanize
```

### zai-org/GLM-5-FP8
```text
使用 sglang-sota-humanize-loop skill。
model_id: zai-org/GLM-5-FP8
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA H200
minimum_gpu_count: 8
precision_quantization: FP8
initial_deployment: SGLang TP=8；DP/MTP 只能在同样 8 卡预算内搜索
要求: 远端使用 ion8-h200 或 ion9-h200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 启动 Humanize/RLCR loop 时，必须先 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: Humanize/RLCR loop 必须在本地 Codex 会话启动后启动；远端机器只作为执行和验证环境，不要在远端机器上启动或托管这个 loop。
要求: 每次 benchmark/profile 前必须确认这 8 张 H200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果那时没有足够的空闲 H200 继续完成当前任务，就等待半小时；如果半小时内资源仍然不足，再停止并报告 blocker。
要求: 只使用 8 张 H200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；重点看 MTP/speculative、long-context behavior、framework parity。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/h200_glm5_fp8_sota_humanize
```

### openai/gpt-oss-120b
```text
使用 sglang-sota-humanize-loop skill。
model_id: openai/gpt-oss-120b
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA H200
minimum_gpu_count: 8
precision_quantization: MXFP4
initial_deployment: SGLang TP=8
要求: 远端使用 ion8-h200 或 ion9-h200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 启动 Humanize/RLCR loop 时，必须先 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: Humanize/RLCR loop 必须在本地 Codex 会话启动后启动；远端机器只作为执行和验证环境，不要在远端机器上启动或托管这个 loop。
要求: 每次 benchmark/profile 前必须确认这 8 张 H200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果那时没有足够的空闲 H200 继续完成当前任务，就等待半小时；如果半小时内资源仍然不足，再停止并报告 blocker。
要求: 只使用 8 张 H200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/h200_gpt_oss_120b_sota_humanize
```

### internlm/Intern-S1-FP8
```text
使用 sglang-sota-humanize-loop skill。
model_id: internlm/Intern-S1-FP8
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA H200
minimum_gpu_count: 8
precision_quantization: FP8
initial_deployment: SGLang TP=8, EP=2
要求: 远端使用 ion8-h200 或 ion9-h200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 启动 Humanize/RLCR loop 时，必须先 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: Humanize/RLCR loop 必须在本地 Codex 会话启动后启动；远端机器只作为执行和验证环境，不要在远端机器上启动或托管这个 loop。
要求: 每次 benchmark/profile 前必须确认这 8 张 H200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果那时没有足够的空闲 H200 继续完成当前任务，就等待半小时；如果半小时内资源仍然不足，再停止并报告 blocker。
要求: 只使用 8 张 H200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/h200_intern_s1_fp8_sota_humanize
```

### Qwen/Qwen3-235B-A22B-FP8
```text
使用 sglang-sota-humanize-loop skill。
model_id: Qwen/Qwen3-235B-A22B-FP8
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA H200
minimum_gpu_count: 8
precision_quantization: FP8
initial_deployment: SGLang TP=8, EP=2
要求: 远端使用 ion8-h200 或 ion9-h200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 启动 Humanize/RLCR loop 时，必须先 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: Humanize/RLCR loop 必须在本地 Codex 会话启动后启动；远端机器只作为执行和验证环境，不要在远端机器上启动或托管这个 loop。
要求: 每次 benchmark/profile 前必须确认这 8 张 H200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果那时没有足够的空闲 H200 继续完成当前任务，就等待半小时；如果半小时内资源仍然不足，再停止并报告 blocker。
要求: 只使用 8 张 H200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；重点看 MoE/EP、attention、long-context prefill/decode。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/h200_qwen3_235b_a22b_fp8_sota_humanize
```

### Qwen/Qwen3-VL-235B-A22B-Instruct-FP8
```text
使用 sglang-sota-humanize-loop skill。
model_id: Qwen/Qwen3-VL-235B-A22B-Instruct-FP8
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA H200
minimum_gpu_count: 8
precision_quantization: FP8
initial_deployment: SGLang TP=8, EP=2
要求: 远端使用 ion8-h200 或 ion9-h200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 启动 Humanize/RLCR loop 时，必须先 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: Humanize/RLCR loop 必须在本地 Codex 会话启动后启动；远端机器只作为执行和验证环境，不要在远端机器上启动或托管这个 loop。
要求: 每次 benchmark/profile 前必须确认这 8 张 H200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果那时没有足够的空闲 H200 继续完成当前任务，就等待半小时；如果半小时内资源仍然不足，再停止并报告 blocker。
要求: 只使用 8 张 H200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 skill 默认 workload，并保留 VLM 输入兼容性验证。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；重点看 VLM preprocess、MoE/EP、attention、decode throughput。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/h200_qwen3_vl_235b_a22b_instruct_fp8_sota_humanize
```

### meta-llama/Llama-4-Scout-17B-16E-Instruct
```text
使用 sglang-sota-humanize-loop skill。
model_id: meta-llama/Llama-4-Scout-17B-16E-Instruct
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA H200
minimum_gpu_count: 8
precision_quantization: BF16
initial_deployment: SGLang TP=8
要求: 远端使用 ion8-h200 或 ion9-h200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 启动 Humanize/RLCR loop 时，必须先 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: Humanize/RLCR loop 必须在本地 Codex 会话启动后启动；远端机器只作为执行和验证环境，不要在远端机器上启动或托管这个 loop。
要求: 每次 benchmark/profile 前必须确认这 8 张 H200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果那时没有足够的空闲 H200 继续完成当前任务，就等待半小时；如果半小时内资源仍然不足，再停止并报告 blocker。
要求: 只使用 8 张 H200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；重点看 MoE execution、attention backend、overlap、decode throughput。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/h200_llama4_scout_17b_16e_instruct_sota_humanize
```
