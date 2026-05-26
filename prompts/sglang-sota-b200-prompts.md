# B200 prompts

## 1 GPU

### mistralai/Mistral-Small-4-119B-2603
```text
使用 sglang-sota-humanize-loop skill。
model_id: mistralai/Mistral-Small-4-119B-2603
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 1x NVIDIA B200
minimum_gpu_count: 1
precision_quantization: FP8
initial_deployment: SGLang TP=1
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 启动 Humanize/RLCR loop 时，必须在本地 Codex 会话里启动；远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 loop。启动前仍必须先在本地 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: 每次 benchmark/profile 前必须确认这 1 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 1 张 B200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 1 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_mistral_small4_119b_2603_sota_humanize
```

### Qwen/Qwen3-30B-A3B-FP8
```text
使用 sglang-sota-humanize-loop skill。
model_id: Qwen/Qwen3-30B-A3B-FP8
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 1x NVIDIA B200
minimum_gpu_count: 1
precision_quantization: FP8
initial_deployment: SGLang TP=1
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 启动 Humanize/RLCR loop 时，必须在本地 Codex 会话里启动；远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 loop。启动前仍必须先在本地 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: 每次 benchmark/profile 前必须确认这 1 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 1 张 B200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 1 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_qwen3_30b_a3b_fp8_sota_humanize
```

### google/gemma-4-31B-it
```text
使用 sglang-sota-humanize-loop skill。
model_id: google/gemma-4-31B-it
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 1x NVIDIA B200
minimum_gpu_count: 1
precision_quantization: BF16
initial_deployment: SGLang TP=1
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 启动 Humanize/RLCR loop 时，必须在本地 Codex 会话里启动；远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 loop。启动前仍必须先在本地 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: 每次 benchmark/profile 前必须确认这 1 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 1 张 B200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 1 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_gemma4_31b_it_sota_humanize
```

## 2 GPU

### Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
```text
使用 sglang-sota-humanize-loop skill。
model_id: Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 2x NVIDIA B200
minimum_gpu_count: 2
precision_quantization: FP8
initial_deployment: SGLang TP=2
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 启动 Humanize/RLCR loop 时，必须在本地 Codex 会话里启动；远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 loop。启动前仍必须先在本地 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: 每次 benchmark/profile 前必须确认这 2 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 2 张 B200；不要测试 4 卡或 8 卡部署。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 2 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；重点看 hybrid attention、Mamba/GDN、radix cache、target verify、CUDA graph。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_qwen3_next_80b_a3b_instruct_fp8_sota_humanize
```

### Qwen/Qwen3-Next-80B-A3B-Thinking-FP8
```text
使用 sglang-sota-humanize-loop skill。
model_id: Qwen/Qwen3-Next-80B-A3B-Thinking-FP8
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 2x NVIDIA B200
minimum_gpu_count: 2
precision_quantization: FP8
initial_deployment: SGLang TP=2
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 启动 Humanize/RLCR loop 时，必须在本地 Codex 会话里启动；远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 loop。启动前仍必须先在本地 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: 每次 benchmark/profile 前必须确认这 2 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 2 张 B200；不要测试 4 卡或 8 卡部署。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 2 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；重点看 thinking workload 下的 Mamba/GDN、decode latency、target verify。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_qwen3_next_80b_a3b_thinking_fp8_sota_humanize
```

## 4 GPU

### nvidia/DeepSeek-V3.2-NVFP4
```text
使用 sglang-sota-humanize-loop skill。
model_id: nvidia/DeepSeek-V3.2-NVFP4
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 4x NVIDIA B200
minimum_gpu_count: 4
precision_quantization: NVFP4
initial_deployment: SGLang TP=4；DP/EP/MTP 只能在同样 4 卡预算内搜索
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 启动 Humanize/RLCR loop 时，必须在本地 Codex 会话里启动；远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 loop。启动前仍必须先在本地 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: 每次 benchmark/profile 前必须确认这 4 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 4 张 B200；不要测试 8 卡，除非 4 卡实测 OOM 并记录 artifact。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 4 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；重点看 NVFP4 kernels、MLA、MoE/EP、DP attention、speculative decoding、memory/cache。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_deepseek_v32_nvfp4_sota_humanize
```

### Qwen/Qwen3.5-397B-A17B-FP8
```text
使用 sglang-sota-humanize-loop skill。
model_id: Qwen/Qwen3.5-397B-A17B-FP8
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 4x NVIDIA B200
minimum_gpu_count: 4
precision_quantization: FP8
initial_deployment: SGLang TP=4；MTP 只能在同样 4 卡预算内搜索
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 启动 Humanize/RLCR loop 时，必须在本地 Codex 会话里启动；远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 loop。启动前仍必须先在本地 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: 每次 benchmark/profile 前必须确认这 4 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 4 张 B200；不要测试 8 卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 4 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；重点看 MoE routing、EP、attention backend、MTP/speculative、long-context serving。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_qwen35_397b_a17b_fp8_sota_humanize
```

### nvidia/Qwen3.5-397B-A17B-NVFP4
```text
使用 sglang-sota-humanize-loop skill。
model_id: nvidia/Qwen3.5-397B-A17B-NVFP4
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 4x NVIDIA B200
minimum_gpu_count: 4
precision_quantization: NVFP4/FP4
initial_deployment: SGLang TP=4；MTP 只能在同样 4 卡预算内搜索
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 启动 Humanize/RLCR loop 时，必须在本地 Codex 会话里启动；远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 loop。启动前仍必须先在本地 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: 每次 benchmark/profile 前必须确认这 4 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 4 张 B200；不要测试 8 卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 4 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；重点看 NVFP4 execution、MoE routing、EP、attention backend、MTP/speculative。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_qwen35_397b_a17b_nvfp4_sota_humanize
```

### nvidia/GLM-5-NVFP4
```text
使用 sglang-sota-humanize-loop skill。
model_id: nvidia/GLM-5-NVFP4
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 4x NVIDIA B200
minimum_gpu_count: 4
precision_quantization: NVFP4
initial_deployment: SGLang TP=4；DP/MTP 只能在同样 4 卡预算内搜索
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 启动 Humanize/RLCR loop 时，必须在本地 Codex 会话里启动；远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 loop。启动前仍必须先在本地 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: 每次 benchmark/profile 前必须确认这 4 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 4 张 B200；不要测试 8 卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 4 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；重点看 NVFP4 kernels、MTP/speculative、long-context behavior、framework parity。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_glm5_nvfp4_sota_humanize
```

### nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
```text
使用 sglang-sota-humanize-loop skill。
model_id: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 4x NVIDIA B200
minimum_gpu_count: 4
precision_quantization: BF16
initial_deployment: SGLang TP=4
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 启动 Humanize/RLCR loop 时，必须在本地 Codex 会话里启动；远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 loop。启动前仍必须先在本地 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: 每次 benchmark/profile 前必须确认这 4 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 4 张 B200；不要测试 8 卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 4 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；重点看 MoE、attention/backend、memory pressure、decode throughput。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_nemotron3_super_120b_a12b_bf16_sota_humanize
```

## 8 GPU

### deepseek-ai/DeepSeek-V3.2
```text
使用 sglang-sota-humanize-loop skill。
model_id: deepseek-ai/DeepSeek-V3.2
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA B200
minimum_gpu_count: 8
precision_quantization: FP8
initial_deployment: SGLang TP=8；DP/EP/MTP 只能在同样 8 卡预算内搜索
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 启动 Humanize/RLCR loop 时，必须在本地 Codex 会话里启动；远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 loop。启动前仍必须先在本地 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: 每次 benchmark/profile 前必须确认这 8 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 8 张 B200；不要测试多机或更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；重点看 MLA、MoE/EP、DP attention、speculative decoding、overlap、memory/cache。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_deepseek_v32_fp8_sota_humanize
```

### nvidia/DeepSeek-R1-0528-FP4-v2
```text
使用 sglang-sota-humanize-loop skill。
model_id: nvidia/DeepSeek-R1-0528-FP4-v2
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA B200
minimum_gpu_count: 8
precision_quantization: FP4
initial_deployment: SGLang TP=8；DP/EP/MTP 只能在同样 8 卡预算内搜索
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 启动 Humanize/RLCR loop 时，必须在本地 Codex 会话里启动；远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 loop。启动前仍必须先在本地 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: 每次 benchmark/profile 前必须确认这 8 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 8 张 B200；不要测试多机或更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；重点看 FP4/NVFP4 kernel、MLA、MoE、memory bandwidth、prefill/decode imbalance。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_deepseek_r1_0528_fp4_sota_humanize
```

### moonshotai/Kimi-K2-Instruct
```text
使用 sglang-sota-humanize-loop skill。
model_id: moonshotai/Kimi-K2-Instruct
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA B200
minimum_gpu_count: 8
precision_quantization: FP8
initial_deployment: SGLang TP=8；DP/EP 只能在同样 8 卡预算内搜索
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 启动 Humanize/RLCR loop 时，必须在本地 Codex 会话里启动；远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 loop。启动前仍必须先在本地 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: 每次 benchmark/profile 前必须确认这 8 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 8 张 B200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；重点看 MLA/MoE/DP-attention、long-context decode、framework parity。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_kimi_k2_instruct_sota_humanize
```

### MiniMaxAI/MiniMax-M2.7
```text
使用 sglang-sota-humanize-loop skill。
model_id: MiniMaxAI/MiniMax-M2.7
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA B200
minimum_gpu_count: 8
precision_quantization: FP8
initial_deployment: SGLang TP=8
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 启动 Humanize/RLCR loop 时，必须在本地 Codex 会话里启动；远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 loop。启动前仍必须先在本地 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: 每次 benchmark/profile 前必须确认这 8 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 8 张 B200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；重点看 MoE、attention、memory pressure、decode throughput。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_minimax_m27_sota_humanize
```

### inclusionAI/Ring-2.5-1T
```text
使用 sglang-sota-humanize-loop skill。
model_id: inclusionAI/Ring-2.5-1T
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA B200
minimum_gpu_count: 8
precision_quantization: FP8
initial_deployment: SGLang TP=8
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 启动 Humanize/RLCR loop 时，必须在本地 Codex 会话里启动；远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 loop。启动前仍必须先在本地 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: 每次 benchmark/profile 前必须确认这 8 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 8 张 B200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；优先保证稳定 launch、memory/capacity 证据和 profile-driven root cause。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_ring25_1t_sota_humanize
```

### deepseek-ai/DeepSeek-Math-V2
```text
使用 sglang-sota-humanize-loop skill。
model_id: deepseek-ai/DeepSeek-Math-V2
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA B200
minimum_gpu_count: 8
precision_quantization: BF16
initial_deployment: SGLang TP=8；DP 只能在同样 8 卡预算内搜索
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 启动 Humanize/RLCR loop 时，必须在本地 Codex 会话里启动；远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 loop。启动前仍必须先在本地 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: 每次 benchmark/profile 前必须确认这 8 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 8 张 B200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_deepseek_math_v2_sota_humanize
```

### zai-org/GLM-5-FP8
```text
使用 sglang-sota-humanize-loop skill。
model_id: zai-org/GLM-5-FP8
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA B200
minimum_gpu_count: 8
precision_quantization: FP8
initial_deployment: SGLang TP=8；DP/MTP 只能在同样 8 卡预算内搜索
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 启动 Humanize/RLCR loop 时，必须在本地 Codex 会话里启动；远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 loop。启动前仍必须先在本地 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: 每次 benchmark/profile 前必须确认这 8 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 8 张 B200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；重点看 MTP/speculative、long-context behavior、framework parity。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_glm5_fp8_sota_humanize
```

### zai-org/GLM-4.6-FP8
```text
使用 sglang-sota-humanize-loop skill。
model_id: zai-org/GLM-4.6-FP8
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA B200
minimum_gpu_count: 8
precision_quantization: FP8
initial_deployment: SGLang TP=8；DP/EP/MTP 只能在同样 8 卡预算内搜索
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 启动 Humanize/RLCR loop 时，必须在本地 Codex 会话里启动；远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 loop。启动前仍必须先在本地 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: 每次 benchmark/profile 前必须确认这 8 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 8 张 B200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_glm46_fp8_sota_humanize
```

### openai/gpt-oss-120b
```text
使用 sglang-sota-humanize-loop skill。
model_id: openai/gpt-oss-120b
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA B200
minimum_gpu_count: 8
precision_quantization: MXFP4
initial_deployment: SGLang TP=8
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 启动 Humanize/RLCR loop 时，必须在本地 Codex 会话里启动；远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 loop。启动前仍必须先在本地 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: 每次 benchmark/profile 前必须确认这 8 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 8 张 B200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_gpt_oss_120b_sota_humanize
```

### internlm/Intern-S1-FP8
```text
使用 sglang-sota-humanize-loop skill。
model_id: internlm/Intern-S1-FP8
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA B200
minimum_gpu_count: 8
precision_quantization: FP8
initial_deployment: SGLang TP=8, EP=2
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 启动 Humanize/RLCR loop 时，必须在本地 Codex 会话里启动；远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 loop。启动前仍必须先在本地 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: 每次 benchmark/profile 前必须确认这 8 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 8 张 B200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_intern_s1_fp8_sota_humanize
```

### Qwen/Qwen3-235B-A22B-FP8
```text
使用 sglang-sota-humanize-loop skill。
model_id: Qwen/Qwen3-235B-A22B-FP8
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA B200
minimum_gpu_count: 8
precision_quantization: FP8
initial_deployment: SGLang TP=8, EP=2
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 启动 Humanize/RLCR loop 时，必须在本地 Codex 会话里启动；远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 loop。启动前仍必须先在本地 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: 每次 benchmark/profile 前必须确认这 8 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 8 张 B200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；重点看 MoE/EP、attention、long-context prefill/decode。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_qwen3_235b_a22b_fp8_sota_humanize
```

### Qwen/Qwen3-VL-235B-A22B-Instruct-FP8
```text
使用 sglang-sota-humanize-loop skill。
model_id: Qwen/Qwen3-VL-235B-A22B-Instruct-FP8
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA B200
minimum_gpu_count: 8
precision_quantization: FP8
initial_deployment: SGLang TP=8, EP=2
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 启动 Humanize/RLCR loop 时，必须在本地 Codex 会话里启动；远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 loop。启动前仍必须先在本地 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: 每次 benchmark/profile 前必须确认这 8 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 8 张 B200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 skill 默认 workload，并保留 VLM 输入兼容性验证。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；重点看 VLM preprocess、MoE/EP、attention、decode throughput。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_qwen3_vl_235b_a22b_instruct_fp8_sota_humanize
```

### meta-llama/Llama-4-Scout-17B-16E-Instruct
```text
使用 sglang-sota-humanize-loop skill。
model_id: meta-llama/Llama-4-Scout-17B-16E-Instruct
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA B200
minimum_gpu_count: 8
precision_quantization: BF16
initial_deployment: SGLang TP=8
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 启动 Humanize/RLCR loop 时，必须在本地 Codex 会话里启动；远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 loop。启动前仍必须先在本地 cd 到当前任务创建出的工程目录，也就是包含 .humanize 的目录，再启动 loop。
要求: 每次 benchmark/profile 前必须确认这 8 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 8 张 B200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 skill 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，profile 后再 patch；重点看 MoE execution、attention backend、overlap、decode throughput。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_llama4_scout_17b_16e_instruct_sota_humanize
```
