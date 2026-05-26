# B200 Codex Goal prompts

这份文档把 `sglang-sota-b200-prompts.md` 改写成 Codex Goal prompt。
每次只使用其中一个模型块，把完整 fenced text 粘贴到支持 `/goal` 的
Codex 线程里。Goal 写法参考
[在 Codex 中使用 Goals：长时间工作的持久目标](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/large-language-model/codex/%E3%80%90%E7%BF%BB%E8%AF%91%E3%80%91%E5%9C%A8%20Codex%20%E4%B8%AD%E4%BD%BF%E7%94%A8%20Goals%EF%BC%9A%E9%95%BF%E6%97%B6%E9%97%B4%E5%B7%A5%E4%BD%9C%E7%9A%84%E6%8C%81%E4%B9%85%E7%9B%AE%E6%A0%87.md)
里的结果、验证面、约束、边界、迭代策略和阻塞停止条件。

Goal mode 本身就是持久循环，本文档是独立的 Codex Goal prompt 集合。流程约束仍尽量贴近原始 B200 SOTA prompt：先完成固定公平
benchmark，再做 profile-driven gap analysis，源码改动前必须使用 `llm-pipeline-analysis` skill 做 pipeline
analysis，真实模型重测，最后用证据支撑 PR。

## Goal 契约

一个强 Goal 需要说明结果、验证面、约束、边界、迭代策略和阻塞停止条件。
这些 prompt 统一编码为：

- 结果：在指定 B200 预算下，SGLang 匹配或超越同 workload/SLA 下可复现的最佳
  vLLM/TensorRT-LLM 结果；只有证据显示不值得 patch 时，才接受稳定 1% 内差距。
- 验证面：benchmark summary、winning commands、profiler trace、
  `llm-pipeline-analysis` skill 输出、必要时的 NCU 证据、final report，以及 PR 描述里的
  benchmark、GSM8K、MMLU 全量精度表。
- 约束：远端 workspace 不能脏，不能做破坏性 git 操作，不能无限扩 GPU，不能清理
  当前模型以外的共享 cache，不能向 `sgl-project/sglang` 推送或开 PR。
- 边界：本地 Codex 线程拥有 Goal 和决策；`ion-b200` 只作为执行、benchmark、
  profile、验证环境。
- 迭代：benchmark、判断 gap、profile、使用 llm-pipeline-analysis skill 跑 pipeline analysis、只 patch 有证据支撑的
  SGLang 路径、revalidate，并持续到成功或出现可解释 blocker。
- 阻塞停止：如果所需 GPU 在等待窗口后仍不可用、benchmark/profile 无法收集可信证据、
  repo/workspace 脏，或没有可辩护的下一步 patch，就停止并报告。

常用生命周期命令：

```text
/goal pause
/goal resume
/goal clear
```

## 1 GPU

### mistralai/Mistral-Small-4-119B-2603
```text
/goal 持续推进 `mistralai/Mistral-Small-4-119B-2603` 在 single-node 1x NVIDIA B200 上的 SGLang SOTA serving 优化，直到 SGLang 在同样 1 张 B200、FP8、相同 workload 和 SLA 下匹配或超越可复现的最佳 vLLM/TensorRT-LLM 结果；由固定公平 benchmark、profile、llm-pipeline-analysis skill、必要的 NCU 证据、最终报告以及 PR 中的 benchmark/GSM8K/MMLU 全量精度表验证，同时保持正确性、精度和环境安全约束不回退。Goal 本身就是循环，所有迭代都在当前 Goal 中推进。
model_id: mistralai/Mistral-Small-4-119B-2603
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 1x NVIDIA B200
minimum_gpu_count: 1
precision_quantization: FP8
initial_deployment: SGLang TP=1
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 使用当前 Codex Goal 作为唯一持久循环。远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 agent loop；所有决策、状态总结、完成/阻塞判定都留在本地 Codex Goal 线程。
要求: 每次 benchmark/profile 前必须确认这 1 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 1 张 B200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 1 卡预算下的公平搜索；使用 llm-serving-auto-benchmark 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，在同一 Goal 中先 profile，再使用 llm-torch-profiler-analysis skill 和 llm-pipeline-analysis skill，必要时使用 ncu-report-skill，随后只 patch 有证据支持的 SGLang 代码路径。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_mistral_small4_119b_2603_sota_goal
要求: 固定公平 benchmark 完成前不要选择代码 patch；benchmark 之后的 gap decision、profile、llm-pipeline-analysis skill、patch、revalidate 全部属于当前 Codex Goal 的持续迭代。
要求: 每一轮都更新 artifact_root 下的 manifest、benchmark/profile/analysis 证据、失败尝试和下一步判断；Goal 只有在证据满足完成条件后才能完成。
要求: 如果达到预算上限、GPU 资源阻塞、benchmark/profile 无法获得可信证据，或者没有可辩护的下一步 patch，就停止实质性工作并报告已尝试路径、证据、阻塞点和继续推进所需输入。
```

### Qwen/Qwen3-30B-A3B-FP8
```text
/goal 持续推进 `Qwen/Qwen3-30B-A3B-FP8` 在 single-node 1x NVIDIA B200 上的 SGLang SOTA serving 优化，直到 SGLang 在同样 1 张 B200、FP8、相同 workload 和 SLA 下匹配或超越可复现的最佳 vLLM/TensorRT-LLM 结果；由固定公平 benchmark、profile、llm-pipeline-analysis skill、必要的 NCU 证据、最终报告以及 PR 中的 benchmark/GSM8K/MMLU 全量精度表验证，同时保持正确性、精度和环境安全约束不回退。Goal 本身就是循环，所有迭代都在当前 Goal 中推进。
model_id: Qwen/Qwen3-30B-A3B-FP8
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 1x NVIDIA B200
minimum_gpu_count: 1
precision_quantization: FP8
initial_deployment: SGLang TP=1
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 使用当前 Codex Goal 作为唯一持久循环。远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 agent loop；所有决策、状态总结、完成/阻塞判定都留在本地 Codex Goal 线程。
要求: 每次 benchmark/profile 前必须确认这 1 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 1 张 B200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 1 卡预算下的公平搜索；使用 llm-serving-auto-benchmark 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，在同一 Goal 中先 profile，再使用 llm-torch-profiler-analysis skill 和 llm-pipeline-analysis skill，必要时使用 ncu-report-skill，随后只 patch 有证据支持的 SGLang 代码路径。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_qwen3_30b_a3b_fp8_sota_goal
要求: 固定公平 benchmark 完成前不要选择代码 patch；benchmark 之后的 gap decision、profile、llm-pipeline-analysis skill、patch、revalidate 全部属于当前 Codex Goal 的持续迭代。
要求: 每一轮都更新 artifact_root 下的 manifest、benchmark/profile/analysis 证据、失败尝试和下一步判断；Goal 只有在证据满足完成条件后才能完成。
要求: 如果达到预算上限、GPU 资源阻塞、benchmark/profile 无法获得可信证据，或者没有可辩护的下一步 patch，就停止实质性工作并报告已尝试路径、证据、阻塞点和继续推进所需输入。
```

### google/gemma-4-31B-it
```text
/goal 持续推进 `google/gemma-4-31B-it` 在 single-node 1x NVIDIA B200 上的 SGLang SOTA serving 优化，直到 SGLang 在同样 1 张 B200、BF16、相同 workload 和 SLA 下匹配或超越可复现的最佳 vLLM/TensorRT-LLM 结果；由固定公平 benchmark、profile、llm-pipeline-analysis skill、必要的 NCU 证据、最终报告以及 PR 中的 benchmark/GSM8K/MMLU 全量精度表验证，同时保持正确性、精度和环境安全约束不回退。Goal 本身就是循环，所有迭代都在当前 Goal 中推进。
model_id: google/gemma-4-31B-it
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 1x NVIDIA B200
minimum_gpu_count: 1
precision_quantization: BF16
initial_deployment: SGLang TP=1
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 使用当前 Codex Goal 作为唯一持久循环。远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 agent loop；所有决策、状态总结、完成/阻塞判定都留在本地 Codex Goal 线程。
要求: 每次 benchmark/profile 前必须确认这 1 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 1 张 B200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 1 卡预算下的公平搜索；使用 llm-serving-auto-benchmark 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，在同一 Goal 中先 profile，再使用 llm-torch-profiler-analysis skill 和 llm-pipeline-analysis skill，必要时使用 ncu-report-skill，随后只 patch 有证据支持的 SGLang 代码路径。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_gemma4_31b_it_sota_goal
要求: 固定公平 benchmark 完成前不要选择代码 patch；benchmark 之后的 gap decision、profile、llm-pipeline-analysis skill、patch、revalidate 全部属于当前 Codex Goal 的持续迭代。
要求: 每一轮都更新 artifact_root 下的 manifest、benchmark/profile/analysis 证据、失败尝试和下一步判断；Goal 只有在证据满足完成条件后才能完成。
要求: 如果达到预算上限、GPU 资源阻塞、benchmark/profile 无法获得可信证据，或者没有可辩护的下一步 patch，就停止实质性工作并报告已尝试路径、证据、阻塞点和继续推进所需输入。
```

## 2 GPU

### Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
```text
/goal 持续推进 `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` 在 single-node 2x NVIDIA B200 上的 SGLang SOTA serving 优化，直到 SGLang 在同样 2 张 B200、FP8、相同 workload 和 SLA 下匹配或超越可复现的最佳 vLLM/TensorRT-LLM 结果；由固定公平 benchmark、profile、llm-pipeline-analysis skill、必要的 NCU 证据、最终报告以及 PR 中的 benchmark/GSM8K/MMLU 全量精度表验证，同时保持正确性、精度和环境安全约束不回退。Goal 本身就是循环，所有迭代都在当前 Goal 中推进。
model_id: Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 2x NVIDIA B200
minimum_gpu_count: 2
precision_quantization: FP8
initial_deployment: SGLang TP=2
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 使用当前 Codex Goal 作为唯一持久循环。远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 agent loop；所有决策、状态总结、完成/阻塞判定都留在本地 Codex Goal 线程。
要求: 每次 benchmark/profile 前必须确认这 2 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 2 张 B200；不要测试 4 卡或 8 卡部署。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 2 卡预算下的公平搜索；使用 llm-serving-auto-benchmark 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，在同一 Goal 中先 profile，再使用 llm-torch-profiler-analysis skill 和 llm-pipeline-analysis skill，必要时使用 ncu-report-skill，随后只 patch 有证据支持的 SGLang 代码路径；重点看 hybrid attention、Mamba/GDN、radix cache、target verify、CUDA graph。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_qwen3_next_80b_a3b_instruct_fp8_sota_goal
要求: 固定公平 benchmark 完成前不要选择代码 patch；benchmark 之后的 gap decision、profile、llm-pipeline-analysis skill、patch、revalidate 全部属于当前 Codex Goal 的持续迭代。
要求: 每一轮都更新 artifact_root 下的 manifest、benchmark/profile/analysis 证据、失败尝试和下一步判断；Goal 只有在证据满足完成条件后才能完成。
要求: 如果达到预算上限、GPU 资源阻塞、benchmark/profile 无法获得可信证据，或者没有可辩护的下一步 patch，就停止实质性工作并报告已尝试路径、证据、阻塞点和继续推进所需输入。
```

### Qwen/Qwen3-Next-80B-A3B-Thinking-FP8
```text
/goal 持续推进 `Qwen/Qwen3-Next-80B-A3B-Thinking-FP8` 在 single-node 2x NVIDIA B200 上的 SGLang SOTA serving 优化，直到 SGLang 在同样 2 张 B200、FP8、相同 workload 和 SLA 下匹配或超越可复现的最佳 vLLM/TensorRT-LLM 结果；由固定公平 benchmark、profile、llm-pipeline-analysis skill、必要的 NCU 证据、最终报告以及 PR 中的 benchmark/GSM8K/MMLU 全量精度表验证，同时保持正确性、精度和环境安全约束不回退。Goal 本身就是循环，所有迭代都在当前 Goal 中推进。
model_id: Qwen/Qwen3-Next-80B-A3B-Thinking-FP8
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 2x NVIDIA B200
minimum_gpu_count: 2
precision_quantization: FP8
initial_deployment: SGLang TP=2
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 使用当前 Codex Goal 作为唯一持久循环。远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 agent loop；所有决策、状态总结、完成/阻塞判定都留在本地 Codex Goal 线程。
要求: 每次 benchmark/profile 前必须确认这 2 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 2 张 B200；不要测试 4 卡或 8 卡部署。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 2 卡预算下的公平搜索；使用 llm-serving-auto-benchmark 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，在同一 Goal 中先 profile，再使用 llm-torch-profiler-analysis skill 和 llm-pipeline-analysis skill，必要时使用 ncu-report-skill，随后只 patch 有证据支持的 SGLang 代码路径；重点看 thinking workload 下的 Mamba/GDN、decode latency、target verify。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_qwen3_next_80b_a3b_thinking_fp8_sota_goal
要求: 固定公平 benchmark 完成前不要选择代码 patch；benchmark 之后的 gap decision、profile、llm-pipeline-analysis skill、patch、revalidate 全部属于当前 Codex Goal 的持续迭代。
要求: 每一轮都更新 artifact_root 下的 manifest、benchmark/profile/analysis 证据、失败尝试和下一步判断；Goal 只有在证据满足完成条件后才能完成。
要求: 如果达到预算上限、GPU 资源阻塞、benchmark/profile 无法获得可信证据，或者没有可辩护的下一步 patch，就停止实质性工作并报告已尝试路径、证据、阻塞点和继续推进所需输入。
```

## 4 GPU

### nvidia/DeepSeek-V3.2-NVFP4
```text
/goal 持续推进 `nvidia/DeepSeek-V3.2-NVFP4` 在 single-node 4x NVIDIA B200 上的 SGLang SOTA serving 优化，直到 SGLang 在同样 4 张 B200、NVFP4、相同 workload 和 SLA 下匹配或超越可复现的最佳 vLLM/TensorRT-LLM 结果；由固定公平 benchmark、profile、llm-pipeline-analysis skill、必要的 NCU 证据、最终报告以及 PR 中的 benchmark/GSM8K/MMLU 全量精度表验证，同时保持正确性、精度和环境安全约束不回退。Goal 本身就是循环，所有迭代都在当前 Goal 中推进。
model_id: nvidia/DeepSeek-V3.2-NVFP4
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 4x NVIDIA B200
minimum_gpu_count: 4
precision_quantization: NVFP4
initial_deployment: SGLang TP=4；DP/EP/MTP 只能在同样 4 卡预算内搜索
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 使用当前 Codex Goal 作为唯一持久循环。远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 agent loop；所有决策、状态总结、完成/阻塞判定都留在本地 Codex Goal 线程。
要求: 每次 benchmark/profile 前必须确认这 4 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 4 张 B200；不要测试 8 卡，除非 4 卡实测 OOM 并记录 artifact。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 4 卡预算下的公平搜索；使用 llm-serving-auto-benchmark 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，在同一 Goal 中先 profile，再使用 llm-torch-profiler-analysis skill 和 llm-pipeline-analysis skill，必要时使用 ncu-report-skill，随后只 patch 有证据支持的 SGLang 代码路径；重点看 NVFP4 kernels、MLA、MoE/EP、DP attention、speculative decoding、memory/cache。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_deepseek_v32_nvfp4_sota_goal
要求: 固定公平 benchmark 完成前不要选择代码 patch；benchmark 之后的 gap decision、profile、llm-pipeline-analysis skill、patch、revalidate 全部属于当前 Codex Goal 的持续迭代。
要求: 每一轮都更新 artifact_root 下的 manifest、benchmark/profile/analysis 证据、失败尝试和下一步判断；Goal 只有在证据满足完成条件后才能完成。
要求: 如果达到预算上限、GPU 资源阻塞、benchmark/profile 无法获得可信证据，或者没有可辩护的下一步 patch，就停止实质性工作并报告已尝试路径、证据、阻塞点和继续推进所需输入。
```

### Qwen/Qwen3.5-397B-A17B-FP8
```text
/goal 持续推进 `Qwen/Qwen3.5-397B-A17B-FP8` 在 single-node 4x NVIDIA B200 上的 SGLang SOTA serving 优化，直到 SGLang 在同样 4 张 B200、FP8、相同 workload 和 SLA 下匹配或超越可复现的最佳 vLLM/TensorRT-LLM 结果；由固定公平 benchmark、profile、llm-pipeline-analysis skill、必要的 NCU 证据、最终报告以及 PR 中的 benchmark/GSM8K/MMLU 全量精度表验证，同时保持正确性、精度和环境安全约束不回退。Goal 本身就是循环，所有迭代都在当前 Goal 中推进。
model_id: Qwen/Qwen3.5-397B-A17B-FP8
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 4x NVIDIA B200
minimum_gpu_count: 4
precision_quantization: FP8
initial_deployment: SGLang TP=4；MTP 只能在同样 4 卡预算内搜索
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 使用当前 Codex Goal 作为唯一持久循环。远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 agent loop；所有决策、状态总结、完成/阻塞判定都留在本地 Codex Goal 线程。
要求: 每次 benchmark/profile 前必须确认这 4 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 4 张 B200；不要测试 8 卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 4 卡预算下的公平搜索；使用 llm-serving-auto-benchmark 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，在同一 Goal 中先 profile，再使用 llm-torch-profiler-analysis skill 和 llm-pipeline-analysis skill，必要时使用 ncu-report-skill，随后只 patch 有证据支持的 SGLang 代码路径；重点看 MoE routing、EP、attention backend、MTP/speculative、long-context serving。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_qwen35_397b_a17b_fp8_sota_goal
要求: 固定公平 benchmark 完成前不要选择代码 patch；benchmark 之后的 gap decision、profile、llm-pipeline-analysis skill、patch、revalidate 全部属于当前 Codex Goal 的持续迭代。
要求: 每一轮都更新 artifact_root 下的 manifest、benchmark/profile/analysis 证据、失败尝试和下一步判断；Goal 只有在证据满足完成条件后才能完成。
要求: 如果达到预算上限、GPU 资源阻塞、benchmark/profile 无法获得可信证据，或者没有可辩护的下一步 patch，就停止实质性工作并报告已尝试路径、证据、阻塞点和继续推进所需输入。
```

### nvidia/Qwen3.5-397B-A17B-NVFP4
```text
/goal 持续推进 `nvidia/Qwen3.5-397B-A17B-NVFP4` 在 single-node 4x NVIDIA B200 上的 SGLang SOTA serving 优化，直到 SGLang 在同样 4 张 B200、NVFP4/FP4、相同 workload 和 SLA 下匹配或超越可复现的最佳 vLLM/TensorRT-LLM 结果；由固定公平 benchmark、profile、llm-pipeline-analysis skill、必要的 NCU 证据、最终报告以及 PR 中的 benchmark/GSM8K/MMLU 全量精度表验证，同时保持正确性、精度和环境安全约束不回退。Goal 本身就是循环，所有迭代都在当前 Goal 中推进。
model_id: nvidia/Qwen3.5-397B-A17B-NVFP4
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 4x NVIDIA B200
minimum_gpu_count: 4
precision_quantization: NVFP4/FP4
initial_deployment: SGLang TP=4；MTP 只能在同样 4 卡预算内搜索
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 使用当前 Codex Goal 作为唯一持久循环。远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 agent loop；所有决策、状态总结、完成/阻塞判定都留在本地 Codex Goal 线程。
要求: 每次 benchmark/profile 前必须确认这 4 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 4 张 B200；不要测试 8 卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 4 卡预算下的公平搜索；使用 llm-serving-auto-benchmark 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，在同一 Goal 中先 profile，再使用 llm-torch-profiler-analysis skill 和 llm-pipeline-analysis skill，必要时使用 ncu-report-skill，随后只 patch 有证据支持的 SGLang 代码路径；重点看 NVFP4 execution、MoE routing、EP、attention backend、MTP/speculative。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_qwen35_397b_a17b_nvfp4_sota_goal
要求: 固定公平 benchmark 完成前不要选择代码 patch；benchmark 之后的 gap decision、profile、llm-pipeline-analysis skill、patch、revalidate 全部属于当前 Codex Goal 的持续迭代。
要求: 每一轮都更新 artifact_root 下的 manifest、benchmark/profile/analysis 证据、失败尝试和下一步判断；Goal 只有在证据满足完成条件后才能完成。
要求: 如果达到预算上限、GPU 资源阻塞、benchmark/profile 无法获得可信证据，或者没有可辩护的下一步 patch，就停止实质性工作并报告已尝试路径、证据、阻塞点和继续推进所需输入。
```

### nvidia/GLM-5-NVFP4
```text
/goal 持续推进 `nvidia/GLM-5-NVFP4` 在 single-node 4x NVIDIA B200 上的 SGLang SOTA serving 优化，直到 SGLang 在同样 4 张 B200、NVFP4、相同 workload 和 SLA 下匹配或超越可复现的最佳 vLLM/TensorRT-LLM 结果；由固定公平 benchmark、profile、llm-pipeline-analysis skill、必要的 NCU 证据、最终报告以及 PR 中的 benchmark/GSM8K/MMLU 全量精度表验证，同时保持正确性、精度和环境安全约束不回退。Goal 本身就是循环，所有迭代都在当前 Goal 中推进。
model_id: nvidia/GLM-5-NVFP4
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 4x NVIDIA B200
minimum_gpu_count: 4
precision_quantization: NVFP4
initial_deployment: SGLang TP=4；DP/MTP 只能在同样 4 卡预算内搜索
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 使用当前 Codex Goal 作为唯一持久循环。远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 agent loop；所有决策、状态总结、完成/阻塞判定都留在本地 Codex Goal 线程。
要求: 每次 benchmark/profile 前必须确认这 4 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 4 张 B200；不要测试 8 卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 4 卡预算下的公平搜索；使用 llm-serving-auto-benchmark 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，在同一 Goal 中先 profile，再使用 llm-torch-profiler-analysis skill 和 llm-pipeline-analysis skill，必要时使用 ncu-report-skill，随后只 patch 有证据支持的 SGLang 代码路径；重点看 NVFP4 kernels、MTP/speculative、long-context behavior、framework parity。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_glm5_nvfp4_sota_goal
要求: 固定公平 benchmark 完成前不要选择代码 patch；benchmark 之后的 gap decision、profile、llm-pipeline-analysis skill、patch、revalidate 全部属于当前 Codex Goal 的持续迭代。
要求: 每一轮都更新 artifact_root 下的 manifest、benchmark/profile/analysis 证据、失败尝试和下一步判断；Goal 只有在证据满足完成条件后才能完成。
要求: 如果达到预算上限、GPU 资源阻塞、benchmark/profile 无法获得可信证据，或者没有可辩护的下一步 patch，就停止实质性工作并报告已尝试路径、证据、阻塞点和继续推进所需输入。
```

### nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
```text
/goal 持续推进 `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` 在 single-node 4x NVIDIA B200 上的 SGLang SOTA serving 优化，直到 SGLang 在同样 4 张 B200、BF16、相同 workload 和 SLA 下匹配或超越可复现的最佳 vLLM/TensorRT-LLM 结果；由固定公平 benchmark、profile、llm-pipeline-analysis skill、必要的 NCU 证据、最终报告以及 PR 中的 benchmark/GSM8K/MMLU 全量精度表验证，同时保持正确性、精度和环境安全约束不回退。Goal 本身就是循环，所有迭代都在当前 Goal 中推进。
model_id: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 4x NVIDIA B200
minimum_gpu_count: 4
precision_quantization: BF16
initial_deployment: SGLang TP=4
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 使用当前 Codex Goal 作为唯一持久循环。远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 agent loop；所有决策、状态总结、完成/阻塞判定都留在本地 Codex Goal 线程。
要求: 每次 benchmark/profile 前必须确认这 4 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 4 张 B200；不要测试 8 卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 4 卡预算下的公平搜索；使用 llm-serving-auto-benchmark 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，在同一 Goal 中先 profile，再使用 llm-torch-profiler-analysis skill 和 llm-pipeline-analysis skill，必要时使用 ncu-report-skill，随后只 patch 有证据支持的 SGLang 代码路径；重点看 MoE、attention/backend、memory pressure、decode throughput。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_nemotron3_super_120b_a12b_bf16_sota_goal
要求: 固定公平 benchmark 完成前不要选择代码 patch；benchmark 之后的 gap decision、profile、llm-pipeline-analysis skill、patch、revalidate 全部属于当前 Codex Goal 的持续迭代。
要求: 每一轮都更新 artifact_root 下的 manifest、benchmark/profile/analysis 证据、失败尝试和下一步判断；Goal 只有在证据满足完成条件后才能完成。
要求: 如果达到预算上限、GPU 资源阻塞、benchmark/profile 无法获得可信证据，或者没有可辩护的下一步 patch，就停止实质性工作并报告已尝试路径、证据、阻塞点和继续推进所需输入。
```

## 8 GPU

### deepseek-ai/DeepSeek-V3.2
```text
/goal 持续推进 `deepseek-ai/DeepSeek-V3.2` 在 single-node 8x NVIDIA B200 上的 SGLang SOTA serving 优化，直到 SGLang 在同样 8 张 B200、FP8、相同 workload 和 SLA 下匹配或超越可复现的最佳 vLLM/TensorRT-LLM 结果；由固定公平 benchmark、profile、llm-pipeline-analysis skill、必要的 NCU 证据、最终报告以及 PR 中的 benchmark/GSM8K/MMLU 全量精度表验证，同时保持正确性、精度和环境安全约束不回退。Goal 本身就是循环，所有迭代都在当前 Goal 中推进。
model_id: deepseek-ai/DeepSeek-V3.2
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA B200
minimum_gpu_count: 8
precision_quantization: FP8
initial_deployment: SGLang TP=8；DP/EP/MTP 只能在同样 8 卡预算内搜索
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 使用当前 Codex Goal 作为唯一持久循环。远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 agent loop；所有决策、状态总结、完成/阻塞判定都留在本地 Codex Goal 线程。
要求: 每次 benchmark/profile 前必须确认这 8 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 8 张 B200；不要测试多机或更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 llm-serving-auto-benchmark 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，在同一 Goal 中先 profile，再使用 llm-torch-profiler-analysis skill 和 llm-pipeline-analysis skill，必要时使用 ncu-report-skill，随后只 patch 有证据支持的 SGLang 代码路径；重点看 MLA、MoE/EP、DP attention、speculative decoding、overlap、memory/cache。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_deepseek_v32_fp8_sota_goal
要求: 固定公平 benchmark 完成前不要选择代码 patch；benchmark 之后的 gap decision、profile、llm-pipeline-analysis skill、patch、revalidate 全部属于当前 Codex Goal 的持续迭代。
要求: 每一轮都更新 artifact_root 下的 manifest、benchmark/profile/analysis 证据、失败尝试和下一步判断；Goal 只有在证据满足完成条件后才能完成。
要求: 如果达到预算上限、GPU 资源阻塞、benchmark/profile 无法获得可信证据，或者没有可辩护的下一步 patch，就停止实质性工作并报告已尝试路径、证据、阻塞点和继续推进所需输入。
```

### nvidia/DeepSeek-R1-0528-FP4-v2
```text
/goal 持续推进 `nvidia/DeepSeek-R1-0528-FP4-v2` 在 single-node 8x NVIDIA B200 上的 SGLang SOTA serving 优化，直到 SGLang 在同样 8 张 B200、FP4、相同 workload 和 SLA 下匹配或超越可复现的最佳 vLLM/TensorRT-LLM 结果；由固定公平 benchmark、profile、llm-pipeline-analysis skill、必要的 NCU 证据、最终报告以及 PR 中的 benchmark/GSM8K/MMLU 全量精度表验证，同时保持正确性、精度和环境安全约束不回退。Goal 本身就是循环，所有迭代都在当前 Goal 中推进。
model_id: nvidia/DeepSeek-R1-0528-FP4-v2
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA B200
minimum_gpu_count: 8
precision_quantization: FP4
initial_deployment: SGLang TP=8；DP/EP/MTP 只能在同样 8 卡预算内搜索
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 使用当前 Codex Goal 作为唯一持久循环。远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 agent loop；所有决策、状态总结、完成/阻塞判定都留在本地 Codex Goal 线程。
要求: 每次 benchmark/profile 前必须确认这 8 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 8 张 B200；不要测试多机或更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 llm-serving-auto-benchmark 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，在同一 Goal 中先 profile，再使用 llm-torch-profiler-analysis skill 和 llm-pipeline-analysis skill，必要时使用 ncu-report-skill，随后只 patch 有证据支持的 SGLang 代码路径；重点看 FP4/NVFP4 kernel、MLA、MoE、memory bandwidth、prefill/decode imbalance。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_deepseek_r1_0528_fp4_sota_goal
要求: 固定公平 benchmark 完成前不要选择代码 patch；benchmark 之后的 gap decision、profile、llm-pipeline-analysis skill、patch、revalidate 全部属于当前 Codex Goal 的持续迭代。
要求: 每一轮都更新 artifact_root 下的 manifest、benchmark/profile/analysis 证据、失败尝试和下一步判断；Goal 只有在证据满足完成条件后才能完成。
要求: 如果达到预算上限、GPU 资源阻塞、benchmark/profile 无法获得可信证据，或者没有可辩护的下一步 patch，就停止实质性工作并报告已尝试路径、证据、阻塞点和继续推进所需输入。
```

### moonshotai/Kimi-K2-Instruct
```text
/goal 持续推进 `moonshotai/Kimi-K2-Instruct` 在 single-node 8x NVIDIA B200 上的 SGLang SOTA serving 优化，直到 SGLang 在同样 8 张 B200、FP8、相同 workload 和 SLA 下匹配或超越可复现的最佳 vLLM/TensorRT-LLM 结果；由固定公平 benchmark、profile、llm-pipeline-analysis skill、必要的 NCU 证据、最终报告以及 PR 中的 benchmark/GSM8K/MMLU 全量精度表验证，同时保持正确性、精度和环境安全约束不回退。Goal 本身就是循环，所有迭代都在当前 Goal 中推进。
model_id: moonshotai/Kimi-K2-Instruct
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA B200
minimum_gpu_count: 8
precision_quantization: FP8
initial_deployment: SGLang TP=8；DP/EP 只能在同样 8 卡预算内搜索
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 使用当前 Codex Goal 作为唯一持久循环。远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 agent loop；所有决策、状态总结、完成/阻塞判定都留在本地 Codex Goal 线程。
要求: 每次 benchmark/profile 前必须确认这 8 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 8 张 B200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 llm-serving-auto-benchmark 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，在同一 Goal 中先 profile，再使用 llm-torch-profiler-analysis skill 和 llm-pipeline-analysis skill，必要时使用 ncu-report-skill，随后只 patch 有证据支持的 SGLang 代码路径；重点看 MLA/MoE/DP-attention、long-context decode、framework parity。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_kimi_k2_instruct_sota_goal
要求: 固定公平 benchmark 完成前不要选择代码 patch；benchmark 之后的 gap decision、profile、llm-pipeline-analysis skill、patch、revalidate 全部属于当前 Codex Goal 的持续迭代。
要求: 每一轮都更新 artifact_root 下的 manifest、benchmark/profile/analysis 证据、失败尝试和下一步判断；Goal 只有在证据满足完成条件后才能完成。
要求: 如果达到预算上限、GPU 资源阻塞、benchmark/profile 无法获得可信证据，或者没有可辩护的下一步 patch，就停止实质性工作并报告已尝试路径、证据、阻塞点和继续推进所需输入。
```

### MiniMaxAI/MiniMax-M2.7
```text
/goal 持续推进 `MiniMaxAI/MiniMax-M2.7` 在 single-node 8x NVIDIA B200 上的 SGLang SOTA serving 优化，直到 SGLang 在同样 8 张 B200、FP8、相同 workload 和 SLA 下匹配或超越可复现的最佳 vLLM/TensorRT-LLM 结果；由固定公平 benchmark、profile、llm-pipeline-analysis skill、必要的 NCU 证据、最终报告以及 PR 中的 benchmark/GSM8K/MMLU 全量精度表验证，同时保持正确性、精度和环境安全约束不回退。Goal 本身就是循环，所有迭代都在当前 Goal 中推进。
model_id: MiniMaxAI/MiniMax-M2.7
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA B200
minimum_gpu_count: 8
precision_quantization: FP8
initial_deployment: SGLang TP=8
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 使用当前 Codex Goal 作为唯一持久循环。远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 agent loop；所有决策、状态总结、完成/阻塞判定都留在本地 Codex Goal 线程。
要求: 每次 benchmark/profile 前必须确认这 8 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 8 张 B200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 llm-serving-auto-benchmark 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，在同一 Goal 中先 profile，再使用 llm-torch-profiler-analysis skill 和 llm-pipeline-analysis skill，必要时使用 ncu-report-skill，随后只 patch 有证据支持的 SGLang 代码路径；重点看 MoE、attention、memory pressure、decode throughput。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_minimax_m27_sota_goal
要求: 固定公平 benchmark 完成前不要选择代码 patch；benchmark 之后的 gap decision、profile、llm-pipeline-analysis skill、patch、revalidate 全部属于当前 Codex Goal 的持续迭代。
要求: 每一轮都更新 artifact_root 下的 manifest、benchmark/profile/analysis 证据、失败尝试和下一步判断；Goal 只有在证据满足完成条件后才能完成。
要求: 如果达到预算上限、GPU 资源阻塞、benchmark/profile 无法获得可信证据，或者没有可辩护的下一步 patch，就停止实质性工作并报告已尝试路径、证据、阻塞点和继续推进所需输入。
```

### inclusionAI/Ring-2.5-1T
```text
/goal 持续推进 `inclusionAI/Ring-2.5-1T` 在 single-node 8x NVIDIA B200 上的 SGLang SOTA serving 优化，直到 SGLang 在同样 8 张 B200、FP8、相同 workload 和 SLA 下匹配或超越可复现的最佳 vLLM/TensorRT-LLM 结果；由固定公平 benchmark、profile、llm-pipeline-analysis skill、必要的 NCU 证据、最终报告以及 PR 中的 benchmark/GSM8K/MMLU 全量精度表验证，同时保持正确性、精度和环境安全约束不回退。Goal 本身就是循环，所有迭代都在当前 Goal 中推进。
model_id: inclusionAI/Ring-2.5-1T
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA B200
minimum_gpu_count: 8
precision_quantization: FP8
initial_deployment: SGLang TP=8
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 使用当前 Codex Goal 作为唯一持久循环。远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 agent loop；所有决策、状态总结、完成/阻塞判定都留在本地 Codex Goal 线程。
要求: 每次 benchmark/profile 前必须确认这 8 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 8 张 B200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 llm-serving-auto-benchmark 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，在同一 Goal 中先 profile，再使用 llm-torch-profiler-analysis skill 和 llm-pipeline-analysis skill，必要时使用 ncu-report-skill，随后只 patch 有证据支持的 SGLang 代码路径；优先保证稳定 launch、memory/capacity 证据和 profile-driven root cause。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_ring25_1t_sota_goal
要求: 固定公平 benchmark 完成前不要选择代码 patch；benchmark 之后的 gap decision、profile、llm-pipeline-analysis skill、patch、revalidate 全部属于当前 Codex Goal 的持续迭代。
要求: 每一轮都更新 artifact_root 下的 manifest、benchmark/profile/analysis 证据、失败尝试和下一步判断；Goal 只有在证据满足完成条件后才能完成。
要求: 如果达到预算上限、GPU 资源阻塞、benchmark/profile 无法获得可信证据，或者没有可辩护的下一步 patch，就停止实质性工作并报告已尝试路径、证据、阻塞点和继续推进所需输入。
```

### deepseek-ai/DeepSeek-Math-V2
```text
/goal 持续推进 `deepseek-ai/DeepSeek-Math-V2` 在 single-node 8x NVIDIA B200 上的 SGLang SOTA serving 优化，直到 SGLang 在同样 8 张 B200、BF16、相同 workload 和 SLA 下匹配或超越可复现的最佳 vLLM/TensorRT-LLM 结果；由固定公平 benchmark、profile、llm-pipeline-analysis skill、必要的 NCU 证据、最终报告以及 PR 中的 benchmark/GSM8K/MMLU 全量精度表验证，同时保持正确性、精度和环境安全约束不回退。Goal 本身就是循环，所有迭代都在当前 Goal 中推进。
model_id: deepseek-ai/DeepSeek-Math-V2
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA B200
minimum_gpu_count: 8
precision_quantization: BF16
initial_deployment: SGLang TP=8；DP 只能在同样 8 卡预算内搜索
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 使用当前 Codex Goal 作为唯一持久循环。远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 agent loop；所有决策、状态总结、完成/阻塞判定都留在本地 Codex Goal 线程。
要求: 每次 benchmark/profile 前必须确认这 8 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 8 张 B200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 llm-serving-auto-benchmark 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，在同一 Goal 中先 profile，再使用 llm-torch-profiler-analysis skill 和 llm-pipeline-analysis skill，必要时使用 ncu-report-skill，随后只 patch 有证据支持的 SGLang 代码路径。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_deepseek_math_v2_sota_goal
要求: 固定公平 benchmark 完成前不要选择代码 patch；benchmark 之后的 gap decision、profile、llm-pipeline-analysis skill、patch、revalidate 全部属于当前 Codex Goal 的持续迭代。
要求: 每一轮都更新 artifact_root 下的 manifest、benchmark/profile/analysis 证据、失败尝试和下一步判断；Goal 只有在证据满足完成条件后才能完成。
要求: 如果达到预算上限、GPU 资源阻塞、benchmark/profile 无法获得可信证据，或者没有可辩护的下一步 patch，就停止实质性工作并报告已尝试路径、证据、阻塞点和继续推进所需输入。
```

### zai-org/GLM-5-FP8
```text
/goal 持续推进 `zai-org/GLM-5-FP8` 在 single-node 8x NVIDIA B200 上的 SGLang SOTA serving 优化，直到 SGLang 在同样 8 张 B200、FP8、相同 workload 和 SLA 下匹配或超越可复现的最佳 vLLM/TensorRT-LLM 结果；由固定公平 benchmark、profile、llm-pipeline-analysis skill、必要的 NCU 证据、最终报告以及 PR 中的 benchmark/GSM8K/MMLU 全量精度表验证，同时保持正确性、精度和环境安全约束不回退。Goal 本身就是循环，所有迭代都在当前 Goal 中推进。
model_id: zai-org/GLM-5-FP8
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA B200
minimum_gpu_count: 8
precision_quantization: FP8
initial_deployment: SGLang TP=8；DP/MTP 只能在同样 8 卡预算内搜索
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 使用当前 Codex Goal 作为唯一持久循环。远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 agent loop；所有决策、状态总结、完成/阻塞判定都留在本地 Codex Goal 线程。
要求: 每次 benchmark/profile 前必须确认这 8 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 8 张 B200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 llm-serving-auto-benchmark 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，在同一 Goal 中先 profile，再使用 llm-torch-profiler-analysis skill 和 llm-pipeline-analysis skill，必要时使用 ncu-report-skill，随后只 patch 有证据支持的 SGLang 代码路径；重点看 MTP/speculative、long-context behavior、framework parity。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_glm5_fp8_sota_goal
要求: 固定公平 benchmark 完成前不要选择代码 patch；benchmark 之后的 gap decision、profile、llm-pipeline-analysis skill、patch、revalidate 全部属于当前 Codex Goal 的持续迭代。
要求: 每一轮都更新 artifact_root 下的 manifest、benchmark/profile/analysis 证据、失败尝试和下一步判断；Goal 只有在证据满足完成条件后才能完成。
要求: 如果达到预算上限、GPU 资源阻塞、benchmark/profile 无法获得可信证据，或者没有可辩护的下一步 patch，就停止实质性工作并报告已尝试路径、证据、阻塞点和继续推进所需输入。
```

### zai-org/GLM-4.6-FP8
```text
/goal 持续推进 `zai-org/GLM-4.6-FP8` 在 single-node 8x NVIDIA B200 上的 SGLang SOTA serving 优化，直到 SGLang 在同样 8 张 B200、FP8、相同 workload 和 SLA 下匹配或超越可复现的最佳 vLLM/TensorRT-LLM 结果；由固定公平 benchmark、profile、llm-pipeline-analysis skill、必要的 NCU 证据、最终报告以及 PR 中的 benchmark/GSM8K/MMLU 全量精度表验证，同时保持正确性、精度和环境安全约束不回退。Goal 本身就是循环，所有迭代都在当前 Goal 中推进。
model_id: zai-org/GLM-4.6-FP8
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA B200
minimum_gpu_count: 8
precision_quantization: FP8
initial_deployment: SGLang TP=8；DP/EP/MTP 只能在同样 8 卡预算内搜索
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 使用当前 Codex Goal 作为唯一持久循环。远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 agent loop；所有决策、状态总结、完成/阻塞判定都留在本地 Codex Goal 线程。
要求: 每次 benchmark/profile 前必须确认这 8 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 8 张 B200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 llm-serving-auto-benchmark 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，在同一 Goal 中先 profile，再使用 llm-torch-profiler-analysis skill 和 llm-pipeline-analysis skill，必要时使用 ncu-report-skill，随后只 patch 有证据支持的 SGLang 代码路径。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_glm46_fp8_sota_goal
要求: 固定公平 benchmark 完成前不要选择代码 patch；benchmark 之后的 gap decision、profile、llm-pipeline-analysis skill、patch、revalidate 全部属于当前 Codex Goal 的持续迭代。
要求: 每一轮都更新 artifact_root 下的 manifest、benchmark/profile/analysis 证据、失败尝试和下一步判断；Goal 只有在证据满足完成条件后才能完成。
要求: 如果达到预算上限、GPU 资源阻塞、benchmark/profile 无法获得可信证据，或者没有可辩护的下一步 patch，就停止实质性工作并报告已尝试路径、证据、阻塞点和继续推进所需输入。
```

### openai/gpt-oss-120b
```text
/goal 持续推进 `openai/gpt-oss-120b` 在 single-node 8x NVIDIA B200 上的 SGLang SOTA serving 优化，直到 SGLang 在同样 8 张 B200、MXFP4、相同 workload 和 SLA 下匹配或超越可复现的最佳 vLLM/TensorRT-LLM 结果；由固定公平 benchmark、profile、llm-pipeline-analysis skill、必要的 NCU 证据、最终报告以及 PR 中的 benchmark/GSM8K/MMLU 全量精度表验证，同时保持正确性、精度和环境安全约束不回退。Goal 本身就是循环，所有迭代都在当前 Goal 中推进。
model_id: openai/gpt-oss-120b
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA B200
minimum_gpu_count: 8
precision_quantization: MXFP4
initial_deployment: SGLang TP=8
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 使用当前 Codex Goal 作为唯一持久循环。远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 agent loop；所有决策、状态总结、完成/阻塞判定都留在本地 Codex Goal 线程。
要求: 每次 benchmark/profile 前必须确认这 8 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 8 张 B200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 llm-serving-auto-benchmark 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，在同一 Goal 中先 profile，再使用 llm-torch-profiler-analysis skill 和 llm-pipeline-analysis skill，必要时使用 ncu-report-skill，随后只 patch 有证据支持的 SGLang 代码路径。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_gpt_oss_120b_sota_goal
要求: 固定公平 benchmark 完成前不要选择代码 patch；benchmark 之后的 gap decision、profile、llm-pipeline-analysis skill、patch、revalidate 全部属于当前 Codex Goal 的持续迭代。
要求: 每一轮都更新 artifact_root 下的 manifest、benchmark/profile/analysis 证据、失败尝试和下一步判断；Goal 只有在证据满足完成条件后才能完成。
要求: 如果达到预算上限、GPU 资源阻塞、benchmark/profile 无法获得可信证据，或者没有可辩护的下一步 patch，就停止实质性工作并报告已尝试路径、证据、阻塞点和继续推进所需输入。
```

### internlm/Intern-S1-FP8
```text
/goal 持续推进 `internlm/Intern-S1-FP8` 在 single-node 8x NVIDIA B200 上的 SGLang SOTA serving 优化，直到 SGLang 在同样 8 张 B200、FP8、相同 workload 和 SLA 下匹配或超越可复现的最佳 vLLM/TensorRT-LLM 结果；由固定公平 benchmark、profile、llm-pipeline-analysis skill、必要的 NCU 证据、最终报告以及 PR 中的 benchmark/GSM8K/MMLU 全量精度表验证，同时保持正确性、精度和环境安全约束不回退。Goal 本身就是循环，所有迭代都在当前 Goal 中推进。
model_id: internlm/Intern-S1-FP8
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA B200
minimum_gpu_count: 8
precision_quantization: FP8
initial_deployment: SGLang TP=8, EP=2
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 使用当前 Codex Goal 作为唯一持久循环。远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 agent loop；所有决策、状态总结、完成/阻塞判定都留在本地 Codex Goal 线程。
要求: 每次 benchmark/profile 前必须确认这 8 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 8 张 B200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 llm-serving-auto-benchmark 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，在同一 Goal 中先 profile，再使用 llm-torch-profiler-analysis skill 和 llm-pipeline-analysis skill，必要时使用 ncu-report-skill，随后只 patch 有证据支持的 SGLang 代码路径。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_intern_s1_fp8_sota_goal
要求: 固定公平 benchmark 完成前不要选择代码 patch；benchmark 之后的 gap decision、profile、llm-pipeline-analysis skill、patch、revalidate 全部属于当前 Codex Goal 的持续迭代。
要求: 每一轮都更新 artifact_root 下的 manifest、benchmark/profile/analysis 证据、失败尝试和下一步判断；Goal 只有在证据满足完成条件后才能完成。
要求: 如果达到预算上限、GPU 资源阻塞、benchmark/profile 无法获得可信证据，或者没有可辩护的下一步 patch，就停止实质性工作并报告已尝试路径、证据、阻塞点和继续推进所需输入。
```

### Qwen/Qwen3-235B-A22B-FP8
```text
/goal 持续推进 `Qwen/Qwen3-235B-A22B-FP8` 在 single-node 8x NVIDIA B200 上的 SGLang SOTA serving 优化，直到 SGLang 在同样 8 张 B200、FP8、相同 workload 和 SLA 下匹配或超越可复现的最佳 vLLM/TensorRT-LLM 结果；由固定公平 benchmark、profile、llm-pipeline-analysis skill、必要的 NCU 证据、最终报告以及 PR 中的 benchmark/GSM8K/MMLU 全量精度表验证，同时保持正确性、精度和环境安全约束不回退。Goal 本身就是循环，所有迭代都在当前 Goal 中推进。
model_id: Qwen/Qwen3-235B-A22B-FP8
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA B200
minimum_gpu_count: 8
precision_quantization: FP8
initial_deployment: SGLang TP=8, EP=2
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 使用当前 Codex Goal 作为唯一持久循环。远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 agent loop；所有决策、状态总结、完成/阻塞判定都留在本地 Codex Goal 线程。
要求: 每次 benchmark/profile 前必须确认这 8 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 8 张 B200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 llm-serving-auto-benchmark 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，在同一 Goal 中先 profile，再使用 llm-torch-profiler-analysis skill 和 llm-pipeline-analysis skill，必要时使用 ncu-report-skill，随后只 patch 有证据支持的 SGLang 代码路径；重点看 MoE/EP、attention、long-context prefill/decode。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_qwen3_235b_a22b_fp8_sota_goal
要求: 固定公平 benchmark 完成前不要选择代码 patch；benchmark 之后的 gap decision、profile、llm-pipeline-analysis skill、patch、revalidate 全部属于当前 Codex Goal 的持续迭代。
要求: 每一轮都更新 artifact_root 下的 manifest、benchmark/profile/analysis 证据、失败尝试和下一步判断；Goal 只有在证据满足完成条件后才能完成。
要求: 如果达到预算上限、GPU 资源阻塞、benchmark/profile 无法获得可信证据，或者没有可辩护的下一步 patch，就停止实质性工作并报告已尝试路径、证据、阻塞点和继续推进所需输入。
```

### Qwen/Qwen3-VL-235B-A22B-Instruct-FP8
```text
/goal 持续推进 `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8` 在 single-node 8x NVIDIA B200 上的 SGLang SOTA serving 优化，直到 SGLang 在同样 8 张 B200、FP8、相同 workload 和 SLA 下匹配或超越可复现的最佳 vLLM/TensorRT-LLM 结果；由固定公平 benchmark、profile、llm-pipeline-analysis skill、必要的 NCU 证据、最终报告以及 PR 中的 benchmark/GSM8K/MMLU 全量精度表验证，同时保持正确性、精度和环境安全约束不回退。Goal 本身就是循环，所有迭代都在当前 Goal 中推进。
model_id: Qwen/Qwen3-VL-235B-A22B-Instruct-FP8
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA B200
minimum_gpu_count: 8
precision_quantization: FP8
initial_deployment: SGLang TP=8, EP=2
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 使用当前 Codex Goal 作为唯一持久循环。远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 agent loop；所有决策、状态总结、完成/阻塞判定都留在本地 Codex Goal 线程。
要求: 每次 benchmark/profile 前必须确认这 8 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 8 张 B200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 llm-serving-auto-benchmark 默认 workload，并保留 VLM 输入兼容性验证。
要求: 如果 SGLang 稳定落后超过 1%，在同一 Goal 中先 profile，再使用 llm-torch-profiler-analysis skill 和 llm-pipeline-analysis skill，必要时使用 ncu-report-skill，随后只 patch 有证据支持的 SGLang 代码路径；重点看 VLM preprocess、MoE/EP、attention、decode throughput。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_qwen3_vl_235b_a22b_instruct_fp8_sota_goal
要求: 固定公平 benchmark 完成前不要选择代码 patch；benchmark 之后的 gap decision、profile、llm-pipeline-analysis skill、patch、revalidate 全部属于当前 Codex Goal 的持续迭代。
要求: 每一轮都更新 artifact_root 下的 manifest、benchmark/profile/analysis 证据、失败尝试和下一步判断；Goal 只有在证据满足完成条件后才能完成。
要求: 如果达到预算上限、GPU 资源阻塞、benchmark/profile 无法获得可信证据，或者没有可辩护的下一步 patch，就停止实质性工作并报告已尝试路径、证据、阻塞点和继续推进所需输入。
```

### meta-llama/Llama-4-Scout-17B-16E-Instruct
```text
/goal 持续推进 `meta-llama/Llama-4-Scout-17B-16E-Instruct` 在 single-node 8x NVIDIA B200 上的 SGLang SOTA serving 优化，直到 SGLang 在同样 8 张 B200、BF16、相同 workload 和 SLA 下匹配或超越可复现的最佳 vLLM/TensorRT-LLM 结果；由固定公平 benchmark、profile、llm-pipeline-analysis skill、必要的 NCU 证据、最终报告以及 PR 中的 benchmark/GSM8K/MMLU 全量精度表验证，同时保持正确性、精度和环境安全约束不回退。Goal 本身就是循环，所有迭代都在当前 Goal 中推进。
model_id: meta-llama/Llama-4-Scout-17B-16E-Instruct
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 8x NVIDIA B200
minimum_gpu_count: 8
precision_quantization: BF16
initial_deployment: SGLang TP=8
要求: 远端使用 ion-b200；SGLang 使用已有 sglang_bbuf 容器，容器内 repo 为 /home/sglang-omni/bbuf/repos/sglang。
要求: vLLM 和 TensorRT-LLM 直接使用最新镜像 vllm/vllm-openai:latest 与 nvcr.io/nvidia/tensorrt-llm/release:latest。
要求: 做环境准备时这台机器只执行一次 git pull；本任务开始前必须确认容器内 SGLang、vLLM、TensorRT-LLM 没有本地修改：repo/workspace 要是干净分支，latest 镜像不能挂载带本地改动的 repo；如果有本地修改、git pull 失败或没有 upstream，停止并报告 blocker，不要 reset/rebase/覆盖未知改动。
要求: 做当前模型 benchmark/profile/patch 前，必须先查询 sgl-project/sglang 和 BBuf/sglang 的 open PR，确认是否已有针对当前 model_id 或同模型家族的性能/正确性优化；如果有，优先在独立分支或临时工作树把这些相关 PR 合入或逐个试跑，记录 PR、commit/patch、命令和结果，用包含相关 open PR 的最佳可复现结果作为 SGLang 候选 baseline；不要把未包含这些 open PR 的 main 分支性能直接当作 baseline。
要求: 使用当前 Codex Goal 作为唯一持久循环。远端 ion-b200 只能作为执行、benchmark、profile、验证环境，不要在远端启动 agent loop；所有决策、状态总结、完成/阻塞判定都留在本地 Codex Goal 线程。
要求: 每次 benchmark/profile 前必须确认这 8 张 B200 没有其他人的重负载进程，并记录 nvidia-smi、进程、显存、利用率、CUDA_VISIBLE_DEVICES；受干扰的数据不可信。
要求: 如果暂时没有足够的空闲资源继续把任务做下去，就等待半小时；如果半小时内仍然没有可用资源，再停止并报告 blocker。
要求: 只使用 8 张 B200；不要测试更多卡。
要求: 对 SGLang、vLLM、TensorRT-LLM 做同样 8 卡预算下的公平搜索；使用 llm-serving-auto-benchmark 默认 workload。
要求: 如果 SGLang 稳定落后超过 1%，在同一 Goal 中先 profile，再使用 llm-torch-profiler-analysis skill 和 llm-pipeline-analysis skill，必要时使用 ncu-report-skill，随后只 patch 有证据支持的 SGLang 代码路径；重点看 MoE execution、attention backend、overlap、decode throughput。
要求: 当前模型的所有任务文件都必须放在上面的 artifact_root 独立目录；任务完成后只删除容器内当前模型 Hugging Face 权重缓存，例如 /root/.cache/huggingface/hub/models--<org>--<repo> 及对应当前模型 lock；不要删除其他模型、共享 cache、镜像或别人的容器。
要求: 如果需要提交 PR，只能 push/open 到 BBuf/sglang；不要 push 到或向 sgl-project/sglang 开 PR。
要求: 一个任务可以提交多个优化 PR 来推进模型性能；只要所有优化 PR 叠加后的效果让 SGLang 在该模型上超越或持平其它框架，就算完成目标；每个优化 PR 的 PR 描述都必须用表格给出性能 benchmark 对比，以及 GSM8K、MMLU 全量精度对比。
artifact_root: /Users/bbuf/工作目录/Common/opt_model/b200_llama4_scout_17b_16e_instruct_sota_goal
要求: 固定公平 benchmark 完成前不要选择代码 patch；benchmark 之后的 gap decision、profile、llm-pipeline-analysis skill、patch、revalidate 全部属于当前 Codex Goal 的持续迭代。
要求: 每一轮都更新 artifact_root 下的 manifest、benchmark/profile/analysis 证据、失败尝试和下一步判断；Goal 只有在证据满足完成条件后才能完成。
要求: 如果达到预算上限、GPU 资源阻塞、benchmark/profile 无法获得可信证据，或者没有可辩护的下一步 patch，就停止实质性工作并报告已尝试路径、证据、阻塞点和继续推进所需输入。
```
