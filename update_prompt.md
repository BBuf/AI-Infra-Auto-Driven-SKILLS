# Refresh the skills from source

刷新目标由用户指定；不要把历史机器、日期、模型矩阵或 PR 当成当前环境。

1. 在干净分支/worktree 工作，记录仓库基线。对相关上游冻结 SHA，阅读调用方、
   dispatch、实现与验证接口。更新 `docs/upstream-source-contracts.md`，不只改日期。
2. 区分 main、模型开发分支、open PR 与安装的 wheel。历史 PR 卡片保留原始审计
   日期；新增卡片需要逐份读 diff。不要把刷新一个 source head 写成全部模型已复测。
3. 更新受影响的 skill、引用路径、HTTP/CLI 协议和脚本。详细案例放 references，
   入口只保留判断流程。对于 PDL、fusion、split-K、metadata，核对模型实际 dispatch、
   shape、布局和数值约定，不能用未命中的 microbenchmark 证明端到端收益。
4. 删除过期机器脚本、无调用的 helper、重复描述，以及只锁定固定文字/日期/SHA/
   版本号的测试。保留行为测试：数据丢失、错误归因、解析失败、fallback、数值、
   边界条件、超时和路径处理。删除前查调用方，修好文档与入口。
5. 跑剩余 CPU 测试、cookbook 校验、相关脚本真实输入 smoke、skill frontmatter 和
   相对链接检查。新 capture 不得悄悄复用旧 trace；compact trace 必须保留原始事件。
6. 修改 GPU/模型执行路径才按需要安排 live 验证。先核对有效分配和进程归属，
   保存依赖版本、实际参数和产物；不可用时明确哪些项未运行。只清理由本任务启动
   的服务，不要求共享机器全卡显存清零。
7. 性能测量与 profiler 分开；精度必须是真实接受模式。报告 kernel correctness、
   实际 dispatch、GSM8K/AIME 等任务评测和无 profiler 性能各自的覆盖范围。
8. 提交一个聚焦 PR，写清实际修改、删除项、检查结果及未覆盖项。不要合并 PR。
