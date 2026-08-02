# TP / SP / CFG / offload / cache flow

继承 [`../common/execution-contract.md`](../common/execution-contract.md)。

1. 先跑单卡可驻留的最小基线；大模型则记录能够启动的最小 GPU 数。
2. 对合法组合逐个扫描 TP、Ulysses/SP、CFG parallel、FSDP/offload、async A2A，
   固定 workload 和卡数，记录 E2E、denoise、通信占比、峰值显存。
3. 校验约束：`tp_size * sp_degree == num_gpus`（模型有额外合同则以模型为准），
   拒绝静默 fallback、重复 shard、错误 gather 或 rank 间 shape 漂移。
4. cache/quant/少步数是近似质量路线，必须与 lossless 并行路线分表。
5. profile NCCL/A2A/all-gather 的等待与 overlap；若通信不是热点，不为“未来扩展”
   改动 collective。
6. 至少测两种卡数并给 speedup/parallel efficiency；100 次短 soak 检查 cache reset、
   显存增长和动态请求切换。

多机时另外记录节点、NIC、rank mapping、跨机/机内 collective；不得拿单机数字推断
多机收益。
