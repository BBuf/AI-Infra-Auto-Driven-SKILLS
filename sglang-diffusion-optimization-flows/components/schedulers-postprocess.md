# Scheduler / postprocess / delivery flow

继承 [`../common/execution-contract.md`](../common/execution-contract.md)。仅当这些阶段
占 E2E `>=3%` 或阻断吞吐时立项。

1. 分开计时 scheduler step/random、latent cast/copy、safety/guardrail、image/video
   postprocess、frame interpolation/upscale、audio resample、mux 与 mesh export。
2. scheduler 候选固定 timesteps/sigmas/shifts、generator state 和 dtype；比较每一步
   latent，不能只看最终感知结果。H3 固定专用 Euler ancestral eta=0 adapter。
3. 查找 Python loop、CPU sync、重复 tensor 构造、device transfer 和可批量的 pointwise
   update；GPU 占比低时优先移除同步，而不是写 CUDA kernel。
4. postprocess 候选保持颜色范围、帧率/时间戳、sample rate/channels、codec/container、
   mesh 坐标/拓扑合同；近似 upscale/interpolation 单列质量预算。
5. guardrail 在隔离 benchmark 中可显式关闭，但生产结论必须补开启数字和功能测试。
6. 报 stage 与完整 E2E；输出可打开只是最低门槛，仍需对应图像/视频/音频/mesh 指标。
