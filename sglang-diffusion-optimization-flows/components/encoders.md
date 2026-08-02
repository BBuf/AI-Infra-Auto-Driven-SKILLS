# 条件 encoder 共享 flow

继承 [`../common/execution-contract.md`](../common/execution-contract.md)。覆盖 T5、CLIP、
Qwen/Gemma/Mistral/PaliGemma、图像 encoder、audio feature encoder 等条件阶段。

1. 分开计时 tokenizer/preprocess、CPU 到 GPU、encoder forward、projection/packing。
2. 固定真实 prompt、图片/音频和最大序列长度，记录 hidden shape、dtype、mask。
3. 先查 prompt/prefix cache、重复条件消除、batched encoder、预分配与 offload；
   cache key 必须包含所有会改变 embedding 的输入。
4. 只有 GPU encoder 占 E2E `>=3%` 才下钻 attention/GEMM/kernel；短 prompt 的 Python
   开销不要误归因成 CUDA kernel。
5. encoder output cosine `>=0.98`、无 NaN/Inf；再验证 CFG、负 prompt、多图片、
   动态 prompt/cache reset。

H3 的 Qwen3-VL 条件编码、Pi0.5 的 PaliGemma prefix，以及 JoyEcho 的 audio feature
窗口都必须用各自模型 flow 中的输入合同，不能共享数值 reference。
