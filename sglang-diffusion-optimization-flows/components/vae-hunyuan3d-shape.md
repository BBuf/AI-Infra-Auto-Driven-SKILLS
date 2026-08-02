# Hunyuan3D shape VAE flow

实现：`runtime/models/vaes/hunyuan3d_vae.py::ShapeVAE`，只用于 Hunyuan3D-2。

1. 从 `hunyuan3d-shape` preset 保存 point/latent/mesh 中间结果。
2. 分离 shape VAE decode、surface extraction/mesh 后处理和条件 encoder；profile
   attention/GEMM/scatter，不把 marching-cubes 类后处理误归因给 VAE。
3. 对真实点数/latent shape 做 microbench，检查 padding、indexing、layout 和空 tensor。
4. 比较 latent cosine `>=0.999`、顶点/面数量、有限值、包围盒和抽样 Chamfer；
   固定输入的 mesh 必须可加载、无明显拓扑破损。
5. 报 component、shape stage 和完整 mesh E2E，至少 3% 且超过方差才接受。
