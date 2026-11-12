# 强引力透镜引力波的校准配对验证

本仓库提供论文所用的核心代码、冻结的评估清单、逐配对预测分数和统计结果。研究比较两种方法：直接处理峰值对齐应变片段的一维 PI-ResNet，以及使用常 Q 变换图像的 CQT--DeiT（SEMD-inspired）基线。

**当前版本 v3.0.0**：论文现行版本的分析位于 [experiments/apjs_revision/v3/](experiments/apjs_revision/v3/)（方案 `PROTOCOL_V3.md`，全部数值 `RESULTS_V3.md`，分级搜索成本模型 `cost_model/`）。透镜与非透镜双黑洞服从观测到的族群分布，星系透镜由 `ler` 抽样，分别对应 O4（H1、L1、V1）与爱因斯坦望远镜（ET）灵敏度。在每对假阳性概率 $10^{-3}$ 下，PI-ResNet 的效率为 36.8%（ET）和 16.1%（O4）；加入时间延迟统计量后为 51.0% 和 64.5%。代码、训练权重和逐配对分数存档于 [doi:10.5281/zenodo.24049483](https://doi.org/10.5281/zenodo.24049483)（MIT 许可），模拟应变数据存档于 [doi:10.5281/zenodo.24049603](https://doi.org/10.5281/zenodo.24049603)。以下内容描述早期版本（v2.2.2）。

本项目解决的是“已识别事件之间的配对验证”，不是完整的搜寻流水线、真实噪声检验或 catalog-level FAR 分析。

主要结果、置信区间和物理诊断见 [docs/RESULTS.md](docs/RESULTS.md)，数据与大文件说明见 [docs/ARTIFACTS.md](docs/ARTIFACTS.md)，基线模型完整配置见 [docs/BASELINE_CONFIG.md](docs/BASELINE_CONFIG.md)，完整执行顺序见 [experiments/reproducibility/README.md](experiments/reproducibility/README.md)。代码及正式技术文档以英文版本为准。

冻结的可复现发布（训练权重、实例变异性研究的十六个附加权重、固定的 DeiT 初始化权重、五个 CQT 缓存及全部派生结果）存档于 Zenodo，引用 concept DOI [10.5281/zenodo.21311077](https://doi.org/10.5281/zenodo.21311077)，它始终指向最新版本（CC BY 4.0）。v2.0 共 389 个经哈希校验的文件，2,944,413,943 字节。引用时请同时标注 Zenodo 记录与所用的仓库 commit。
