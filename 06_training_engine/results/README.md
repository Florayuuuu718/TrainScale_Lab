# 06 结果与证据

`raw/` 已被 Git 忽略，用于命令、stdout/stderr、rank JSON 和 profiler trace。汇总 artifact 必须包含
commit/dirty、环境、配置哈希、correctness、测量口径、原始文件 SHA-256 和适用边界。性能数字
不得手工修改。

发布验收为 [`module06-acceptance.json`](module06-acceptance.json)，已审核公开摘要为
[`module06_final_summary.json`](module06_final_summary.json)。它同时覆盖正式
ablation、AMP overflow 和 1 MiB overlap 延伸实验。延伸归档 SHA-256 为
`3baafa93ac3f50ac8112eed4ded6012735bb93146f9c2dd76cafb0934671648f`，33 个 raw artifact
均已验证。完整解释见 [`../experiments/06_final_report.md`](../experiments/06_final_report.md)。
