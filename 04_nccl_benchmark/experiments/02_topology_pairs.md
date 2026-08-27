# 实验 02：GPU pair 与拓扑

`pair01`、`pair02` 只是设备编号，不预设哪一对更近。运行前保存：

```bash
nvidia-smi -L
nvidia-smi topo -m
```

正式配置比较 AllReduce 的 `[0,1]`、`[0,2]` 和 `[0,1,2,3]`。只有实际拓扑显示二者
路径不同，才能把差异解释为 NUMA/PCIe/NVLink 路径影响。一次只改变设备集合，消息
大小、dtype、warm-up、迭代和软件环境保持不变。

