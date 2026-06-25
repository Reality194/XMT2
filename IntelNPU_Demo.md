# Intel NPU（OpenVINO）示例说明

## 概述
本示例演示如何使用 OpenVINO Runtime 将一个 ONNX 模型部署到 Intel NPU（例如使用 MYRIAD/VPU 的设备）并运行一次推理。脚本会：

- 自动下载 ResNet50 ONNX 模型（来自 Hugging Face `Xenova/resnet-50`，如果本地未找到）。
- 检测可用设备并优先选择 `MYRIAD`（若不可用则回退到 `CPU`）。
- 构造随机输入运行一次前向推理并输出耗时与 top-5 预测结果索引。

## 文件
- `intel_npu_demo.py`：演示脚本（主脚本）。
- `requirements.txt`：建议的 Python 依赖。

## 先决条件
- Python 3.8+。
- 安装依赖：

```bash
pip install -r requirements.txt
```

- 如果要在物理 Intel NPU（如 Intel Movidius Neural Compute Stick）上运行：
  - 请确保设备驱动和固件已安装并连接到主机。
  - OpenVINO 的 NPU 后端（例如 MYRIAD）可用。

## 运行
在仓库根目录运行：

```bash
python intel_npu_demo.py
```

脚本会打印可用设备、选择的设备、一次推理耗时和 top-5 结果。

## 关于设备名称
- `MYRIAD`：常见用于 Intel Movidius VPU（例如 Neural Compute Stick）。
- `CPU`：通用回退选项。
- 其他设备名如 `GPU`、`HDDL` 等，取决于你的 OpenVINO 安装与硬件平台。

## 可能的问题与排查
- 如果脚本报错找不到 `openvino`，请通过 `pip install openvino` 安装。
- 若检测不到 `MYRIAD` 但设备已连接：请确认驱动/固件、以及 OpenVINO 的 VPU 插件已正确安装。

- 若无法通过脚本自动下载模型（网络或证书问题），请手动下载模型并放到项目根目录，推荐链接：

```
https://huggingface.co/Xenova/resnet-50/resolve/main/onnx/model.onnx
```
  下载后确保文件名为 `xenova-resnet50.onnx`。

## 扩展建议
- 用真实图片替换随机输入，并添加后处理以映射到类别标签（需要下载相应的 labels 文件）。
- 使用 OpenVINO 的模型优化工具预转换并量化模型以获得更好性能。

