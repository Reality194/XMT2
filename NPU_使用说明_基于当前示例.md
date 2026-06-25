# NPU 正确使用说明（基于 `intel_npu_demo.py`）

本文档基于当前示例脚本 `intel_npu_demo.py`，说明如何**正确使用 Intel NPU**进行推理。

## 1. 先明确脚本有两条执行路径

当前脚本有两种模式：

1. OpenVINO 路径（会尝试使用 `MYRIAD/NPU/GPU/CPU`）
2. Hugging Face pipeline 路径（`USE_HF=1` 或设置 `HF_MODEL` 时启用，不走 OpenVINO NPU）

如果你的目标是演示 **Intel NPU**，请走 **OpenVINO 路径**。

## 2. 运行前准备

建议先安装依赖：

```bash
pip install -r requirements.txt
```

如果有 SSL 证书问题，额外确认：

```bash
pip install certifi
```

## 3. 如何以 NPU 方式运行（关键）

### 3.1 不要启用 HF 路径

确保以下环境变量未开启：

- `USE_HF`
- `USE_HUGGINGFACE`
- `HF_MODEL`

PowerShell 可这样清理：

```powershell
Remove-Item Env:USE_HF -ErrorAction SilentlyContinue
Remove-Item Env:USE_HUGGINGFACE -ErrorAction SilentlyContinue
Remove-Item Env:HF_MODEL -ErrorAction SilentlyContinue
python intel_npu_demo.py
```

### 3.2 看设备选择日志

脚本会打印：

- `可用设备： [...]`
- `将使用设备：...`

当前逻辑优先级是：

1. `MYRIAD`
2. `NPU`
3. `GPU`
4. `CPU`

只有当输出是 `MYRIAD` 或 `NPU`，才说明真正走了 NPU。

## 4. 当前脚本中与 NPU 相关的关键点

### 4.1 设备发现

通过 OpenVINO `Core().available_devices` 列出设备，然后按优先级选设备。

### 4.2 模型下载

默认模型：

- `https://huggingface.co/Xenova/resnet-50/resolve/main/onnx/model.onnx`
- 本地文件名：`xenova-resnet50.onnx`

### 4.3 动态输入处理

若模型输入是动态 shape，脚本会自动 reshape 到 `[1, 3, 224, 224]`，避免 `to_shape was called on a dynamic shape` 错误。

### 4.4 推理执行

当前使用 `infer_request` 执行推理并读取 `output_tensor(0).data`，避免不同 OpenVINO 版本返回类型差异。

## 5. 如何判断“正确使用 NPU”

满足以下条件才算 NPU 跑通：

1. 运行日志里 `可用设备` 包含 `NPU` 或 `MYRIAD`
2. 日志 `将使用设备` 显示为 `NPU` 或 `MYRIAD`
3. 推理成功输出耗时与 Top-5 结果

如果最终显示 `CPU`，说明脚本正常回退了，但不是 NPU 推理。

## 6. 常见问题排查

### 6.1 只显示 CPU/GPU，没有 NPU

可能原因：

- NPU 驱动或运行时未安装完整
- OpenVINO 未识别到 NPU 插件
- 设备未正确连接/系统未识别

### 6.2 模型下载失败（401/404/SSL）

- 401/404：模型 URL 不可用或访问策略变化
- SSL：本地证书链问题，先尝试安装 `certifi`

可采用手动下载方式：把模型下载到项目根目录并命名为 `xenova-resnet50.onnx`。

### 6.3 误进入 Hugging Face 路径

如果看到日志 `检测到 Hugging Face 模式`，说明走的是 HF pipeline，不是 NPU 路径。请清理相关环境变量后重跑。

## 7. 一条推荐的最小验证命令

```powershell
Remove-Item Env:USE_HF -ErrorAction SilentlyContinue
Remove-Item Env:USE_HUGGINGFACE -ErrorAction SilentlyContinue
Remove-Item Env:HF_MODEL -ErrorAction SilentlyContinue
python intel_npu_demo.py
```

观察 `将使用设备：NPU`（或 `MYRIAD`）即可确认。
