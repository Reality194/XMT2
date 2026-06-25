"""
Intel NPU (OpenVINO) 示范脚本

说明：
- 本脚本使用 OpenVINO Runtime 加载一个 ONNX 模型（MobileNetV2），
  尝试在 Intel NPU（MYRIAD/VPU）上运行推理；如果不可用会回退到 CPU。
- 脚本会自动下载模型（如果本地不存在），构造一个随机输入并运行一次前向推理，
  输出推理耗时和 top-5 结果索引。

注意：运行该脚本前需安装 OpenVINO（pip install openvino）并确保 NPU 驱动/固件可用。
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import os
import sys
import time
import numpy as np



# 使用可公开访问的 Hugging Face ONNX 模型（已验证可下载）
MODEL_URL = "https://huggingface.co/Xenova/resnet-50/resolve/main/onnx/model.onnx"
MODEL_FILE = "xenova-resnet50.onnx"


# Hugging Face 支持：如果环境变量 `USE_HF` 为 "1" 或设置了 `HF_MODEL`，脚本会使用
# transformers pipeline 做一次图像分类推理（不经过 OpenVINO）。
HF_USE = os.environ.get("USE_HF", os.environ.get("USE_HUGGINGFACE", "0"))
HF_MODEL = os.environ.get("HF_MODEL")


def download_model(url=MODEL_URL, target=MODEL_FILE):
    if os.path.exists(target):
        print(f"模型已存在：{target}")
        return target
    try:
        import requests
    except Exception:
        print("缺少 requests 库，尝试通过 pip 安装：pip install requests")
        raise
    print(f"正在下载模型：{url} -> {target}")
    # 优先使用 certifi 的 CA 证书包以避免系统缺失根证书导致的验证失败
    verify_arg = True
    try:
        import certifi
        verify_arg = certifi.where()
    except Exception:
        # 没有 certifi 时继续使用 requests 的默认验证（可能会失败）
        print("警告：未安装 certifi，若出现 SSL 验证错误请运行 `pip install certifi`。")
        verify_arg = True
    try:
        r = requests.get(url, stream=True, verify=verify_arg)
        r.raise_for_status()
    except requests.exceptions.SSLError as e:
        # 在 SSL 验证失败时，提示并回退到不验证（仅在用户信任网络时使用）
        print("SSL 验证失败：", e)
        print("尝试不验证 SSL（不推荐）。如果你想要安全的解决方案，请安装 certifi 并重试：pip install certifi")
        r = requests.get(url, stream=True, verify=False)
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        # 更友好的 404 提示
        status = None
        try:
            status = e.response.status_code
        except Exception:
            pass
        if status == 404:
            print("模型未找到 (404)。可能的原因是 URL 已更改或模型已移动。")
            print("请从 Hugging Face 下载模型并将其放到项目根目录，推荐链接：", MODEL_URL)
        raise
    with open(target, "wb") as f:
        for chunk in r.iter_content(1024 * 64):
            if chunk:
                f.write(chunk)
    print("模型下载完成")
    return target


def prepare_input(shape):
    # 将动态维度替换为 1
    fixed = [1 if s is None or s == 0 else int(s) for s in shape]
    # 构造随机输入，范围符合一般图像输入 (0-255)
    data = np.random.rand(*fixed).astype(np.float32)
    return data


def main():
    # 如果用户要求使用 Hugging Face 模型，优先走 transformers pipeline
    if HF_USE == "1" or HF_MODEL:
        hf_model = HF_MODEL or "google/vit-base-patch16-224"
        print(f"检测到 Hugging Face 模式，尝试加载模型：{hf_model}")
        try:
            from transformers import pipeline
        except Exception:
            print("缺少 transformers 库。请安装：pip install transformers[torch] 或 pip install transformers" )
            return
        try:
            from PIL import Image
        except Exception:
            print("缺少 Pillow 库。请安装：pip install pillow")
            return

        # 构造一个随机图像作为演示输入
        img_size = 224
        img_arr = (np.random.rand(img_size, img_size, 3) * 255).astype('uint8')
        img = Image.fromarray(img_arr)

        print("正在从 Hugging Face 加载并运行推理（可能会下载模型到本地缓存）...")
        try:
            classifier = pipeline("image-classification", model=hf_model)
            results = classifier(img, top_k=5)
        except Exception as e:
            print("Hugging Face 推理失败：", e)
            return

        print("Hugging Face top-5 结果：")
        for r in results:
            print(f"  {r['label']}: {r['score']:.4f}")
        return

    try:
        from openvino.runtime import Core
    except Exception as e:
        print("请先安装 OpenVINO Runtime：pip install openvino")
        raise

    # 下载模型（如果需要）
    try:
        model_path = download_model()
    except Exception as e:
        print("模型下载失败：", e)
        return

    core = Core()
    devices = core.available_devices
    print("可用设备：", devices)

    # 优先使用 MYRIAD（常见的 Intel VPU/Movidius 设备名），否则回退 CPU
    preferred = "MYRIAD"
    device = preferred if preferred in devices else "NPU" if "NPU" in devices else "GPU" if "GPU" in devices else "CPU"
    print(f"将使用设备：{device}")

    # 读取模型并编译到选定设备
    model = core.read_model(model=model_path)

    # 获取输入形状（取第一个 input）；若为动态形状则尝试设为常见图像尺寸
    input_tensor = model.inputs[0]
    try:
        input_shape = input_tensor.shape
    except RuntimeError:
        # 某些 ONNX 模型输入是动态 shape，先进行 reshape 再读取
        in_name = input_tensor.get_any_name()
        reshape_shape = [1, 3, 224, 224]
        print(f"检测到动态输入形状，尝试 reshape {in_name} -> {reshape_shape}")
        model.reshape({in_name: reshape_shape})
        input_tensor = model.inputs[0]
        input_shape = input_tensor.shape
    print("模型输入形状：", input_shape)

    input_data = prepare_input(input_shape)

    compiled = core.compile_model(model=model, device_name=device)

    # 执行 1000 次推理并计时（统一使用 infer_request，避免不同版本返回类型差异）
    start = time.time()
    req = compiled.create_infer_request()
    in_name = compiled.input(0).get_any_name()
    for _ in range(10000):
        req.infer({in_name: input_data})
    out0 = req.get_output_tensor(0).data
    end = time.time()

    duration_ms = (end - start)
    print(f"10000 次推理耗时：{duration_ms:.2f} s")

    # 将输出扁平化并取 top-5
    flat = np.array(out0).ravel()
    top5_idx = np.argsort(flat)[-5:][::-1]
    top5_vals = flat[top5_idx]
    print("Top-5 索引：", top5_idx)
    print("Top-5 值：", top5_vals)


if __name__ == "__main__":
    main()
