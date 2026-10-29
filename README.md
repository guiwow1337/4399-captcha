# 4399-captcha



100%识别 `ptlogin.4399.com` 的四位验证码。onnx模型是用matcher识别完验证码后训练出的产物。

下面是跟这个项目([OxygenNEL-OpenNELOCR](https://github.com/MC20018/OxygenNEL-OpenNELOCR))的对比。


<img src="Comparison.png" width="657" />

## 警告

若4399对此类验证码进行了大幅更新 则此方法可能失效


## 用法

### matcher

```bash
./releases/matcher.exe 验证码.jpg       # 输出 4 个字符，一行
CFD6
```

一次一张，单线程，只支持 JPEG(4399使用此格式)。仓库中仅给出实现逻辑，使用时请自己修改。

### model.onnx

见下面「ONNX 模型：配置与用法」，那段里有一份复制就能跑的完整代码。

## 文件

| 文件 | 说明 |
|---|---|
| `matcher.c` | 识别器 |
| `glyphs.png` | 模板图 |
| `templates.h` | 字形坐标表 |
| `stb_image.h` | JPEG 解码（[nothings/stb](https://github.com/nothings/stb)，公有领域 / MIT） |
| `model/model.onnx` | 神经网络权重（ONNX，CPU 推理） |
| `model/model.json` | 神经网络配置：字符集、结构参数、输入尺寸、训练与验证成绩 |
---

## ONNX 模型：配置与用法


```
输入   image    (batch, 3, 30, 60)  float32
输出   p0..p3   各 (batch, 25)      float32
```

**不需要任何配套配置就能加载**。
有两个常量在外边，不写无法正常运行：**字符集顺序**和**预处理**。
前者在 `model/model.json` 里，后者就是下面这几行。

`model/model.json` 的字段：

| 字段 | 值 | 说明 |
|---|---|---|
| `charset` | `23467ACDEFGHJKMNPQRTUVWXY` | 25 个字符索引 |
| `seq_len` | `4` | 4 位验证码 |
| `num_classes` | `25` | 每个输出头的宽度 |
| `img_h` / `img_w` | `30` / `60` | 输入尺寸 |
| `widths` | `[16, 32, 64, 96]` | 分段池化的 4 段宽度（4 个槽位各管一段） |
| `seg_hidden` | `64` | 每个槽位自己的特征宽度 |
| `dropout` | `0.2` | 训练时的 dropout（推理无效） |
| `trained_on` | `8099` | 训练集张数 |
| `val_n` / `val_exact` / `val_per_pos` | `1911` / `0.9979` / `[…]` | 留出集的验证成绩 |

### 预处理

```python
a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
a = (a - 0.5) / 0.5                      # 等价于 /127.5 - 1
x = a.transpose(2, 0, 1)[None]           # (1, 3, 30, 60)
```

1. **保持 RGB，不要转灰度** —— 每个字符颜色不同
2. **不要缩放**，原图就是 60×30；喂别的尺寸会直接掉到 ~0%
3. 输出是 4 个**各自独立**的 25 类 logits，不是 CTC：
   **每位各自 argmax 再查 charset**，不需要合并重复、也不用去 blank

### 完整可跑的代码

```python
import json
import numpy as np
import onnxruntime as ort
from PIL import Image

cfg = json.load(open("model/model.json", encoding="utf-8"))
sess = ort.InferenceSession("model/model.onnx",
                            providers=["CPUExecutionProvider"])

im = Image.open("验证码.jpg").convert("RGB")          # 彩色，别转灰度
a = np.asarray(im, dtype=np.float32) / 255.0
a = (a - 0.5) / 0.5
x = a.transpose(2, 0, 1)[None]                        # (1, 3, 30, 60)

outs = sess.run(None, {"image": x})                   # 4 个 (1, 25)
label = "".join(cfg["charset"][int(o[0].argmax())] for o in outs)
print(label)
```

依赖：`pip install onnxruntime numpy pillow`。





## 声明

作者不为这个克苏鲁产物负任何责任。如果它帮到了你。你可以顺便点一个star。




## 许可证

见 [LICENSE](LICENSE)：**MIT**。

`stb_image.h` 是公有领域 / MIT（[nothings/stb](https://github.com/nothings/stb)），
只用于构建，不改变本项目的许可证。
