# **♻️ 基于两阶段迁移学习的实时垃圾分类系统**

💡 **核心贡献:** 实现了从异构数据（TrashNet & TACO）整合、MobileNetV2 两阶段迁移学习优化，到 Gradio \+ Docker 容器化部署的 **端到端实时垃圾分类** 工作流。

## **📚 目录 (Table of Contents)**

1. [项目概述 (Overview)](#bookmark=id.4o0mcqgr5kzo)  
2. [快速部署 (Quick Deployment)](#bookmark=id.s8j7zomwcrdf)  
3. [系统工作流 (System Workflow)](#bookmark=id.b0yx3n28zwkl)  
4. [数据集构建 (Dataset)](#bookmark=id.cly8kpsbcoax)  
5. [模型方法 (Methodology)](#bookmark=id.kmvf2e6rcz9b)  
   * [5.1 基础模型选择 (MobileNetV2)](#bookmark=id.lnry086a1885)  
   * [5.2 两阶段训练策略 (Two-Stage Training)](#bookmark=id.4tele2y5fodg)  
6. [实验与结果 (Experiments & Results)](#bookmark=id.f9f1hgpqrg3n)   
7. [未来展望 (Future Work)](#bookmark=id.jpbt72kqp5d)  
8. [作者与致谢 (Authors & Acknowledgments)](#bookmark=id.6cez1y9n1q6i)

## **1\. 项目概述 (Overview)**

随着深度学习在计算机视觉领域的飞速发展，自动化垃圾分类已成为解决城市废弃物管理挑战的新兴研究方向。然而，将高精度模型转化为稳定、高效、易于部署的实时应用仍面临挑战，尤其是在处理来源多样、质量不一的真实世界数据时。

本项目提出并实现了一套完整的、基于深度学习的实时垃圾分类识别与部署方案。方案的核心在于构建了一个**从异构数据处理、两阶段模型优化到容器化应用部署**的全流程工作管线。

* **数据层：** 融合 TrashNet (背景统一) 与 TACO (背景复杂) 数据集，并通过自动化脚本解析 TACO 的 COCO JSON 标注，依据边界框 (Bounding Box) 精确裁剪目标物体，创建了高质量的统一训练源。  
* **模型层：** 采用轻量级的 MobileNetV2 作为基础，实施**两阶段迁移学习**（特征提取 \+ 微调），在保证轻量化的同时，将测试集准确率提升至 **90%**。  
* **应用层：** 采用 Gradio 构建实时摄像头交互界面，并通过 Docker 实现便捷的应用交付与可靠部署。

本项目旨在为轻量级分类模型在真实世界场景中的应用与部署提供一个标准化的、可复现的参考范例。

## **2\. 快速部署 (Quick Deployment)**

### **🐳 推荐：Docker 容器化部署**

Docker 是运行本项目最简单、最推荐的方式，它确保了环境的一致性，避免了复杂的依赖安装。

1. **构建镜像:** 在项目根目录下执行：  
   docker build \-t waste-classification .

2. **运行容器:** 将容器内部的 7860 端口映射到本地：  
   docker run \-p 7860:7860 waste-classification

3. **访问应用:** 打开浏览器，访问 [http://localhost:7860](http://localhost:7860) 即可开始使用摄像头实时识别功能。

### **🧩 本地环境运行**

确保你的 Python 版本为 3.10 或更高。

1. **安装依赖:**  
   pip install \-r requirements.txt

2. **运行应用:⚠️ 注意:** 为了确保模型和静态文件能被正确加载，请在项目根目录运行 app.py。  
   \# (推荐) 先 cd 到项目根目录  
   cd /path/to/your/project/

   \# 再运行 app.py  
   python app.py

3. 访问 Gradio 界面 [http://127.0.0.1:7860](http://127.0.0.1:7860)。

## **3\. 系统工作流 (System Workflow)**

本项目采用前后端分离的逻辑，以下是实时识别系统的主要数据流向：

graph TD  
    subgraph Frontend \[前端 Gradio 应用\]  
        A\[用户 (User)\] \--\> B(1. Gradio 界面);  
        B \--\> C(2. 捕获摄像头帧);  
        C \--\> D\[2. 单帧图像 (NumPy)\];  
    end

    subgraph Backend \[后端 Python 服务器 (app.py)\]  
        D \--\> E(3. 接收单帧图像);  
        E \--\> F\[5. 预处理模块\];  
        F \--\> G\[6. MobileNetV2 模型推理\];  
        G \--\> H\[7. 后处理模块 (解析类别/置信度)\];  
        H \--\> I\[8. 结果 (类别 & 置信度)\];  
    end

    style Frontend fill:\#c8e6c9,stroke:\#388e3c,stroke-width:2px;  
    style Backend fill:\#bbdefb,stroke:\#1976d2,stroke-width:2px;  
    style G fill:\#ffcc80,stroke:\#f57c00,stroke-width:2px;

    I \--\> B;

    %% 详细描述连接  
    C \-. 实时视频流 .-\> B  
    E \-. 模型加载 .-\> G  
    G \-. 推理结果 .-\> H  
    H \-. 返回前端 .-\> I

### **流程说明**

1. **前端捕获 (1 & 2):** Gradio 界面持续从用户的摄像头捕获视频流，并将其分解为 NumPy 数组格式的单帧图像 (2. 摄像头帧)。  
2. **数据传输 (3):** 单帧图像通过 Gradio 的 API 传输给后端的 Python 函数进行处理。  
3. **预处理 (5):** 图像在进入模型前被标准化处理（调整尺寸到 224x224，并进行归一化）。  
4. **模型推理 (6):** **MobileNetV2** 模型加载 .keras 文件，并对预处理后的图像进行预测，输出概率向量。  
5. **后处理与反馈 (7 & 8):** 后处理模块将概率向量解析为最高概率的类别和置信度，并将最终结果回传给 Gradio 界面，实时显示给用户。

## **4\. 数据集构建 (Dataset)**

高质量的数据集是模型性能的基石。本项目通过定制化的数据工程流程，构建了一个兼具多样性与高质量的统一训练源。

* **数据来源：**  
  * **TrashNet:** 图像背景统一，标签规范。  
  * **TACO:** 真实环境，背景复杂，提供 COCO JSON 标注。  

> 💾 完整数据集未随仓库上传，可从 [TrashNet](https://github.com/garythung/trashnet) 与 [TACO](https://tacodataset.org) 下载。

* **数据处理流程：**  
  1. 定义统一分类标准（4类）：**塑料制品, 纸制品, 金属制品, 玻璃制品**。  
  2. 编写自动化脚本，解析 TACO 的 JSON 标注文件。  
  3. 遍历标注对象，利用其**边界框坐标**，从原始图像中精确地**裁剪**出单个垃圾物体样本，有效去除复杂背景干扰。  
  4. 与 TrashNet 数据集合并，并手动筛除部分低质量图像。  
  5. 应用在线数据增强（随机水平翻转、小范围旋转、随机缩放、亮度和对比度调整）。  
* **最终数据集规模 (70/15/15 分层抽样)：**

| 数据集 | 样本数 |
| :---- | :---- |
| 训练集 | 9291 |
| 验证集 | 1991 |
| 测试集 | 1992 |

## **5\. 模型方法 (Methodology)**

### **5.1 基础模型选择 (MobileNetV2)**

考虑到项目最终需要实现实时摄像头识别，对模型的推理速度、计算效率以及轻量化有较高要求，我们选择了 **MobileNetV2** 作为基础模型。

* **优势：** MobileNetV2 采用了**深度可分离卷积**（显著减少模型参数）和**倒残差网络结构**（先升维提取特征再降维，减少计算量），使其在 CPU 和资源受限的情况下也能完成快速推理，具有极好的响应速度。

### **5.2 两阶段训练策略 (Two-Stage Training)**

为最大限度地发挥 MobileNetV2 的性能并保证训练稳定，我们设计并实施了严谨的两阶段训练策略：

| 阶段 | 目标 | 策略 | 学习率 (LR) |
| :---- | :---- | :---- | :---- |
| **I. 特征提取** | 让新分类头快速学习高级特征 | **冻结** MobileNetV2 主干网络，仅训练顶层分类头 (GAP, Dropout, Dense)。 | 0.001 (Adam) |
| **II. 模型微调** | 适配垃圾数据集细节，突破性能瓶颈 | 加载第一阶段最佳模型，**解冻** MobileNetV2 顶层卷积层 (从第100层开始)。 | 1e-5 (极低, Adam) |

此策略首先利用预训练模型的泛化能力快速达到一个较高的基准（验证集 89%），然后再通过微调（Fine-Tuning）在不破坏预训练知识的前提下，让模型更适应我们的特定任务，最终将测试集准确率提升至 **90%**。

## **6\. 实验与结果 (Experiments & Results)**

### **实验环境**

* **Python:** 3.10  
* **框架:** TensorFlow 2.16.1 (CPU 版本)  
* **主要依赖:** Gradio, NumPy, OpenCV-Python, Docker

### **训练超参数**

| 参数 | 值 |
| :---- | :---- |
| Batch Size | 32 |
| 损失函数 | Categorical Crossentropy (分类交叉熵) |
| 图像输入尺寸 | 224 x 224 x 3 |
| **阶段一 (特征提取)** |  |
| 优化器 | Adam |
| 学习率 | 0.001 |
| **阶段二 (微调)** |  |
| 解冻策略 | 解冻第 100 层及之后的所有层 |
| 优化器 | Adam |
| 学习率 | 1e-5 (0.00001) |

### **最终性能**

微调后的模型在测试集上的准确率达到了 **90%**，相较于第一阶段（特征提取）有明显提升，验证了微调策略的有效性。

在实时识别测试中，系统能够通过摄像头流畅获取视频流，并实时显示识别结果（类别及置信度），在大多数情况下识别准确。


## **7\. 未来展望 (Future Work)**

* **模型优化：** 进一步优化模型结构和训练策略（例如引入注意力机制），提高模型在复杂场景下（如光线不足、遮挡）的鲁棒性。  
* **数据扩展：** 收集更丰富、更多样化的垃圾图像数据，增加模型对各种实际场景的适应能力。  
* **边缘部署：** 探索将模型转换为 TFLite 格式，并在 Jetson Nano / 树莓派等边缘计算设备上部署，实现更低成本的智能垃圾桶方案。

## **8\. 作者与致谢 (Authors & Acknowledgments)**

💼 GitHub: [@pkangle](https://github.com/pkangle)

### **致谢**

感谢以下开源数据集和工具为本项目提供的支持：

* [TrashNet Dataset](https://github.com/garythung/trashnet)  
* [TACO Dataset](https://tacodataset.org)  
* [Gradio](https://www.gradio.app/) & [TensorFlow](https://www.tensorflow.org/)

⭐ 如果本项目对你的研究或学习有帮助，请在 GitHub 上点亮一颗 Star！