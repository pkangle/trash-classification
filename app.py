import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as model_specific_preprocess_input
import numpy as np
import os

# -----------------------------------------------------------------------------
# 1. 配置参数 (请根据你的项目情况进行修改)
# -----------------------------------------------------------------------------
# --- 模型和标签配置 ---
# 指向你微调后表现最佳的模型文件
MODEL_PATH = 'saved_models/best_model_finetuned.keras'

# 指向你的训练数据目录，用于自动获取类别名称和顺序
TRAIN_DIR =  r"C:\Users\Lenovo\Desktop\final_split_dataset\train"
IMG_HEIGHT = 224
IMG_WIDTH = 224

# -----------------------------------------------------------------------------
# 2. 全局初始化：加载模型和类别信息
# -----------------------------------------------------------------------------
# 将模型和类别信息加载到全局变量中，这样应用启动时只需加载一次
try:
    print("[INFO] 正在加载训练好的模型...")
    model = load_model(MODEL_PATH)
    print("[INFO] 模型加载成功！")
    
    # 从训练目录中获取类别名称列表，并按字母顺序排序以保证一致性
    try:
        with open('labels.txt', 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f.readlines()]
        if not class_names:
            raise ValueError("labels.txt 文件为空。")
        print(f"[INFO] 从 labels.txt 加载了 {len(class_names)} 个类别: {class_names}")
    except Exception as e:
        print(f"[ERROR] 加载类别名称失败: {e}")

except Exception as e:
    print(f"[ERROR] 初始化失败: {e}")
    print("[ERROR] 请确保模型路径和训练目录路径正确。")
    model = None
    class_names = []

# -----------------------------------------------------------------------------
# 3. 定义核心的预测函数
# -----------------------------------------------------------------------------
def classify_waste(image_from_webcam):
    """
    接收来自Gradio摄像头组件的NumPy图像，并返回一个包含类别和置信度的字典。
    """
    if model is None or not class_names:
        return {"错误": 1.0, "模型未正确加载": 1.0}

    # Gradio的摄像头输入直接是 (height, width, 3) 的RGB NumPy数组
    image_tensor = tf.convert_to_tensor(image_from_webcam, dtype=tf.uint8)
    resized_image = tf.image.resize(image_tensor, [IMG_HEIGHT, IMG_WIDTH])

    #  增加一个批次维度，因为模型期望的输入是 (1, height, width, channels)
    input_batch = tf.expand_dims(resized_image, 0)
    preprocessed_tensor = model_specific_preprocess_input(input_batch)

    predictions = model.predict(preprocessed_tensor, verbose=0)
    confidences = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
    
    return confidences


with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css="footer {display: none !important}") as demo:
    gr.Markdown(
        """
        # 实时垃圾分类识别系统 ♻️
        **项目成品展示**
        
        请将摄像头清晰地对准**单个**垃圾物品，系统将实时进行分类预测。
        """
    )
    
    with gr.Row(variant="panel"):
        # 输入组件：摄像头
        webcam_input = gr.Image(
            sources="webcam",      # 数据来源是摄像头
            streaming=True,        # 开启实时视频流模式
            label="摄像头实时画面",
            type="numpy"           # 将图像数据作为NumPy数组传递给函数
        )
        
        # 输出组件：标签
        label_output = gr.Label(
            label="识别结果",         
            num_top_classes=4     # 显示前4个类别的置信度
        )

    # 设置组件之间的交互逻辑
    # 当 webcam_input 的视频流 (stream) 更新时，自动调用 classify_waste 函数，
    # 并将函数的返回值更新到 label_output 组件上。
    webcam_input.stream(
        fn=classify_waste,        # 要调用的函数
        inputs=webcam_input,      # 函数的输入源
        outputs=label_output,     # 函数的输出目标
        #every=0.5                 # 每0.5秒处理一次画面，以平衡实时性和性能
    )
    
    gr.Markdown("--- \n *关闭浏览器标签页退出。*")

if __name__ == "__main__":
    print("[INFO] 正在启动 Gradio 应用...")
    demo.launch(share=True)
