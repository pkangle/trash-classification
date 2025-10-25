import os
import shutil
from sklearn.model_selection import train_test_split
import glob

SOURCE_DATA_DIR = r"C:\Users\Lenovo\Desktop\数据集"
TARGET_SPLIT_DIR = r"C:\Users\Lenovo\Desktop\final_split_dataset"
CATEGORIES = ["塑料制品", "纸制品", "金属制品","玻璃制品"] 
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15 

def create_target_directories(base_dir, sub_dirs, categories):
    for sub_dir in sub_dirs:
        for category in categories:
            os.makedirs(os.path.join(base_dir, sub_dir, category), exist_ok=True)
create_target_directories(TARGET_SPLIT_DIR, ['train', 'validation', 'test'], CATEGORIES)

all_files = []
all_labels = []

for category in CATEGORIES:
    category_path = os.path.join(SOURCE_DATA_DIR, category)
    images_in_category = glob.glob(os.path.join(category_path, '*.jpg')) 
    
    all_files.extend(images_in_category)
    all_labels.extend([category] * len(images_in_category))

# 第一次划分：训练集 vs (验证集+测试集)
train_files, remaining_files, train_labels, remaining_labels = train_test_split(
    all_files, all_labels, test_size=(1 - TRAIN_RATIO), random_state=42, stratify=all_labels
)

# 第二次划分：验证集 vs 测试集 (从 remaining_files 中划分)
val_relative_ratio = VALIDATION_RATIO / (VALIDATION_RATIO + TEST_RATIO)
val_files, test_files, val_labels, test_labels = train_test_split(
    remaining_files, remaining_labels, test_size=(1 - val_relative_ratio), random_state=42, stratify=remaining_labels
)

print(f"训练集样本数: {len(train_files)}")
print(f"验证集样本数: {len(val_files)}")
print(f"测试集样本数: {len(test_files)}")

def copy_files_to_split_dir(files, labels, split_name):
    print(f"开始拷贝 {split_name} 数据...")
    for file_path, label in zip(files, labels):
        target_category_dir = os.path.join(TARGET_SPLIT_DIR, split_name, label)
        if os.path.exists(file_path):
            shutil.copy(file_path, target_category_dir)
        else:
            print(f"警告: 文件未找到 {file_path}")
    print(f"{split_name} 数据拷贝完成。")

#  执行拷贝

copy_files_to_split_dir(train_files, train_labels, "train")
copy_files_to_split_dir(val_files, val_labels, "validation")
copy_files_to_split_dir(test_files, test_labels, "test")

print("数据集划分完成！") 