import os
import json
from pathlib import Path

# ================= 配置 =================
IMAGE_DIR = "images"           # 图片文件夹路径
OUTPUT_FILE = "train_images.json" # 输出的 JSON 文件名
DEFAULT_LABEL = 1              # 默认标签 (1 代表有害图片)

# 支持的图片扩展名
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}

def generate_json():
    # 检查图片目录是否存在
    if not os.path.exists(IMAGE_DIR):
        print(f"[Error] Directory '{IMAGE_DIR}' not found. Please create it and put images inside.")
        return

    dataset = []
    
    # 遍历目录
    print(f"Scanning directory: {IMAGE_DIR} ...")
    files = os.listdir(IMAGE_DIR)
    
    count = 0
    for filename in files:
        # 获取文件后缀
        file_path = Path(os.path.join(IMAGE_DIR, filename))
        suffix = file_path.suffix.lower()
        
        # 检查是否为图片
        if suffix in VALID_EXTENSIONS:
            # 提取 sid (文件名去掉后缀)
            sid = file_path.stem 
            
            # 构建记录
            record = {
                "sid": sid,
                # 注意：这里存的是相对路径，确保主程序能找到
                "image_path": str(file_path).replace("\\", "/"), 
                "label": DEFAULT_LABEL
            }
            
            dataset.append(record)
            count += 1
    
    # 排序（可选，方便查看）
    dataset.sort(key=lambda x: x['sid'])

    # 写入 JSON 文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        
    print(f"\n[Success] Processed {count} images.")
    print(f"[Saved] Dataset saved to: {os.path.abspath(OUTPUT_FILE)}")
    
    # 打印前3条数据示例
    if dataset:
        print("\n--- Example Data ---")
        print(json.dumps(dataset[:3], indent=2))

if __name__ == "__main__":
    generate_json()