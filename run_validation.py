import os
import json
import glob
import textwrap
import random
from PIL import Image, ImageDraw, ImageFont
from config import DATA_FILE, HATE_SPEECH_DEF, OUTPUT_CONSTRAINT
from llm_client import call_generator

# ================= 配置 =================
IMAGE_DIR = "images"  # 图片文件夹
FONT_PATH = "arial.ttf"  # 字体路径 (Windows/Linux通常有，如果没有请换成其他ttf)
FONT_SIZE = 20

OUTPUT_JSON_BEST = "final_results_best.json"
OUTPUT_DIR_BEST = "pair_results"

OUTPUT_JSON_INIT = "final_results_initial.json"
OUTPUT_DIR_INIT = "initial_pair_results"

def find_latest_history():
    list_of_files = glob.glob('ga_history_*.json') 
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def add_text_to_image(image_path, text, output_path):
    """
    将文本拼接到图片下方，类似长图模式，方便放进论文
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return

    # 准备画布宽度和基础字体
    width, height = img.size
    
    # 尝试加载字体，失败则使用默认
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except:
        font = ImageFont.load_default()

    # 自动换行
    # 估算每行字符数 (假设平均字符宽度为字体大小的0.5倍)
    chars_per_line = int(width / (FONT_SIZE * 0.5))
    lines = textwrap.wrap(text, width=chars_per_line)
    
    # 计算文本区域高度
    # 简单估算：行数 * (字体大小 + 行间距) + Padding
    line_height = int(FONT_SIZE * 1.5)
    text_area_height = len(lines) * line_height + 40
    
    # 创建新画布 (白色背景)
    new_img = Image.new('RGB', (width, height + text_area_height), (255, 255, 255))
    new_img.paste(img, (0, 0))
    
    draw = ImageDraw.Draw(new_img)
    
    # 逐行绘制
    y_text = height + 20
    for line in lines:
        # 居中绘制 (可选，这里用左对齐+一点padding)
        draw.text((10, y_text), line, font=font, fill=(0, 0, 0))
        y_text += line_height
        
    new_img.save(output_path)

def run_generation_batch(prompts, image_files, output_json_name, output_img_dir):
    """
    使用一组 Prompts 对一组图片进行生成
    """
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
        
    results = []
    
    print(f"\n>>> Processing batch for {output_json_name}...")
    
    for p_idx, prompt in enumerate(prompts):
        print(f"  Using Prompt {p_idx+1}/{len(prompts)}")
        
        for img_file in image_files:
            img_path = os.path.join(IMAGE_DIR, img_file)
            sid = os.path.splitext(img_file)[0]
            
            # 生成文本
            # 注意：这里需要加上 Output Constraint，保持和训练时一致
            full_instruction = prompt + "\n" + OUTPUT_CONSTRAINT
            
            tweet_text = call_generator(img_path, HATE_SPEECH_DEF, full_instruction)
            
            # 记录结果
            record = {
                "sid": sid,
                "prompt_id": p_idx,
                "prompt_content": prompt,
                "generated_text": tweet_text
            }
            results.append(record)
            
            # 生成图片
            out_img_name = f"{sid}_p{p_idx}.jpg"
            out_img_path = os.path.join(output_img_dir, out_img_name)
            add_text_to_image(img_path, tweet_text, out_img_path)
    
    # 保存 JSON
    with open(output_json_name, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved results to {output_json_name} and images to {output_img_dir}/")

def main():
    # 1. 读取最新的历史记录
    history_file = find_latest_history()
    if not history_file:
        print("No ga_history file found!")
        return
    
    print(f"Loading history from: {history_file}")
    with open(history_file, "r", encoding="utf-8") as f:
        history = json.load(f)
    
    # 2. 获取图片列表
    if not os.path.exists(IMAGE_DIR):
        print("Image directory not found.")
        return
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(image_files)} images.")

    # 3. 提取 Prompts
    # --- 3.1 最优 Prompts (从最后一代选前3) ---
    last_gen = history[-1]
    # 确保已按 fitness 排序
    sorted_individuals = sorted(last_gen['individuals'], key=lambda x: x['fitness'], reverse=True)
    best_prompts_data = sorted_individuals[:3]
    best_prompts = [item['prompt_text'] for item in best_prompts_data]
    print(f"Extracted {len(best_prompts)} best prompts from Generation {last_gen['generation']}.")

    # --- 3.2 初始 Prompts (从第1代随机选3个) ---
    first_gen = history[0]
    initial_individuals = first_gen['individuals']
    # 随机抽3个，如果不够就全拿
    random_initials = random.sample(initial_individuals, min(3, len(initial_individuals)))
    initial_prompts = [item['prompt_text'] for item in random_initials]
    print(f"Extracted {len(initial_prompts)} random initial prompts from Generation 1.")

    # 4. 执行生成任务
    # Task A: Best Prompts
    run_generation_batch(best_prompts, image_files, OUTPUT_JSON_BEST, OUTPUT_DIR_BEST)
    
    # Task B: Initial Prompts
    run_generation_batch(initial_prompts, image_files, OUTPUT_JSON_INIT, OUTPUT_DIR_INIT)

if __name__ == "__main__":
    main()