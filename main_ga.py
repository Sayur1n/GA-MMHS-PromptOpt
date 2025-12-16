# main_ga.py
import json
import logging
import os
import random
import time
from config import (
    DATA_FILE, INITIAL_SEED_PROMPT, POPULATION_SIZE, 
    GENERATIONS, ELITISM_COUNT
)
from evolution import init_population_expansion, get_next_variant, crossover_prompts
from evaluator import calculate_fitness

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()
file_handler = logging.FileHandler("ga_training.log", mode="w", encoding="utf-8")
logger.addHandler(file_handler)

# 早停参数
PATIENCE_LIMIT = 3
MIN_DELTA = 0.005
TARGET_SCORE = 0.98

# 结果保存文件
HISTORY_FILE = f"ga_history_{int(time.time())}.json"

def load_data():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return [d for d in data if d.get('label') == 1]

def save_history(history_data):
    """将整个历史记录保存到 JSON"""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2, ensure_ascii=False)
    logger.info(f"  [SAVED] Full history saved to {HISTORY_FILE}")

def run_genetic_algorithm():
    dataset = load_data()
    if not dataset:
        logger.error("Data is empty!")
        return

    # 1. 初始化
    population = init_population_expansion(INITIAL_SEED_PROMPT, POPULATION_SIZE)
    
    global_best_prompt = ""
    global_best_score = -1.0
    patience_counter = 0 
    
    # 核心历史记录数据结构
    ga_history = [] 
    
    # 2. 迭代循环
    for gen in range(GENERATIONS):
        logger.info(f"\n{'='*20} Generation {gen + 1} / {GENERATIONS} {'='*20}")
        
        # 当前代的数据记录
        current_gen_data = {
            "generation": gen + 1,
            "individuals": [] # 存放每个 prompt 的详情
        }
        
        scored_population = []
        
        # --- 评估 ---
        for i, prompt in enumerate(population):
            # 注意：这里接收了第三个返回值 details
            fitness, metrics, details = calculate_fitness(prompt, dataset)
            
            scored_population.append((prompt, fitness, metrics))
            
            # 记录该 Prompt 的详细信息
            current_gen_data["individuals"].append({
                "prompt_id": f"gen_{gen+1}_id_{i}",
                "prompt_text": prompt,
                "fitness": fitness,
                "average_metrics": metrics,
                "sample_evaluations": details # 这里包含了具体的生成文本和得分
            })
            
            logger.info(f"  [P{i}] Score: {fitness:.4f} | Hate: {metrics['hate']:.2f}")

        # --- 排序 ---
        scored_population.sort(key=lambda x: x[1], reverse=True)
        current_gen_data["individuals"].sort(key=lambda x: x["fitness"], reverse=True) # JSON里也排个序
        
        current_best = scored_population[0]
        
        # 更新代最佳信息
        current_gen_data["best_score"] = current_best[1]
        current_gen_data["best_prompt"] = current_best[0]
        
        # 添加到总历史并保存
        ga_history.append(current_gen_data)
        save_history(ga_history)
        
        # --- 早停检查逻辑 ---
        score_improvement = current_best[1] - global_best_score
        if score_improvement > MIN_DELTA:
            logger.info(f"  >>> New Global Best Found! (+{score_improvement:.4f})")
            global_best_score = current_best[1]
            global_best_prompt = current_best[0]
            patience_counter = 0 
        else:
            patience_counter += 1
        
        if global_best_score >= TARGET_SCORE or patience_counter >= PATIENCE_LIMIT:
            logger.info("  !!! Stopping Early !!!")
            break
        
        if gen == GENERATIONS - 1:
            break

        # --- 繁殖下一代 ---
        new_population = []
        
        # A. 精英保留
        for i in range(ELITISM_COUNT):
            new_population.append(scored_population[i][0])
            # 在历史记录中标记一下谁是精英（可选，但通常通过文本对比能看出来）
        
        # B. 变异与交叉
        while len(new_population) < POPULATION_SIZE:
            candidates = random.sample(scored_population, 2)
            parent = max(candidates, key=lambda x: x[1])[0]
            
            if random.random() < 0.8:
                child = get_next_variant(parent)
                # 可以在这里记录 parent -> child 的关系，但 GA 标准通常只看每一代的表现
            else:
                candidates_2 = random.sample(scored_population, 2)
                parent_2 = max(candidates_2, key=lambda x: x[1])[0]
                child = crossover_prompts(parent, parent_2)
            
            if child not in new_population:
                new_population.append(child)
        
        population = new_population

    logger.info("Optimization Done. Check ga_history json file.")

if __name__ == "__main__":
    run_genetic_algorithm()