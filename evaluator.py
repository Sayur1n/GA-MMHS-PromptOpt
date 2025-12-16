# evaluator.py
import random
import logging
from llm_client import call_generator, call_evaluator
from config import HATE_SPEECH_DEF, SAMPLES_PER_EVAL, W_HATE, W_FLUENCY, W_REL, W_STYLE, W_PREACHY

def calculate_fitness(prompt_candidate, dataset):
    """
    在随机抽样的数据集上评估 Prompt 的表现
    返回: (avg_fitness, average_metrics, detailed_results)
    """
    # 随机采样
    test_samples = random.sample(dataset, min(len(dataset), SAMPLES_PER_EVAL))
    
    total_score = 0
    metrics_log = {"hate": 0, "fluency": 0, "relevance": 0, "style": 0, "preachy": 0}
    detailed_results = []
    
    for sample in test_samples:
        img_path = sample['image_path']
        sid = sample.get('sid', 'unknown')
        
        # 1. 生成
        gen_text = call_generator(img_path, HATE_SPEECH_DEF, prompt_candidate)
        
        # 2. 评分
        scores = call_evaluator(img_path, gen_text, HATE_SPEECH_DEF)
        
        # 3. 归一化计算
        # Hate: 越低越好
        s_hate = max(0, 100 - scores.get('hate_score', 100)) / 100.0
        
        # Preachy: 越低越好 (新增)
        s_preachy = max(0, 100 - scores.get('preachiness_score', 100)) / 100.0
        
        # 其他: 越高越好
        s_fluency = scores.get('fluency_score', 0) / 100.0
        s_rel = scores.get('relevance_score', 0) / 100.0
        s_style = scores.get('style_score', 0) / 100.0
        
        # 加权求和
        fitness = (s_hate * W_HATE) + \
                  (s_preachy * W_PREACHY) + \
                  (s_fluency * W_FLUENCY) + \
                  (s_rel * W_REL) + \
                  (s_style * W_STYLE)
        
        total_score += fitness
        
        # 累加指标
        metrics_log["hate"] += scores.get('hate_score', 0)
        metrics_log["fluency"] += scores.get('fluency_score', 0)
        metrics_log["relevance"] += scores.get('relevance_score', 0)
        metrics_log["style"] += scores.get('style_score', 0)
        metrics_log["preachy"] += scores.get('preachiness_score', 0)
        
        # 记录详情
        detailed_results.append({
            "sid": sid,
            "image_path": img_path,
            "generated_text": gen_text,
            "raw_scores": scores,
            "fitness": fitness
        })

    # 计算平均值
    avg_fitness = total_score / len(test_samples)
    for k in metrics_log:
        metrics_log[k] /= len(test_samples)
        
    return avg_fitness, metrics_log, detailed_results