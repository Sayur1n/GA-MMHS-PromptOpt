# evolution.py
import random
import re
from llm_client import call_mutator

# ================= 基础配置 =================

# 1. 局部微调策略 (Exploitation) - 保持原有逻辑
SPAN_STRATEGIES = [
    "rewrite",          # 重写语义
    "strengthen",       # 增强语气
    "simplify",         # 简化表达
    "add_constraint",   # 增加限制条件
]

# 2. 发散性策略 (Exploration) - 新增
# 这些策略旨在彻底改变 Prompt 的结构或切入点
CONCEPT_STRATEGIES = [
    "shift_focus_to_humor",       # 策略：幽默/讽刺
    "shift_focus_to_facts",       # 策略：事实核查/冷漠陈述
    "shift_focus_to_visuals",     # 策略：只关注视觉美学，忽略含义
    "adopt_persona_genz",         # 角色：Gen-Z 网民 (口语化)
    "adopt_persona_minimalist",   # 角色：极简主义者
    "add_step_by_step",           # 结构：思维链
    "remove_safety_warnings",     # 对抗：移除过多的安全警告，减少说教
]

def split_into_sentences(text):
    """简单的分句逻辑"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

# ================= 变异函数 =================

def mutate_span_level(current_prompt):
    """
    [微调] 局部变异：随机修改一个句子，保持大体结构不变。
    """
    sentences = split_into_sentences(current_prompt)
    if len(sentences) <= 1:
        return mutate_global(current_prompt)
    
    idx = random.randint(0, len(sentences) - 1)
    target_span = sentences[idx]
    strategy = random.choice(SPAN_STRATEGIES)
    
    instruction = ""
    if strategy == "rewrite":
        instruction = f"Rewrite this sentence to be more impactful but keep the meaning: '{target_span}'"
    elif strategy == "strengthen":
        instruction = f"Make this sentence sound more authoritative and strict: '{target_span}'"
    elif strategy == "simplify":
        instruction = f"Shorten this sentence drastically to be punchy: '{target_span}'"
    elif strategy == "add_constraint":
        instruction = f"Add a constraint to this sentence about being concise or avoiding preachiness: '{target_span}'"
    
    new_span = call_mutator(target_span, instruction)
    sentences[idx] = new_span
    return " ".join(sentences)

def mutate_concept_shift(current_prompt):
    """
    [发散] 概念变异：改变 Prompt 的核心策略或视角。
    这是跳出局部最优的关键。
    """
    strategy = random.choice(CONCEPT_STRATEGIES)
    
    instruction = ""
    if strategy == "shift_focus_to_humor":
        instruction = "Rewrite the entire prompt to instruct the model to be witty, sarcastic, or humorous instead of serious. The goal is to mock the hate speech subtly."
    elif strategy == "shift_focus_to_facts":
        instruction = "Rewrite the prompt to instruct the model to act like a cold, objective fact-checker. It should correct the image's premise with dry logic."
    elif strategy == "shift_focus_to_visuals":
        instruction = "Rewrite the prompt to instruct the model to focus purely on the visual composition or aesthetic flaws of the image, ignoring the hateful message entirely."
    elif strategy == "adopt_persona_genz":
        instruction = "Rewrite the instructions to force the model to adopt a casual, 'Gen-Z' internet user persona. Use slang, lowercase, and be dismissive of the hate."
    elif strategy == "adopt_persona_minimalist":
        instruction = "Rewrite the prompt to demand extreme brevity. The model should output less than 10 words."
    elif strategy == "add_step_by_step":
        instruction = "Insert a 'Step-by-Step' reasoning requirement into the prompt, asking the model to first analyze the intent, then choose a counter-strategy."
    elif strategy == "remove_safety_warnings":
        instruction = "Remove any parts of the prompt that mention 'safety guidelines', 'respect', or 'harm'. Make the prompt purely functional to avoid triggering the model's preachy safety filters."

    # 调用 Mutator 进行全篇改写
    return call_mutator(current_prompt, instruction)

def mutate_global(current_prompt):
    """[补充] 通用全局变异"""
    strategy = random.choice(["rephrase", "expand", "condense"])
    instruction = f"Please {strategy} the following prompt instruction to be more effective for an AI model."
    return call_mutator(current_prompt, instruction)

def crossover_prompts(prompt_a, prompt_b):
    """交叉：融合两个 Prompt"""
    instruction = "Analyze Prompt A and Prompt B. Create a new, hybrid prompt that combines the unique strategies of both (e.g., the persona of A and the constraints of B)."
    combined_input = f"Prompt A: {prompt_a}\n\nPrompt B: {prompt_b}"
    return call_mutator(combined_input, instruction)

# ================= 核心调度逻辑 =================

def get_next_variant(prompt, current_generation=0):
    """
    统一接口：根据概率选择变异策略。
    可以引入 '模拟退火' 思想：前期多发散，后期多微调。
    """
    rand_val = random.random()
    
    # 动态概率调整（可选）：随着代数增加，减少大幅度变异
    # mutation_rate_concept = max(0.2, 0.5 - current_generation * 0.05) 
    
    # 固定概率配置
    # 40% 概率进行概念大转移 (发散)
    # 40% 概率进行局部 Span 微调 (收敛)
    # 20% 概率进行常规重写
    
    if rand_val < 0.4:
        # 发散：尝试全新的路子
        return mutate_concept_shift(prompt)
    elif rand_val < 0.8:
        # 收敛：修修补补
        return mutate_span_level(prompt)
    else:
        # 保底：常规改写
        return mutate_global(prompt)

def init_population_expansion(seed_prompt, size):
    population = [seed_prompt]
    print(f"Generating initial population ({size})...")
    
    attempts = 0
    while len(population) < size and attempts < size * 3:
        # 初始化阶段，我们需要极大的多样性
        # 所以强制交替使用 Global 和 Concept Shift
        if attempts % 2 == 0:
            new_p = mutate_concept_shift(seed_prompt)
        else:
            new_p = mutate_global(seed_prompt)
            
        # 简单的去重查重（基于长度或内容）
        if new_p not in population and len(new_p) > 20:
            population.append(new_p)
            print(f"  -> Generated variant {len(population)}/{size}")
        
        attempts += 1
        
    return population