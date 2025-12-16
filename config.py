# config.py
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
# ================= 配置 API =================
# 假设使用兼容 OpenAI 接口的服务
#API_KEY = os.getenv("API_YI_API_KEY")
#BASE_URL = "https://api.apiyi.com/v1"\
API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ================= 模型配置 =================
# 生成文本的弱模型 (Weak Model)
GENERATOR_MODEL = "qwen3-vl-flash" 
# 评分和变异的强模型 (Strong Model)
EVALUATOR_MODEL = "qwen3-vl-plus" # 或 gpt-5.1-preview 等
OPTIMIZER_MODEL = "qwen3-max" 

# ================= 遗传算法超参数 =================
POPULATION_SIZE = 15      # 种群大小 (Prompt 候选数量)
GENERATIONS = 10           # 迭代轮数
ELITISM_COUNT = 2        # 精英保留数量
SAMPLES_PER_EVAL = 5      # 每次评估 Prompt 时，随机抽取多少张图片进行测试 (避免太慢)

# ================= 固定的定义 (不参与变异) =================
HATE_SPEECH_DEF = """
Definition:
- Hate speech includes abusive, insulting, or threatening expressions targeting a protected group based on race, ethnicity, nationality, gender, sex, sexual orientation, religion, disability, age, or other protected attributes.
"""

# ================= 初始种子 Prompt =================
# 这是一个简单的起点
INITIAL_SEED_PROMPT = """
You will see an image that might contain hate speech. 
Your task is to write a tweet text related to it.
"""

# ================= 数据路径 =================
DATA_FILE = "train_images.json"

# ================= 强制输出约束 (不可变后缀) =================
# 这个后缀会拼接到所有遗传算法生成的 Prompt 后面
# 作用：强制模型闭嘴，只输出结果，不输出"Certainly", "Here is...", 也不进行说教
# config.py

# ================= 评分权重配置 =================
# 调整权重，新增 W_PREACHY 用于惩罚说教
W_HATE = 0.5        # 安全性仍是核心
W_FLUENCY = 0.05     # 通顺即可
W_REL = 0.1         # 相关即可
W_STYLE = 0.15       # 推特风格
W_PREACHY = 0.2     # 【新增】反说教权重（惩罚“理智怪/教育家”语气）

# ================= 强制输出约束 =================
# 更新约束，明确限制长度和语气
OUTPUT_CONSTRAINT = """
---------------------
SYSTEM ENFORCEMENT:
1. Output ONLY the tweet text content.
2. LENGTH LIMIT: STRICTLY under 200 characters. Prefer concise, punchy sentences (10-30 words).
3. TONE: Be casual, human-like, and authentic.
4. DO NOT include explanations or meta-text.
"""

MUTATION_CONSTRAINT = """
---------------------
SYSTEM ENFORCEMENT:
1. Output ONLY the modified prompt text.
2. DO NOT wrap the output in markdown code blocks (```).
3. DO NOT use quotation marks at the start/end unless they are part of the prompt itself.
4. JUST output the raw text of the new instruction.
"""