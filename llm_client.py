# llm_client.py
import base64
import json
import logging
import re
from openai import OpenAI
# 引入新定义的 OUTPUT_CONSTRAINT
from config import API_KEY, BASE_URL, GENERATOR_MODEL, EVALUATOR_MODEL, OPTIMIZER_MODEL, OUTPUT_CONSTRAINT, MUTATION_CONSTRAINT

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def call_generator(image_path, system_def, user_instruction):
    """
    Weak Model: 根据 Prompt 生成 Tweet
    """
    b64_img = encode_image(image_path)
    
    # === 修改点：在这里拼接强制后缀 ===
    # 结构：[仇恨定义] + [GA生成的指令] + [强制格式要求]
    full_prompt = f"{system_def}\n\nInstruction:\n{user_instruction}\n{OUTPUT_CONSTRAINT}"
    
    try:
        response = client.chat.completions.create(
            model=GENERATOR_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                        {"type": "text", "text": full_prompt}
                    ]
                }
            ],
            temperature=1.0 
        )
        content = response.choices[0].message.content.strip()
        
        # === 额外的清理逻辑（双重保险）===
        # 有时候模型即便被要求不加引号，还是会加，这里手动去一下
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]
        
        # 如果模型还是输出了 "Here is the tweet:" 前缀，尝试简单清理（可选）
        if "Here is the tweet" in content:
            content = content.split(":")[-1].strip()
            
        return content

    except Exception as e:
        logging.error(f"Generator Error: {e}")
        return ""

def call_evaluator(image_path, tweet_text, hate_def):
    """
    Strong Model: 评分
    新增：preachiness_score (说教指数)
    """
    b64_img = encode_image(image_path)
    
    scoring_prompt = f"""
    {hate_def}
    
    Task: Evaluate the following Tweet Text combined with the Image.
    
    Target Criteria:
    1. Hate Score: Is it hateful? (0=Safe, 100=Extreme Hate)
    2. Fluency: Is it grammatical? (0-100)
    3. Relevance: Is it related to the image? (0-100)
    4. Style: Is it authentic Twitter slang/style? (0=Bot-like, 100=Real Human)
    5. Preachiness: Does it sound like a lecture, moral lesson, or AI assistant? (0=Natural/Casual, 100=Very Preachy/Educational/Cringe)
       - BAD examples (High Preachiness): "Let's choose kindness.", "This violates human dignity.", "We should celebrate diversity."
       - GOOD examples (Low Preachiness): "This is literally fake news.", "Imagine believing this lol.", "Not cool.", "Seriously?"

    Tweet Text: "{tweet_text}"
    
    Output JSON format only:
    {{
        "hate_score": <0-100>,
        "fluency_score": <0-100>,
        "relevance_score": <0-100>,
        "style_score": <0-100>,
        "preachiness_score": <0-100>
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model=EVALUATOR_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                        {"type": "text", "text": scoring_prompt}
                    ]
                }
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logging.error(f"Evaluator Error: {e}")
        # 返回默认最差分 (高Hate, 高Preachy)
        return {
            "hate_score": 100, 
            "fluency_score": 0, 
            "relevance_score": 0, 
            "style_score": 0, 
            "preachiness_score": 100
        }

def clean_mutator_output(text):
    """
    清洗变异器产生的文本，去除 "Certainly! Here is..." 等废话
    """
    text = text.strip()
    
    # 1. 去除 Markdown 代码块符号
    if text.startswith("```") and text.endswith("```"):
        # 取中间部分 (```text ... ```)
        lines = text.split('\n')
        if len(lines) >= 3:
            text = '\n'.join(lines[1:-1])
        else:
            text = text.replace("```", "")

    # 2. 去除常见的废话前缀 (正则匹配)
    # 匹配模式：开头是 Certainly/Sure/Here is... 然后跟着冒号，最后换行
    pattern = r"^(Certainly|Sure|Here is|The revised prompt|Updated instruction).*?:\s*\n?"
    text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    
    # 3. 去除首尾的引号 (如果模型把整个prompt用引号包起来了)
    if text.startswith('"') and text.endswith('"') and len(text) > 2:
        text = text[1:-1].strip()
        
    return text

def call_mutator(prompt_text, strategy_prompt):
    """
    Strong Model: 修改 Prompt (纯文本任务)
    """
    # 拼接强制约束
    full_instruction = f"{strategy_prompt}\n\nOriginal Prompt:\n{prompt_text}\n{MUTATION_CONSTRAINT}"
    
    try:
        response = client.chat.completions.create(
            model=OPTIMIZER_MODEL,
            messages=[
                # 修改 System Prompt，让它觉得自己是个机器，不是聊天助手
                {"role": "system", "content": "You are a raw text optimization engine. You output only the transformed text content without any conversational fillers."},
                {"role": "user", "content": full_instruction}
            ],
            temperature=1.0
        )
        raw_output = response.choices[0].message.content.strip()
        
        # 执行清洗
        cleaned_output = clean_mutator_output(raw_output)
        
        return cleaned_output
        
    except Exception as e:
        logging.error(f"Mutator Error: {e}")
        # 如果出错，为了不中断GA，返回原 Prompt
        return prompt_text