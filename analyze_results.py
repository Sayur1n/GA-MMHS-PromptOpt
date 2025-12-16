import json
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

def find_latest_history():
    list_of_files = glob.glob('ga_history_*.json')
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

def analyze():
    history_file = find_latest_history()
    if not history_file:
        print("No history file found.")
        return

    print(f"Analyzing: {history_file}")
    with open(history_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取数据
    gens = []
    max_fitness = []
    avg_fitness = []
    
    # 细分指标 (Average per generation)
    avg_hate = []
    avg_preachy = []
    avg_style = []
    avg_fluency = []
    avg_relevance = []
    stats_table = []

    for gen_data in data:
        g_num = gen_data['generation']
        individuals = gen_data['individuals']
        
        # 提取当前代所有个体的分数
        fits = [ind['fitness'] for ind in individuals]
        
        # 提取当前代所有个体的 metrics
        # 注意：这里我们取当前代所有 Prompt 的 metrics 平均值
        hates = [ind['average_metrics']['hate'] for ind in individuals]
        preachies = [ind['average_metrics'].get('preachy', 0) for ind in individuals] # 兼容旧版
        styles = [ind['average_metrics']['style'] for ind in individuals]
        fluencies = [ind['average_metrics']['fluency'] for ind in individuals]
        relevances = [ind['average_metrics']['relevance'] for ind in individuals]
        
        m_fit = max(fits)
        a_fit = sum(fits) / len(fits)
        
        gens.append(g_num)
        max_fitness.append(m_fit)
        avg_fitness.append(a_fit)
        
        avg_hate.append(sum(hates) / len(hates))
        avg_preachy.append(sum(preachies) / len(preachies))
        avg_style.append(sum(styles) / len(styles))
        avg_fluency.append(sum(fluencies) / len(fluencies))
        avg_relevance.append(sum(relevances) / len(relevances))
        
        # 添加到表格数据
        stats_table.append({
            "Generation": g_num,
            "Best Fitness": f"{m_fit:.4f}",
            "Avg Fitness": f"{a_fit:.4f}",
            "Avg Hate Score": f"{avg_hate[-1]:.2f}",
            "Avg Preachy": f"{avg_preachy[-1]:.2f}",
            "Avg Style": f"{avg_style[-1]:.2f}",
            "Avg Fluency": f"{avg_fluency[-1]:.2f}",
            "Avg Relevance": f"{avg_relevance[-1]:.2f}",
        })

    # ================= 1. 绘制 Fitness 收敛图 =================
    plt.figure(figsize=(10, 6))
    plt.plot(gens, max_fitness, 'r-o', label='Max Fitness (Best Prompt)')
    plt.plot(gens, avg_fitness, 'b--s', label='Average Fitness (Population)')
    plt.title('Optimization Trajectory (Genetic Algorithm)')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('convergence_plot.png')
    print("Saved convergence_plot.png")

    # ================= 2. 绘制 关键指标变化图 =================
    plt.figure(figsize=(10, 6))

    plt.plot(gens, avg_hate, 'r-o', label='Avg Hate (Lower is Better)')
    plt.plot(gens, avg_preachy, 'g-^', label='Avg Preachy (Lower is Better)')
    plt.plot(gens, avg_style, 'b--s', label='Avg Style (Higher is Better)')
    plt.plot(gens, avg_fluency, 'c--d', label='Avg Fluency (Higher is Better)')
    plt.plot(gens, avg_relevance, 'm--x', label='Avg Relevance (Higher is Better)')

    plt.title('Sub-metrics Evolution over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Score (0–10)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('metrics_plot.png')

    print("Saved metrics_plot.png")


    # ================= 3. 生成表格 =================
    df = pd.DataFrame(stats_table)
    print("\n=== Experiment Statistics Table ===")
    print(df.to_string(index=False))
    
    # 保存为 CSV
    df.to_csv("experiment_stats.csv", index=False)
    print("\nSaved experiment_stats.csv")

if __name__ == "__main__":
    analyze()