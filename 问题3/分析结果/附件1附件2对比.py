import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 各附件厚度结果
results = {
    "附件1": {"条纹法": 3.25, "拟合法": 3.18},
    "附件2": {"条纹法": 3.28, "拟合法": 3.22},
    "附件3": {"拟合法": 3.30},
    "附件4": {"拟合法": 3.32}
}

# 转为 DataFrame，空值自动填 NaN
df_results = pd.DataFrame(results).T

# 绘制条形图
x = np.arange(len(df_results.index))
width = 0.35  

fig, ax = plt.subplots(figsize=(8,6))

bars = []
labels = []

# 如果有条纹法
if "条纹法" in df_results.columns:
    bar1 = ax.bar(x - width/2, df_results["条纹法"], width, label="条纹间隔法")
    bars.append(bar1); labels.append("条纹间隔法")

# 如果有拟合法
if "拟合法" in df_results.columns:
    bar2 = ax.bar(x + (width/2 if "条纹法" in df_results.columns else 0), 
                  df_results["拟合法"], width, label="曲线拟合法")
    bars.append(bar2); labels.append("曲线拟合法")

# 坐标轴与标题
ax.set_xlabel("数据集")
ax.set_ylabel("厚度 d (μm)")
ax.set_title("不同附件厚度计算结果对比")
ax.set_xticks(x)
ax.set_xticklabels(df_results.index)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.6)

# 柱子顶部标注数值
for bar_group in bars:
    for bar in bar_group:
        height = bar.get_height()
        if not np.isnan(height):
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig("问题3/分析结果/厚度结果对比图.png", dpi=300, bbox_inches="tight")
plt.show()

# 保存表格
df_results.to_excel("问题3/分析结果/厚度结果对比表.xlsx")
