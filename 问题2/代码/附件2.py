import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import minimize_scalar
import os
import matplotlib

# 物理常数
e = 1.602e-19
epsilon_0 = 8.854e-12
m_star = 0.25 * 9.109e-31
c = 2.998e8

# 实验参数
incident_angle = 15.0  
theta_i = np.radians(incident_angle)


matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
output_dir = '问题2/分析结果/附件2'
os.makedirs(output_dir, exist_ok=True)

def fresnel_reflectance(n, theta_i):
    """计算菲涅尔反射率"""
    sin_theta_t = np.clip(np.sin(theta_i) / n, 0, 0.999)
    theta_t = np.arcsin(sin_theta_t)
    
    cos_i = np.cos(theta_i)
    cos_t = np.cos(theta_t)
    
    r_s = (cos_i - n * cos_t) / (cos_i + n * cos_t)
    r_p = (n * cos_i - cos_t) / (n * cos_i + cos_t)
    
    return 0.5 * (r_s**2 + r_p**2)

def solve_n_from_reflectance(R_val):
    """从反射率求解折射率"""
    if R_val <= 0.01: return 1.01
    if R_val >= 0.99: return 3.99
    
    def objective(n):
        return (fresnel_reflectance(n, theta_i) - R_val)**2
    
    try:
        result = minimize_scalar(objective, bounds=(1.01, 5.0), method='bounded')
        return result.x if result.success else 2.0
    except:
        sqrt_R = np.sqrt(np.clip(R_val, 0.01, 0.99))
        return (1 + sqrt_R) / (1 - sqrt_R)

# 读取数据
data = pd.read_excel('问题2/附件/附件2.xlsx', header=None)
data = data.dropna().apply(pd.to_numeric, errors='coerce').dropna()

# 数据处理
wavelength = 10000 / data.iloc[:, 0]  
reflectance = data.iloc[:, 1] / 100

# 滤波
if len(reflectance) > 5:
    window_length = min(11, len(reflectance) if len(reflectance) % 2 == 1 else len(reflectance) - 1)
    window_length = max(window_length, 5)
    reflectance_filtered = savgol_filter(reflectance, window_length, 3)
    print(f"滤波完成 (窗口: {window_length})")
else:
    reflectance_filtered = reflectance

# 计算折射率
n_measured = np.array([solve_n_from_reflectance(R) for R in reflectance_filtered])

# 三项Cauchy拟合
valid = (n_measured > 1) & (n_measured < 5)
X = np.vstack([np.ones(np.sum(valid)), 1/wavelength[valid]**2, 1/wavelength[valid]**4]).T
A, B, C = np.linalg.lstsq(X, n_measured[valid], rcond=None)[0]

print(f"拟合参数: A = {A:.4f}, B = {B:.6f}, C = {C:.8f}")

# 折射率函数 - 简化版本，去掉复杂的Drude修正
def complete_n_3term(lam):
    """简化的Cauchy色散模型"""
    return A + B/lam**2 + C/lam**4

# 评估拟合质量
n_fitted = complete_n_3term(wavelength[valid])
r2 = 1 - np.sum((n_measured[valid] - n_fitted)**2) / np.sum((n_measured[valid] - np.mean(n_measured[valid]))**2)
rmse = np.sqrt(np.mean((n_measured[valid] - n_fitted)**2))

print(f"拟合质量: R² = {r2:.4f}, RMSE = {rmse:.6f}")

# 生成结果表格
result = pd.DataFrame({
    '波长_微米': wavelength,
    '折射率': complete_n_3term(wavelength)
})
result.to_excel(os.path.join(output_dir, '波长_折射率_附件2.xlsx'), index=False)

# 核心验证图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('SiC 15°入射角拟合结果', fontsize=14, fontweight='bold')

# 折射率对比
ax1.scatter(wavelength[valid], n_measured[valid], alpha=0.8, color='blue', s=30, label='实验值')
ax1.plot(wavelength[valid], n_fitted, 'r-', linewidth=2, label='拟合值')
ax1.set_xlabel('波长 (μm)')
ax1.set_ylabel('折射率')
ax1.set_title(f'折射率拟合 (R² = {r2:.4f})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 反射率验证
R_fitted = np.array([fresnel_reflectance(n, theta_i) for n in n_fitted])
ax2.scatter(wavelength[valid], reflectance_filtered[valid]*100, alpha=0.8, color='blue', s=30, label='实验反射率')
ax2.plot(wavelength[valid], R_fitted*100, 'r-', linewidth=2, label='拟合反射率')
ax2.set_xlabel('波长 (μm)')
ax2.set_ylabel('反射率 (%)')
ax2.set_title('反射率验证')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'SiC_15度拟合.png'), dpi=300, bbox_inches='tight')
plt.show()

# 简化报告
report = f"""
SiC 15°入射角分析报告
===================

拟合参数:
A = {A:.6f}
B = {B:.6f} μm²
C = {C:.8f} μm⁴

拟合质量:
R² = {r2:.6f}
RMSE = {rmse:.8f}

数据统计:
有效数据点: {len(n_measured[valid])}
波长范围: {wavelength[valid].min():.1f}-{wavelength[valid].max():.1f} μm
折射率范围: {n_measured[valid].min():.3f}-{n_measured[valid].max():.3f}
"""

with open(os.path.join(output_dir, '分析报告_附件2.txt'), 'w', encoding='utf-8') as f:
    f.write(report)

print(f"R² = {r2:.4f}, RMSE = {rmse:.6f}")

from scipy.signal import find_peaks

# 厚度计算部分 - 修正核心公式
# 1. 提取条纹极值点

peaks, _ = find_peaks(reflectance_filtered, distance=30)  # 可调 distance 参数
lambda_peaks = wavelength[peaks]

# 计算相邻条纹间隔 Δλ（取绝对值保证为正）
delta_lambda = np.abs(np.diff(lambda_peaks))
delta_lambda_mean = np.mean(delta_lambda)

print(f"相邻条纹间隔 Δλ (前10个): {delta_lambda[:10]}")
print(f"平均条纹间隔 Δλ: {delta_lambda_mean:.4f} μm")

# 2. 厚度计算 - 修正公式
lambda_center = np.median(lambda_peaks)   # 选择中间条纹作为代表
n_center = complete_n_3term(lambda_center)  # 对应折射率

# Snell 定律求折射角
theta_t = np.arcsin(np.sin(theta_i) / n_center)

# 修正的厚度计算公式：d = Δλ / (2 * n * cos(θ))
d = delta_lambda_mean / (2 * n_center * np.cos(theta_t))

print(f"外延层厚度 d ≈ {d:.4f} μm")

# 3. 保存条纹与厚度结果

df_stripes = pd.DataFrame({
    "条纹波长_μm": lambda_peaks,
    "条纹间隔Δλ_μm": np.append(delta_lambda, np.nan)
})
df_stripes.to_excel(os.path.join(output_dir, "条纹间隔_附件2.xlsx"), index=False)

with open(os.path.join(output_dir, "厚度计算结果_附件2.txt"), "w", encoding="utf-8") as f:
    f.write(f"外延层厚度 d ≈ {d:.4f} μm\n")
    f.write(f"平均条纹间隔 Δλ ≈ {delta_lambda_mean:.4f} μm\n")
    f.write(f"中心波长 λ ≈ {lambda_center:.4f} μm\n")
    f.write(f"中心折射率 n ≈ {n_center:.4f}\n")

# 干涉条纹标记图
plt.figure(figsize=(8, 5))
plt.plot(wavelength, reflectance_filtered * 100, label="滤波后反射率", color="blue", linewidth=1.5)
plt.scatter(lambda_peaks, reflectance_filtered[peaks] * 100,
            color="red", marker="o", s=40, label="干涉峰")
for i, lam in enumerate(lambda_peaks[:10]):  # 只标前10个，避免太乱
    plt.text(lam, reflectance_filtered[peaks[i]] * 100 + 1,
             f"{lam:.2f}", fontsize=8, rotation=45)

plt.xlabel("波长 (μm)")
plt.ylabel("反射率 (%)")
plt.title(f"干涉条纹标记 (Δλ≈{delta_lambda_mean:.4f} μm)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "干涉条纹标记.png"), dpi=300, bbox_inches="tight")
plt.show()