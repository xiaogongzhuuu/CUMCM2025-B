import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import minimize_scalar
import os, matplotlib

# 实验参数
theta_i = np.radians(15)  # 入射角 15°
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
output_dir = '问题2/分析结果/附件2'
os.makedirs(output_dir, exist_ok=True)

# 菲涅尔反射率
def fresnel_reflectance(n, theta_i):
    theta_t = np.arcsin(np.sin(theta_i) / n)
    r_s = (np.cos(theta_i) - n*np.cos(theta_t)) / (np.cos(theta_i) + n*np.cos(theta_t))
    r_p = (n*np.cos(theta_i) - np.cos(theta_t)) / (n*np.cos(theta_i) + np.cos(theta_t))
    return 0.5 * (r_s**2 + r_p**2)

# 反射率 -> 折射率
def solve_n_from_reflectance(R_val):
    obj = lambda n: (fresnel_reflectance(n, theta_i) - R_val)**2
    return minimize_scalar(obj, bounds=(1.01, 5.0), method='bounded').x

# 数据读取 & 转换
data = pd.read_excel('问题2/附件/附件2.xlsx', header=None, skiprows=1)
data = data.apply(pd.to_numeric, errors='coerce').dropna()
wavelength = 10000 / data.iloc[:, 0].to_numpy()  
reflectance = data.iloc[:, 1].to_numpy() / 100    # 小数

# 平滑
reflectance_filtered = savgol_filter(reflectance, 11, 3)

# 折射率拟合
n_measured = np.array([solve_n_from_reflectance(R) for R in reflectance_filtered])
valid = (n_measured > 1) & (n_measured < 5)
X = np.vstack([np.ones(np.sum(valid)), 1/wavelength[valid]**2, 1/wavelength[valid]**4]).T
A, B, C = np.linalg.lstsq(X, n_measured[valid], rcond=None)[0]

def n_cauchy(lam): 
    return A + B/lam**2 + C/lam**4

# 拟合质量
n_fit = n_cauchy(wavelength[valid])
r2 = 1 - np.sum((n_measured[valid]-n_fit)**2)/np.sum((n_measured[valid]-np.mean(n_measured[valid]))**2)
rmse = np.sqrt(np.mean((n_measured[valid]-n_fit)**2))

# -------- 厚度计算部分 --------
peaks, _ = find_peaks(reflectance_filtered, distance=30)
lambda_peaks = wavelength[peaks]
delta_lambda = np.abs(np.diff(lambda_peaks))
delta_lambda_mean = np.mean(delta_lambda)
lambda_center = np.median(lambda_peaks)
n_center = n_cauchy(lambda_center)
theta_t = np.arcsin(np.sin(theta_i) / n_center)
d = (lambda_center**2) / (2 * n_center * np.cos(theta_t) * delta_lambda_mean)

# 保存报告
report = f"""
SiC 15° 入射角分析报告
====================
Cauchy 拟合参数:
A = {A:.6f}, B = {B:.6f}, C = {C:.6e}
拟合质量:
R²   = {r2:.6f}
RMSE = {rmse:.6e}
厚度计算:
d ≈ {d:.4f} μm
Δλ ≈ {delta_lambda_mean:.4f} μm
λ_center ≈ {lambda_center:.4f} μm
n_center ≈ {n_center:.4f}
"""
with open(os.path.join(output_dir, "分析报告_附件2.txt"), "w", encoding="utf-8") as f:
    f.write(report)

# 导出结果表格
pd.DataFrame({"波长_μm": wavelength, "折射率": n_cauchy(wavelength)}).to_excel(
    os.path.join(output_dir, "波长_折射率_附件2.xlsx"), index=False
)
pd.DataFrame({"条纹波长_μm": lambda_peaks, "条纹间隔Δλ_μm": np.append(delta_lambda, np.nan)}).to_excel(
    os.path.join(output_dir, "条纹间隔_附件2.xlsx"), index=False
)

# 绘图1：折射率拟合 + 反射率验证
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(wavelength[valid], n_measured[valid], s=20, label='实验')
ax1.plot(wavelength[valid], n_fit, 'r-', label='拟合')
ax1.set_xlabel('波长 (μm)'); ax1.set_ylabel('折射率')
ax1.set_title(f'折射率拟合 (R²={r2:.3f}, RMSE={rmse:.4f})')
ax1.legend(); ax1.grid(alpha=0.3)

R_fit = np.array([fresnel_reflectance(n, theta_i) for n in n_fit])
ax2.scatter(wavelength[valid], reflectance_filtered[valid]*100, s=20, label='实验反射率')
ax2.plot(wavelength[valid], R_fit*100, 'r-', label='拟合反射率')
ax2.set_xlabel('波长 (μm)'); ax2.set_ylabel('反射率 (%)')
ax2.set_title('反射率验证'); ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'SiC_15度拟合.png'), dpi=300)

# 绘图2：干涉条纹标记
plt.figure(figsize=(8, 5))
plt.plot(wavelength, reflectance_filtered*100, label="滤波反射率", color="blue")
plt.scatter(lambda_peaks, reflectance_filtered[peaks]*100, color="red", s=40, label="条纹峰")
plt.xlabel("波长 (μm)"); plt.ylabel("反射率 (%)")
plt.title(f"干涉条纹标记 (Δλ≈{delta_lambda_mean:.4f} μm)")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "干涉条纹标记.png"), dpi=300)
plt.show()
