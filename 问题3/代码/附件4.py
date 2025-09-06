import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import find_peaks, savgol_filter
import os, matplotlib

# 实验参数
theta_i = np.radians(15)  # 入射角 15°
n0, n2 = 1.0, 3.4         # 空气 / 衬底折射率
output_dir = '问题3/分析结果/附件4'
os.makedirs(output_dir, exist_ok=True)

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# --- 折射率模型 ---
def n_cauchy(lam, A, B, C):
    return A + B/lam**2 + C/lam**4

# --- Airy反射率 ---
def airy_reflectance(lam, d, A, B, C):
    n1 = n_cauchy(lam, A, B, C)
    theta1 = np.arcsin(n0*np.sin(theta_i)/n1)
    r01 = (n0*np.cos(theta_i) - n1*np.cos(theta1)) / (n0*np.cos(theta_i) + n1*np.cos(theta1))
    r12 = (n1*np.cos(theta1) - n2*np.cos(theta1)) / (n1*np.cos(theta1) + n2*np.cos(theta1))
    delta = (2*np.pi/lam) * n1 * d * np.cos(theta1)
    return (r01**2 + r12**2 + 2*np.abs(r01*r12)*np.cos(2*delta)) / \
           (1 + (r01*r12)**2 + 2*np.abs(r01*r12)*np.cos(2*delta))

# --- 数据读取 ---
df = pd.read_excel("问题3/附件/附件4.xlsx", header=None, skiprows=1)
wavelength = 1e4 / df.iloc[:,0].astype(float).to_numpy()
R_exp = df.iloc[:,1].astype(float).to_numpy() / 100

# --- 数据平滑 ---
reflectance_filtered = savgol_filter(R_exp, 11, 3)

# --- 优化求解 (基于原始数据) ---
init_params = [3.0, 2.6, 0.01, 0.0]
res = minimize(objective := lambda params, lam, R_exp: 
               np.mean((airy_reflectance(lam, *params) - R_exp)**2),
               init_params, args=(wavelength, R_exp), method="Nelder-Mead")
d_fit, A_fit, B_fit, C_fit = res.x

# --- 拟合结果 ---
R_fit = airy_reflectance(wavelength, d_fit, A_fit, B_fit, C_fit)
r2 = 1 - np.sum((R_exp - R_fit)**2) / np.sum((R_exp - np.mean(R_exp))**2)
rmse = np.sqrt(np.mean((R_fit - R_exp)**2))

# --- 图1: 原始 vs 滤波 vs 拟合 ---
plt.figure(figsize=(8,5))
plt.plot(wavelength, R_exp, 'b.', alpha=0.5, label="实验数据(原始)")
plt.plot(wavelength, reflectance_filtered, 'g-', linewidth=1, label="滤波后数据")
plt.plot(wavelength, R_fit, 'r-', label=f"拟合曲线 (d={d_fit:.3f} μm)")
plt.xlabel("波长 (μm)"); plt.ylabel("反射率")
plt.title("附件4公式拟合 (15° 入射角)")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig(os.path.join(output_dir, "附件4_拟合对比.png"), dpi=300)
plt.show()

# --- 图2: 干涉条纹 (基于滤波数据) ---
peaks, _ = find_peaks(reflectance_filtered, distance=30)
lambda_peaks = wavelength[peaks]
delta_lambda = np.abs(np.diff(lambda_peaks))
delta_lambda_mean = np.mean(delta_lambda)

plt.figure(figsize=(8,5))
plt.plot(wavelength, reflectance_filtered*100, label="滤波反射率", color="blue")
plt.scatter(lambda_peaks, reflectance_filtered[peaks]*100, color="red", s=40, label="干涉峰")
for i, lam in enumerate(lambda_peaks[:10]):
    plt.text(lam, reflectance_filtered[peaks[i]]*100 + 1, f"{lam:.2f}", fontsize=8, rotation=45)
plt.xlabel("波长 (μm)"); plt.ylabel("反射率 (%)")
plt.title(f"干涉条纹标记 (Δλ≈{delta_lambda_mean:.4f} μm)")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig(os.path.join(output_dir, "附件4_干涉条纹标记.png"), dpi=300)
plt.show()

# --- 导出条纹数据 ---
pd.DataFrame({"条纹波长_μm": lambda_peaks, 
              "条纹间隔Δλ_μm": np.append(delta_lambda, np.nan)}).to_excel(
    os.path.join(output_dir, "附件4_条纹间隔数据.xlsx"), index=False
)

# --- 蒙特卡洛分析 (基于滤波数据) ---
n_trials = 200
d_values = []
for _ in range(n_trials):
    noise = np.random.normal(0, 0.005, size=reflectance_filtered.shape)
    R_noisy = np.clip(reflectance_filtered + noise, 0, 1)
    obj = lambda d: np.mean((airy_reflectance(wavelength, d, A_fit, B_fit, C_fit) - R_noisy)**2)
    d_noisy = minimize(obj, d_fit, method="Nelder-Mead").x[0]
    d_values.append(d_noisy)

plt.figure(figsize=(7,5))
plt.hist(d_values, bins=20, color="skyblue", edgecolor="black")
plt.axvline(np.mean(d_values), color="red", linestyle="--", 
            label=f"均值={np.mean(d_values):.3f} μm")
plt.xlabel("厚度 d (μm)")
plt.ylabel("频数")
plt.title("厚度蒙特卡洛分布 (基于滤波数据)")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "附件4_厚度分布_MC.png"), dpi=300)
plt.show()

