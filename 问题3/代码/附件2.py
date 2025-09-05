import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib, os

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

output_dir = '问题3/分析结果/附件2(修正）'
os.makedirs(output_dir, exist_ok=True)

# 1. 读取附件3数据

df = pd.read_excel("问题3/附件/附件2.xlsx", header=None)
df = df.drop(0)  
df.columns = ["wavenumber_cm-1", "reflectance_percent"]

# 转换波长 (μm) 反射率
wavelength = 1e4 / df["wavenumber_cm-1"].astype(float).to_numpy()
R_exp = (df["reflectance_percent"].astype(float) / 100).to_numpy()

# ======================
# 2. 折射率模型 (Cauchy)
# ======================
def n1_cauchy(lam, A, B, C):
    return A + B/lam**2 + C/lam**4

# ======================
# 3. Airy 反射率公式
# ======================
theta_i = np.radians(15)  # 入射角 15°
n0, n2 = 1.0, 2.6        # 空气和衬底折射率 

def airy_reflectance(lam, d, A, B, C):
    n1 = n1_cauchy(lam, A, B, C)
    theta1 = np.arcsin(n0*np.sin(theta_i)/n1)
    
    r01 = (n0*np.cos(theta_i) - n1*np.cos(theta1)) / (n0*np.cos(theta_i) + n1*np.cos(theta1))
    r12 = (n1*np.cos(theta1) - n2*np.cos(theta1)) / (n1*np.cos(theta1) + n2*np.cos(theta1))
    
    delta = (2*np.pi/lam) * n1 * d * np.cos(theta1)
    R = (r01**2 + r12**2 + 2*np.abs(r01*r12)*np.cos(2*delta)) / \
        (1 + (r01*r12)**2 + 2*np.abs(r01*r12)*np.cos(2*delta))
    return R

# ======================
# 4. 拟合目标函数
# ======================
def objective(params, lam, R_exp):
    d, A, B, C = params
    R_th = airy_reflectance(lam, d, A, B, C)
    return np.mean((R_th - R_exp)**2)

# ======================
# 5. 优化求解
# ======================
init_params = [3.0, 2.6, 0.01, 0.0]  # 初值: 厚度3 μm, A=2.6, B=0.01, C=0
res = minimize(objective, init_params, args=(wavelength, R_exp), method="Nelder-Mead")
d_fit, A_fit, B_fit, C_fit = res.x

print(f"拟合得到的厚度 d = {d_fit:.4f} μm")
print(f"Cauchy参数: A={A_fit:.4f}, B={B_fit:.6f}, C={C_fit:.6e}")

# ======================
# 6. 可视化对比
# ======================
R_fit = airy_reflectance(wavelength, d_fit, A_fit, B_fit, C_fit)

plt.figure(figsize=(8,5))
plt.plot(wavelength, R_exp, 'b.', label="实验数据")
plt.plot(wavelength, R_fit, 'r-', label=f"拟合曲线 (d={d_fit:.3f} μm)")
plt.xlabel("波长 (μm)")
plt.ylabel("反射率")
plt.title("附件2公式拟合 (15° 入射角)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, "附件2拟合结果.png"), dpi=300, bbox_inches="tight")
plt.show()

# ======================
# 7. 保存分析结果
# ======================

# 计算拟合质量
mse = np.mean((R_fit - R_exp)**2)
rmse = np.sqrt(mse)
r2 = 1 - np.sum((R_exp - R_fit)**2) / np.sum((R_exp - np.mean(R_exp))**2)

# 保存参数和结果
report = f"""
SiC 外延层厚度拟合报告 (15° 入射角)
================================
拟合得到厚度 d = {d_fit:.4f} μm

Cauchy 模型参数:
A = {A_fit:.6f}
B = {B_fit:.6f}
C = {C_fit:.6e}

拟合质量:
R²   = {r2:.6f}
RMSE = {rmse:.6e}

优化方法: Nelder-Mead
"""
with open(os.path.join(output_dir, "附件2拟合报告.txt"), "w", encoding="utf-8") as f:
    f.write(report)

# 保存对比数据表格
df_result = pd.DataFrame({
    "波长 (μm)": wavelength,
    "实验反射率": R_exp,
    "拟合反射率": R_fit
})
df_result.to_excel(os.path.join(output_dir, "附件2拟合曲线数据.xlsx"), index=False)

print(f"R² = {r2:.4f}, RMSE = {rmse:.6f}")
