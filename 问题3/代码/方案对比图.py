import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def create_method_comparison_plots():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flat

    # 1. 方法适用性范围图
    thickness = np.logspace(-1, 2, 100)
    fringe_acc = np.where(thickness > 5, 0.95, np.where(thickness > 1, 0.7, 0.3))
    airy_acc = np.where(thickness < 10, 0.85, np.where(thickness < 50, 0.6, 0.3))
    ax1.semilogx(thickness, fringe_acc, 'b-', label='条纹间隔法', lw=2)
    ax1.semilogx(thickness, airy_acc, 'r-', label='Airy拟合法', lw=2)
    ax1.axvline(2, c='r', ls='--', alpha=0.7, label='本研究-Airy结果')
    ax1.axvline(15, c='b', ls='--', alpha=0.7, label='本研究-条纹结果')
    ax1.set(xlabel='厚度 (μm)', ylabel='测量精度', title='(a) 两种方法适用性对比')
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # 2. 结果对比柱状图
    methods = ['条纹间隔法\n(10°)', '条纹间隔法\n(15°)', 'Airy拟合\n(10°)', 'Airy拟合\n(15°)']
    thicknesses = [14.42, 14.73, 1.87, 1.84]
    colors = ['skyblue', 'lightblue', 'salmon', 'lightcoral']
    bars = ax2.bar(methods, thicknesses, color=colors, alpha=0.8)
    ax2.set(ylabel='测量厚度 (μm)', title='(b) SiC样品厚度测量结果')
    for bar, val in zip(bars, thicknesses):
        ax2.text(bar.get_x()+bar.get_width()/2, val+0.2, f'{val:.2f}',
                 ha='center', va='bottom', fontweight='bold')

    # 3. 拟合质量对比
    r2_vals = [0.200, 0.210, 0.257, 0.261]
    bars3 = ax3.bar(methods, r2_vals, color=colors, alpha=0.8)
    ax3.set(ylabel='拟合质量 (R²)', title='(c) 拟合质量对比', ylim=(0, 1))
    for thr, color, lbl in [(0.8,'g','优秀'), (0.6,'orange','良好'), (0.3,'r','可接受')]:
        ax3.axhline(thr, c=color, ls='--', alpha=0.7, label=f'{lbl} (R²>{thr})')
    for bar, r2 in zip(bars3, r2_vals):
        ax3.text(bar.get_x()+bar.get_width()/2, r2+0.01, f'{r2:.3f}',
                 ha='center', va='bottom', fontweight='bold')
    ax3.legend(fontsize=8)

    # 4. 误差来源雷达图
    categories = ['峰位识别', '色散近似', '界面粗糙度', '模型复杂度', '环境因素']
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    data = {
        "条纹间隔法": [0.3, 0.6, 0.2, 0.1, 0.3],
        "Airy拟合法": [0.1, 0.4, 0.7, 0.8, 0.3]
    }
    for label, vals, color in zip(data.keys(), data.values(), ['blue','red']):
        vals = vals + vals[:1]
        ax4.plot(angles, vals, 'o-', lw=2, label=label, color=color)
        ax4.fill(angles, vals, alpha=0.25, color=color)
    ax4.set_xticks(angles[:-1]); ax4.set_xticklabels(categories)
    ax4.set_ylim(0,1); ax4.set_title('(d) 误差来源分析', y=1.08)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3,1.0))

    # 5. 光谱特征对比
    wavelength = np.linspace(800, 2500, 1000)
    thick_spectrum = 0.3 + 0.2*np.sin(2*np.pi*wavelength/50)*np.exp(-(wavelength-1200)**2/3e5)
    thin_spectrum  = 0.3 + 0.15*np.sin(2*np.pi*wavelength/300)*np.exp(-(wavelength-1200)**2/3e5)
    ax5.plot(wavelength, thick_spectrum, 'b-', lw=2, label='厚膜 (~15μm)')
    ax5.plot(wavelength, thin_spectrum+0.1, 'r-', lw=2, label='薄膜 (~2μm)')
    ax5.set(xlabel='波长 (nm)', ylabel='反射率', title='(e) 光谱特征差异示意')
    ax5.legend(); ax5.grid(alpha=0.3)

    # 6. 结论总结
    ax6.axis('off')
    conclusion = """物理原因分析结论:

1. 厚度差异原因:
   • 条纹间隔法: 测量总外延层厚度
   • Airy拟合法: 测量表面有效层厚度
2. SiC外延层结构:
   • 可能具有梯度折射率分布
   • 表层2μm: 成分/密度变化层
   • 总厚度15μm: 完整外延层
3. 方法选择建议:
   • 厚膜(>5μm): 条纹间隔法
   • 薄膜(<5μm): Airy拟合法
   • 复杂结构: 多方法联合分析
4. 拟合质量改进:
   • 考虑界面粗糙度
   • 多层膜模型
   • 梯度折射率模型"""
    ax6.text(0.05, 0.95, conclusion, transform=ax6.transAxes,
             fontsize=10, va='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    ax6.set_title('(f) 物理机制分析结论', fontweight='bold')

    plt.tight_layout()
    plt.savefig('问题3/分析结果/方法对比分析.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_method_comparison_plots()
