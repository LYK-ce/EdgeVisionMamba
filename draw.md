

## 背景
编写一个Python脚本来绘图，用于学术论文发表。

## 学术规范要求

### 1. 字体与排版
- 字体：Times New Roman（西文）或 serif 族
- 标题字号：14-16pt，加粗
- 坐标轴标签：12-14pt，加粗
- 刻度标签：10-12pt
- 图例字号：10-12pt
- 确保缩放至论文尺寸（单栏~3.5in，双栏~7in）后仍清晰可读

### 2. 配色方案
- NATURE配色方案 (https://color.amfe.space/#/palette/101):
  - #CC247C (Magenta)
  - #E95351 (Red)
  - #F7A24F (Orange)
  - #FBEB66 (Yellow)
  - #4EA660 (Green)
  - #79CAFB (Light Blue)
  - #5C9AD4 (Blue)
  - #AA77E9 (Purple)

- 同时提供hatch patterns（填充图案），支持黑白打印
- 线条/边框颜色统一使用黑色

### 3. 图表元素
- 网格线：虚线(--), 透明度0.3, 置于图形底层
- 边框线宽：1.0-1.2pt
- 柱状图间距：合理的bar width（0.6-0.8组宽度）
- 图例：带边框，位于不遮挡数据的位置
- 分辨率：300 DPI（打印质量）

### 4. 必须的matplotlib配置
```python
ACADEMIC_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'axes.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
}
plt.rcParams.update(ACADEMIC_STYLE)
```
