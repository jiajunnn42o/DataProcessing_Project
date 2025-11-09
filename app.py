# app.py
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap  # 渐变色用

# ----------------------------
# 页面设置
# ----------------------------
st.set_page_config(
    page_title="CO₂ Visualisation System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# 读取 theme（完全以 config.toml 为准）
# ----------------------------
BG   = st.get_option("theme.backgroundColor") or "#FFFFFF"
TXT  = st.get_option("theme.textColor") or "#262730"
SEC  = st.get_option("theme.secondaryBackgroundColor") or "#F7F7F7"
PRIMARY = st.get_option("theme.primaryColor") or "#ff41ec"

def _hex_to_rgb(h):
    h = (h or "#FFFFFF").lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _luma(hex_color):
    r, g, b = _hex_to_rgb(hex_color)
    return (0.2126*r + 0.7152*g + 0.0722*b) / 255.0

# 网格颜色：根据背景亮度自动取浅/深
GRID = "#E5E7EB" if _luma(BG) >= 0.5 else "#3A3F47"

# Matplotlib / Seaborn 只做必要同步（不注入任何 CSS）
mpl.rcdefaults()
sns.reset_orig()
mpl.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.transparent": True,
    "text.color": TXT,
    "axes.labelcolor": TXT,
    "axes.edgecolor": GRID,
    "xtick.color": TXT,
    "ytick.color": TXT,
    "axes.titlecolor": TXT,
    "grid.color": GRID,
    "legend.facecolor": "none",
    "legend.edgecolor": GRID,
})
sns.set_style("whitegrid", {"axes.facecolor": BG, "grid.color": GRID})

def apply_theme(ax):
    ax.title.set_color(TXT)
    ax.xaxis.label.set_color(TXT)
    ax.yaxis.label.set_color(TXT)
    ax.tick_params(axis="both", colors=TXT)
    ax.grid(True, color=GRID, alpha=0.3)
    for sp in ax.spines.values():
        sp.set_color(GRID)

    # legend：深浅模式分别设置背景
    leg = ax.get_legend()
    if leg:
        if _luma(BG) >= 0.5:  # 浅色背景
            leg.get_frame().set_facecolor((1, 1, 1, 0.85))
        else:                 # 深色背景
            leg.get_frame().set_facecolor((0.1, 0.1, 0.1, 0.85))
        leg.get_frame().set_edgecolor(GRID)
        leg.get_frame().set_linewidth(1.0)
        if leg.get_title():
            leg.get_title().set_color(TXT)
        for t in leg.get_texts():
            t.set_color(TXT)

# ----------------------------
# 数据
# ----------------------------
DATA_PATH = "data/co2_clean_asean.csv"
df = pd.read_csv(DATA_PATH)
df["Year"] = df["Year"].astype(int)
df["CO2_per_capita"] = pd.to_numeric(df["CO2_per_capita"], errors="coerce")
df = df.dropna(subset=["CO2_per_capita"])

ASEAN = [
    "Malaysia","Singapore","Thailand","Indonesia","Vietnam",
    "Philippines","Cambodia","Lao PDR","Myanmar","Brunei Darussalam"
]
asean_mean = (
    df[df["Country Name"].isin(ASEAN)]
    .groupby("Year")["CO2_per_capita"]
    .mean()
    .reset_index()
)
asean_mean["Country Name"] = "ASEAN Average"
combined = pd.concat([df, asean_mean], ignore_index=True)

# ----------------------------
# 控件
# ----------------------------
st.sidebar.header("Controls")

PRIMARY = st.get_option("theme.primaryColor") or "#ff41ec"

# 定义内部逻辑值 + 显示标签
labels = {
    "line": "**Line** Malaysia vs ASEAN vs World",
    "bar": "**Bar** Latest Year (ASEAN + World)",
    "heat": "**Heatmap** ASEAN & World (1990-2023)",
}

# 创建 radio（保持逻辑判断不变）
view = st.sidebar.radio(
    "Visualisation",
    options=["line", "bar", "heat"],
    format_func=lambda k: labels[k]
)

# 纯静态粉紫框样式（不变色、不hover）
st.markdown(f"""
<style>
:root {{
  --primary: {PRIMARY};
}}

/* 调整整体行距与字体大小 */
[data-testid="stSidebar"] .stRadio > div > label {{
  margin-bottom: 18px;  /* 控制选项之间的间距 */
}}

/* 保持行高与原始字体 */
[data-testid="stSidebar"] .stRadio > div > label p {{
  line-height: 1.35;
  font-size: 1rem;
}}

/* 让第一个粗体词（Line / Bar / Heatmap）变成粉紫色框 */
[data-testid="stSidebar"] .stRadio label p strong:first-child {{
  color: var(--primary);
  font-weight: 700;
  border: 1px solid var(--primary);
  padding: 2px 8px;
  border-radius: 999px;
  margin-right: 6px;
}}
</style>
""", unsafe_allow_html=True)


year_min, year_max = 1990, 2023
year_range = st.sidebar.slider(
    "Year range", min_value=year_min, max_value=year_max,
    value=(year_min, year_max), step=1
)


# ----------------------------
# 标题
# ----------------------------
st.title("CO₂ Emissions per Capita : Malaysia in ASEAN Context")
st.caption("Source: World Bank (EN.GHG.CO2.PC.CE.AR5) - Metric tons per capita")

# ----------------------------
# 视图
# ----------------------------
if view.startswith("line"):
    plot_df = combined[
        combined["Country Name"].isin(["Malaysia", "ASEAN Average", "World"])
        & (combined["Year"].between(*year_range))
    ]

    palette = {
    "Malaysia": PRIMARY,          # 主色 (粉紫)
    "ASEAN Average": "#9b8aff",   # 淡紫蓝
    "World": "#6fc3ff"            # 柔和蓝
}
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    sns.lineplot(
        data=plot_df, x="Year", y="CO2_per_capita",
        hue="Country Name", palette=palette, marker="o", linewidth=2.5, ax=ax
    )
    ax.set_ylim(0, 9)
    ax.set_xlabel("Year"); ax.set_ylabel("CO$_2$ (metric tons per capita)")
    ax.set_title("Malaysia vs ASEAN Average vs World")
    apply_theme(ax)
    plt.tight_layout(pad=1.5)
    st.pyplot(fig)

    st.markdown(
        f"> **Observation:** Since the late 1990s, Malaysia's per capita emissions have been consistently higher than the **world** average,"
        "and the gap with the **ASEAN** average widened in the 2000s but has gradually narrowed in recent years."
    )

elif view.startswith("bar"):
    latest_year = min(max(df["Year"]), year_range[1])
    latest = df[
        (df["Year"] == latest_year) & (df["Country Name"].isin(ASEAN + ["World"]))
    ].sort_values("CO2_per_capita", ascending=False)

    fig, ax = plt.subplots(figsize=(8.8, 5))

    # —— 渐变调色（方案1）：主色 → 中性灰蓝 → 深灰 —— #
    cmap = LinearSegmentedColormap.from_list(
        "rank_grad",
        [PRIMARY, "#8B8FA7", "#2B2F36"],
        N=len(latest)
    )
    denom = max(1, len(latest) - 1)
    bar_palette = [cmap(i/denom) for i in range(len(latest))]

    sns.barplot(
        data=latest,
        x="CO2_per_capita",
        y="Country Name",
        palette=bar_palette,
        ax=ax
    )

    ax.set_xlabel("CO$_2$ (metric tons per capita)"); ax.set_ylabel("Country")
    ax.set_title(f"Latest Year Comparison – {latest_year}")
    for i, v in enumerate(latest["CO2_per_capita"]):
        ax.text(v + 0.1, i, f"{v:.1f}", va="center", color=TXT)
    apply_theme(ax)
    plt.tight_layout(pad=1.5)
    st.pyplot(fig)

    st.markdown(
        f"> **Observation:** In **{latest_year}**, **Malaysia** was located upstream in **ASEAN**; **Brunei/SG** was significantly higher,"
        " and**Cambodia/Myanmar** was lower; **World** was used as the baseline reference."
    )

else:
    # --- 数据与透视 ---
    hm = df[df["Country Name"].isin(ASEAN + ["World"])]
    hm = hm[hm["Year"].between(*year_range)]
    pivot = hm.pivot_table(values="CO2_per_capita",
                           index="Country Name", columns="Year", aggfunc="mean")

    # --- 按最近年份排序（滑块上限内的最近年） ---
    latest_year = min(max(df["Year"]), year_range[1])
    if latest_year in pivot.columns:
        order = pivot[latest_year].sort_values(ascending=False).index
        pivot = pivot.loc[order]

    # --- 让色阶更有对比：按 5%~95% 分位裁切 ---
    vals = pivot.values.astype(float)
    vmin, vmax = np.nanpercentile(vals, [5, 95])

    # --- 与主题更协调的配色（基于 primaryColor） ---
    if _luma(BG) < 0.5:
        # 深色：从面板深灰 -> 过渡蓝灰 -> 主色
        cmap = LinearSegmentedColormap.from_list(
            "heat_dark", [SEC, "#3b4252", PRIMARY], N=256
        )
    else:
        # 浅色：从很浅粉 -> 主色 -> 深紫
        cmap = LinearSegmentedColormap.from_list(
            "heat_light", ["#fff1fa", PRIMARY, "#4a154b"], N=256
        )

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(12, 6))
    h = sns.heatmap(
        pivot, cmap=cmap, vmin=vmin, vmax=vmax,
        cbar_kws={"label": "CO$_2$ per capita"},
        ax=ax, linewidths=0.4, linecolor=GRID, square=False
    )

    # 色条样式
    cbar = h.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(TXT)
    cbar.ax.tick_params(colors=TXT)
    if cbar.outline:  # 描边用网格色
        cbar.outline.set_edgecolor(GRID)

    # 轴标题与刻度：年份每隔 2 年显示一次，保持水平
    years = list(pivot.columns)
    step = 2 if len(years) > 15 else 1
    ax.set_xticks(range(0, len(years), step))
    ax.set_xticklabels(years[::step], rotation=0)
    ax.set_xlabel("Year"); ax.set_ylabel("Country")
    ax.set_title("Heatmap of CO$_2$ per Capita (ASEAN & World)")

    # 可选高亮 Malaysia / World 行
    for target, color, lw in [("Malaysia", PRIMARY, 2.0), ("World", "#1f77b4", 1.8)]:
        if target in pivot.index:
            i = list(pivot.index).index(target)
            ax.add_patch(plt.Rectangle(
                (0, i), len(pivot.columns), 1, fill=False, lw=lw, edgecolor=color
            ))

    apply_theme(ax)
    st.pyplot(fig)

    st.markdown(
        "> **Observation:** The heat map is sorted by the most recent year and cropped for contrast to highlight the relative levels of each country;"
        " Malaysia saw a significant increase in the 2000s and has stabilized in recent years."
    )


st.divider()
st.caption("Tip: Use the sidebar to adjust year range or switch visualisations.")
