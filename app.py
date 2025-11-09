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

def color_map_for(selected, primary="#ff41ec"):
    """给多选系列分配颜色：Malaysia 用主色，World 和 ASEAN Average 用固定清爽配色，其它轮询柔和色。"""
    nice_cycle = [
        "#8bd6a8", "#f9a8d4", "#fcd34d", "#a7f3d0", "#93c5fd",
        "#fca5a5", "#c4b5fd", "#86efac", "#fde68a", "#7dd3fc"
    ]
    m = {}
    for name in selected:
        if name == "Malaysia":
            m[name] = primary
        elif name == "ASEAN Average":
            m[name] = "#9b8aff"   # 淡紫蓝（与你现在风格一致）
        elif name == "World":
            m[name] = "#6fc3ff"   # 柔和蓝
        else:
            m[name] = nice_cycle[0] if nice_cycle else "#bbbbbb"
            if nice_cycle: nice_cycle.pop(0)
    return m

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

# --- 候选国家列表（含 ASEAN Average 与 World） ---
COUNTRIES = (
    sorted(df["Country Name"].unique().tolist())
    + (["ASEAN Average"] if "ASEAN Average" not in df["Country Name"].unique() else [])
)
# 默认选择：Malaysia + ASEAN Average + World（存在就选）
DEFAULT_LINE_SET = [x for x in ["Malaysia", "ASEAN Average", "World"] if x in COUNTRIES]
if not DEFAULT_LINE_SET:  # 兜底至少一个
    DEFAULT_LINE_SET = COUNTRIES[:1]

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

# 仅当 Line 视图时出现“多选国家”
if view == "line":
    st.sidebar.markdown("### Countries")
    selected_countries = st.sidebar.multiselect(
        "Select up to 6 series", 
        options=COUNTRIES,
        default=DEFAULT_LINE_SET,
        max_selections=6,
        help="Pick countries/aggregates to compare (Malaysia/ASEAN Average/World included).",
    )
else:
    selected_countries = DEFAULT_LINE_SET  # 其他视图不使用


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
    # 1) 校验选择
    if not selected_countries:
        st.info("Please select at least one country/aggregate in the sidebar.")
        st.stop()

    # 2) 过滤数据（从 combined 里取，包含 ASEAN Average）
    plot_df = combined[
        combined["Country Name"].isin(selected_countries)
        & (combined["Year"].between(*year_range))
    ].copy()

    # 3) 颜色：Malaysia 用主色，其它自动分配
    PRIMARY = st.get_option("theme.primaryColor") or "#ff41ec"
    palette = color_map_for(selected_countries, primary=PRIMARY)

    # 4) 画图
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    sns.lineplot(
        data=plot_df, x="Year", y="CO2_per_capita",
        hue="Country Name", palette=palette,
        marker="o", linewidth=2.5, ax=ax
    )
    ax.set_xlabel("Year"); ax.set_ylabel("CO$_2$ (metric tons per capita)")
    # —— 动态 y 轴范围：按数据自动缩放并留 10% 边距 —— #
    ymin = float(plot_df["CO2_per_capita"].min())
    ymax = float(plot_df["CO2_per_capita"].max())
    pad  = max(0.2, (ymax - ymin) * 0.10)  # 至少留 0.2 的视觉缓冲
    ax.set_ylim(bottom=max(0.0, ymin - pad), top=ymax + pad)

    # —— 动态标题：把选中的系列名拼成 “A vs B vs C …” —— #
    def short_title(names, limit=4):
        # 保持 Malaysia 在最前，其次 World/ASEAN Average，再到其它
        order_key = {"Malaysia": 0, "World": 1, "ASEAN Average": 2}
        names_sorted = sorted(set(names), key=lambda x: order_key.get(x, 99))
        if len(names_sorted) <= limit:
            return " vs ".join(names_sorted)
        return " vs ".join(names_sorted[:limit]) + " …"
    ax.set_title(short_title(selected_countries))

    apply_theme(ax)
    plt.tight_layout(pad=1.5)
    st.pyplot(fig)

    # 5) 简短解读（可按需保留/修改）
    # --- 自动生成针对 Malaysia 的观察结论 ---
    if "Malaysia" in selected_countries:
        malaysia_mean = plot_df.loc[plot_df["Country Name"] == "Malaysia", "CO2_per_capita"].mean()
        others = plot_df.loc[plot_df["Country Name"] != "Malaysia", "CO2_per_capita"]
        other_mean = others.mean() if not others.empty else malaysia_mean
        ratio = malaysia_mean / other_mean if other_mean else 1

        if ratio > 1.2:
            obs_text = (
                f"> **Observation:** Malaysia's per-capita CO₂ emissions remain **substantially higher** "
                f"than the average of the selected countries across {year_range[0]}–{year_range[1]}. "
                f"This indicates Malaysia's comparatively larger industrial and energy intensity."
            )
        elif ratio < 0.8:
            obs_text = (
                f"> **Observation:** Malaysia's per-capita CO₂ emissions are **consistently lower** "
                f"than most selected countries, suggesting a relatively smaller carbon footprint per person."
            )
        else:
            obs_text = (
                f"> **Observation:** Malaysia's per-capita CO₂ emissions are **comparable** "
                f"to the overall regional level, with fluctuations closely following ASEAN and world trends."
            )
    else:
        obs_text = (
            "> **Observation:** CO₂ emission trends vary notably among the selected countries; "
            "select Malaysia to view its relative position and long-term changes."
        )

    st.markdown(obs_text)



elif view.startswith("bar"):
    # --- 交互开关 ---
    show_diff = st.sidebar.toggle("Show difference vs World (Δ)", value=True)
    show_rank = st.sidebar.toggle("Show rank numbers", value=True)

    # --- 最近可用年份（受滑块上限约束） ---
    latest_year = min(int(df["Year"].max()), int(year_range[1]))

    # --- 抽取当年 ASEAN + World ---
    latest = df[
        (df["Year"] == latest_year) &
        (df["Country Name"].isin(ASEAN + ["World"]))
    ].copy()

    # 防御：如果该年缺 World，直接退回绝对值模式
    if "World" not in latest["Country Name"].values:
        show_diff = False

    # --- 计算差值 Δ vs World 或保留绝对值 ---
    if show_diff:
        world_val = latest.loc[latest["Country Name"]=="World", "CO2_per_capita"].iloc[0]
        latest["Delta_vs_World"] = latest["CO2_per_capita"] - world_val
        plot_col = "Delta_vs_World"
        x_label  = "Δ vs World (metric tons per capita)"
        # 排序：Δ 从大到小
        latest = latest.sort_values(plot_col, ascending=False)
    else:
        plot_col = "CO2_per_capita"
        x_label  = "CO$_2$ (metric tons per capita)"
        latest = latest.sort_values(plot_col, ascending=False)

    # --- 生成排名（用于可选展示）---
    latest["Rank"] = range(1, len(latest) + 1)

    # --- 调色：Malaysia 高亮，其它柔和灰蓝；差值模式下负值用灰 ---
    base_gray  = "#8B8FA7"
    dark_gray  = "#2B2F36"
    pos_color  = PRIMARY     # 正向/高值
    neg_color  = "#6b7280"   # 负向/低值（差值模式）

    colors = []
    for _, row in latest.iterrows():
        name = row["Country Name"]
        if name == "Malaysia":
            colors.append(pos_color)
        else:
            if show_diff:
                colors.append(pos_color if row[plot_col] > 0 else neg_color)
            else:
                colors.append(base_gray)

    # --- 画图 ---
    fig, ax = plt.subplots(figsize=(8.8, 5))
    sns.barplot(
        data=latest,
        x=plot_col,
        y="Country Name",
        palette=colors,
        ax=ax
    )

    # 差值模式：在 0 处画参考线
    if show_diff:
        ax.axvline(0, color=GRID, linewidth=1.2)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Country")
    title_mode = "Difference from World Average" if show_diff else "Absolute level"
    ax.set_title(f"ASEAN + World – {title_mode} • {latest_year}")

    # 数值标签
    for i, v in enumerate(latest[plot_col]):
        ax.text(v + (0.12 if v >= 0 else -0.12), i,
                f"{v:.1f}",
                va="center",
                ha="left" if v >= 0 else "right",
                color=TXT)

    # 排名标注（可选）：在每条最左侧画 #rank
    if show_rank:
        ymin, ymax = ax.get_ylim()
        for i, r in enumerate(latest["Rank"]):
            ax.text(ax.get_xlim()[0], i, f"#{r}",
                    va="center", ha="left",
                    color=TXT, alpha=0.85)

    apply_theme(ax)
    plt.tight_layout(pad=1.5)
    st.pyplot(fig)

    # --- Observation：针对 Malaysia 动态文案 ---
    # 取 Malaysia、World 的数值（若缺少则跳过）
    mal_val = latest.loc[latest["Country Name"]=="Malaysia", plot_col].iloc[0] \
              if "Malaysia" in latest["Country Name"].values else None

    if show_diff and mal_val is not None:
        # 差值模式：直接解释 Malaysia 高/低于世界多少
        higher_lower = "higher than" if mal_val >= 0 else "lower than"
        st.markdown(
            f"> **Observation:** In **{latest_year}**, Malaysia is **{abs(mal_val):.1f}** t higher/lower than the **World** average "
            f"({higher_lower}). Countries are ranked by Δ vs World to emphasize relative gaps."
            .replace("higher/lower", "higher" if mal_val >= 0 else "lower")
        )
    else:
        # 绝对值模式：解释 Malaysia 的排名
        if "Malaysia" in latest["Country Name"].values:
            mal_rank = int(latest.loc[latest["Country Name"]=="Malaysia", "Rank"].iloc[0])
            top = latest.iloc[0]
            st.markdown(
                f"> **Observation:** In **{latest_year}**, **Malaysia** ranks **#{mal_rank}** by per-capita CO$_2$. "
                f"**{top['Country Name']}** is highest in the region for this year."
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
