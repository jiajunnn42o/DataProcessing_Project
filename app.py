# app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# ------------------------------------------
# 页面设置
# ------------------------------------------
st.set_page_config(page_title="CO₂ Visualisation System",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------------------------
# 🌙 永久深色主题（不跟随系统）
# ------------------------------------------
PAGE_BG = "#0E1117"     # 主背景
SIDEBAR_BG = "#1A1D21"  # 侧栏背景
TEXT_COLOR = "#FAFAFA"  # 白字
GRID_COLOR = "#444444"  # 网格线
CHART_BG = PAGE_BG      # 图表背景

# === 根据 Streamlit 主题自动调整页面颜色 ===
base_mode = st.get_option("theme.base") or "light"
IS_LIGHT = base_mode.lower() == "light"

PAGE_BG = "#FFFFFF" if IS_LIGHT else "#0E1117"
SIDEBAR_BG = "#F7F7F7" if IS_LIGHT else "#1A1D21"
TEXT_COLOR = "#262730" if IS_LIGHT else "#FAFAFA"
GRID_COLOR = "#DDDDDD" if IS_LIGHT else "#444444"
CHART_BG = PAGE_BG

# === 计算是否为浅色/深色（来自 Streamlit 主题）===
base_mode = (st.get_option("theme.base") or "light").lower()
IS_LIGHT = base_mode == "light"

PAGE_BG    = "#FFFFFF" if IS_LIGHT else "#0E1117"
SIDEBAR_BG = "#F7F7F7" if IS_LIGHT else "#1A1D21"
TEXT_COLOR = "#262730" if IS_LIGHT else "#FAFAFA"
GRID_COLOR = "#DDDDDD" if IS_LIGHT else "#444444"

# === 覆盖页面、侧栏、顶部栏与下拉菜单的样式 ===
st.markdown(f"""
<style>
/* 整个 App */
.stApp {{
  background-color: {PAGE_BG};
  color: {TEXT_COLOR};
}}

/* 侧栏 */
[data-testid="stSidebar"] {{
  background-color: {SIDEBAR_BG};
  color: {TEXT_COLOR};
}}

/* 顶部栏（包含 Deploy）*/
[data-testid="stAppHeader"] {{
  background-color: {SIDEBAR_BG} !important;
  color: {TEXT_COLOR} !important;
  border-bottom: 1px solid {GRID_COLOR};
}}
/* 顶部栏内所有文字与图标（含 Deploy/设置图标） */
[data-testid="stAppHeader"] * {{
  color: {TEXT_COLOR} !important;
  fill: {TEXT_COLOR} !important;
}}

/* 顶部的下拉菜单（点击 … 打开的菜单） */
.stApp [role="menu"] {{
  background: {SIDEBAR_BG} !important;
  color: {TEXT_COLOR} !important;
  border: 1px solid {GRID_COLOR};
  box-shadow: none !important;
}}
.stApp [role="menu"] * {{
  color: {TEXT_COLOR} !important;
  fill: {TEXT_COLOR} !important;
}}

/* 统一正文文字颜色（防止局部被主题覆盖） */
h1, h2, h3, h4, h5, h6,
p, div, span, label,
.stMarkdown, .stText, .stCaption {{
  color: {TEXT_COLOR} !important;
}}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------
# Matplotlib / Seaborn 样式
# ------------------------------------------
mpl.rcdefaults()
sns.reset_orig()
mpl.rcParams.update({
    "figure.facecolor": CHART_BG,
    "axes.facecolor": CHART_BG,
    "savefig.transparent": True,
    "text.color": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.edgecolor": GRID_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "axes.titlecolor": TEXT_COLOR,
    "grid.color": GRID_COLOR,
    "legend.facecolor": "none",
    "legend.edgecolor": GRID_COLOR,
    "legend.labelcolor": TEXT_COLOR,
})
sns.set_style("whitegrid", {"axes.facecolor": CHART_BG, "grid.color": GRID_COLOR})

def apply_theme(ax):
    ax.title.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.tick_params(axis="both", colors=TEXT_COLOR)
    ax.grid(True, color=GRID_COLOR, alpha=0.3)
    for sp in ax.spines.values():
        sp.set_color(GRID_COLOR)

def polish_legend(ax):
    leg = ax.get_legend()
    if leg:
        leg.get_frame().set_facecolor((1, 1, 1, 0))
        leg.get_frame().set_edgecolor(GRID_COLOR)
        if leg.get_title():
            leg.get_title().set_color(TEXT_COLOR)
        for t in leg.get_texts():
            t.set_color(TEXT_COLOR)

# ------------------------------------------
# 数据加载
# ------------------------------------------
DATA_PATH = "data/co2_clean_asean.csv"
df = pd.read_csv(DATA_PATH)
df["Year"] = df["Year"].astype(int)
df["CO2_per_capita"] = pd.to_numeric(df["CO2_per_capita"], errors="coerce")
df = df.dropna(subset=["CO2_per_capita"])

ASEAN = [
    "Malaysia","Singapore","Thailand","Indonesia","Vietnam",
    "Philippines","Cambodia","Lao PDR","Myanmar","Brunei Darussalam"
]

# ASEAN 平均
asean_mean = (
    df[df["Country Name"].isin(ASEAN)]
    .groupby("Year")["CO2_per_capita"].mean().reset_index()
)
asean_mean["Country Name"] = "ASEAN Average"
combined = pd.concat([df, asean_mean], ignore_index=True)

# ------------------------------------------
# 控制面板
# ------------------------------------------
st.sidebar.header("Controls")
view = st.sidebar.radio("Visualisation", [
    "Line: Malaysia vs ASEAN vs World",
    "Bar: Latest Year (ASEAN + World)",
    "Heatmap: ASEAN & World (1990–2023)"
])
year_min, year_max = 1990, 2023
year_range = st.sidebar.slider(
    "Year range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max),
    step=1
)


# ------------------------------------------
# 主标题
# ------------------------------------------
st.title("CO₂ Emissions per Capita – Malaysia in ASEAN Context")
st.caption("Source: World Bank (EN.GHG.CO2.PC.CE.AR5) – Metric tons per capita")

# ------------------------------------------
# 折线图
# ------------------------------------------
if view.startswith("Line"):
    plot_df = combined[
        combined["Country Name"].isin(["Malaysia", "ASEAN Average", "World"])
        & (combined["Year"].between(*year_range))
    ]
    palette = {'Malaysia':'#d62728', 'ASEAN Average':'#2ca02c', 'World':'#1f77b4'}

    fig, ax = plt.subplots(figsize=(11,6), facecolor=CHART_BG)
    sns.lineplot(data=plot_df, x="Year", y="CO2_per_capita",
                 hue="Country Name", palette=palette,
                 marker="o", linewidth=2.5, ax=ax)
    ax.set_ylim(0, 9)
    ax.set_xlabel("Year"); ax.set_ylabel("CO₂ (metric tons per capita)")
    ax.set_title("Malaysia vs ASEAN Average vs World")
    apply_theme(ax)
    polish_legend(ax)
    st.pyplot(fig)

    st.markdown(
        "> 观察：自 1990s 后期起，**Malaysia** 的人均排放长期高于 **World** 平均值，"
        "且与 **ASEAN Average** 的差距在 2000s 扩大、近几年逐步收敛。"
    )

# ------------------------------------------
# 条形图
# ------------------------------------------
elif view.startswith("Bar"):
    latest_year = min(max(df["Year"]), year_range[1])
    latest = df[(df["Year"] == latest_year) & (df["Country Name"].isin(ASEAN + ["World"]))].copy()
    latest = latest.sort_values("CO2_per_capita", ascending=False)

    fig, ax = plt.subplots(figsize=(10,6), facecolor=CHART_BG)
    sns.barplot(data=latest, x="CO2_per_capita", y="Country Name", palette="YlOrRd", ax=ax)
    ax.set_xlabel("CO₂ (metric tons per capita)"); ax.set_ylabel("Country")
    ax.set_title(f"Latest Year Comparison – {latest_year}")
    for i, v in enumerate(latest["CO2_per_capita"]):
        ax.text(v + 0.1, i, f"{v:.1f}", va="center", color=TEXT_COLOR)
    apply_theme(ax)
    st.pyplot(fig)

    st.markdown(
        f"> 观察：在 **{latest_year}** 年，**Malaysia** 位于东盟上游；"
        "**Brunei/SG** 显著更高，**Cambodia/Myanmar** 较低；"
        "**World** 作为基线参考。"
    )

# ------------------------------------------
# 热力图
# ------------------------------------------
else:
    hm = df[df["Country Name"].isin(ASEAN + ["World"])]
    hm = hm[hm["Year"].between(*year_range)]
    pivot = hm.pivot_table(values="CO2_per_capita", index="Country Name", columns="Year", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(12,6), facecolor=CHART_BG)
    h = sns.heatmap(pivot, cmap="YlOrRd", cbar_kws={"label": "CO₂ per capita"}, ax=ax)
    cbar = h.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(TEXT_COLOR)
    cbar.ax.tick_params(colors=TEXT_COLOR)
    ax.set_title("Heatmap of CO₂ per Capita (ASEAN & World)")
    ax.set_xlabel("Year"); ax.set_ylabel("Country")

    for target, color, lw in [("Malaysia","black",2.2), ("World","#1f77b4",2.0)]:
        if target in pivot.index:
            i = list(pivot.index).index(target)
            ax.add_patch(plt.Rectangle((0,i), len(pivot.columns), 1, fill=False, lw=lw, edgecolor=color))
    apply_theme(ax)
    st.pyplot(fig)

    st.markdown(
        "> 观察：热力图显示国家 **横向** 的绝对水平差异；"
        "Malaysia 在 2000s 出现显著上升，近年趋稳。"
    )

st.divider()
st.caption("Tip: Use the sidebar to adjust year range or switch visualisations.")
