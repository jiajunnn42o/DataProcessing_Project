# app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="CO₂ Visualisation (Malaysia & ASEAN)", page_icon="🌏", layout="wide")
sns.set_style("whitegrid")

# -----------------------------
# Load data
# -----------------------------
DATA_PATH = "data/co2_clean_asean.csv"
df = pd.read_csv(DATA_PATH)

# 基本清洗（双保险）
df["Year"] = df["Year"].astype(int)
df["CO2_per_capita"] = pd.to_numeric(df["CO2_per_capita"], errors="coerce")
df = df.dropna(subset=["CO2_per_capita"])

ASEAN = [
    "Malaysia","Singapore","Thailand","Indonesia","Vietnam",
    "Philippines","Cambodia","Lao PDR","Myanmar","Brunei Darussalam"
]

# 计算 ASEAN 平均（用于折线图对比）
asean_mean = (
    df[df["Country Name"].isin(ASEAN)]
    .groupby("Year")["CO2_per_capita"]
    .mean()
    .reset_index()
)
asean_mean["Country Name"] = "ASEAN Average"
combined = pd.concat([df, asean_mean], ignore_index=True)

# -----------------------------
# Sidebar (controls)
# -----------------------------
st.sidebar.header("Controls")
view = st.sidebar.radio("Visualisation", [
    "Line: Malaysia vs ASEAN vs World",
    "Bar: Latest Year (ASEAN + World)",
    "Heatmap: ASEAN & World (1990–2023)"
])

year_min, year_max = 1990, 2023
year_range = st.sidebar.slider("Year range", min_value=year_min, max_value=year_max, value=(year_min, year_max))

# -----------------------------
# Header
# -----------------------------
st.title("🌏 CO₂ Emissions per Capita – Malaysia in ASEAN Context")
st.caption("Source: World Bank (EN.GHG.CO2.PC.CE.AR5) | Metric tons per capita")

# -----------------------------
# Visualisations
# -----------------------------
if view.startswith("Line"):
    # 只取马来西亚、东盟均值、世界
    plot_df = combined[
        combined["Country Name"].isin(["Malaysia","ASEAN Average","World"])
        & (combined["Year"].between(*year_range))
    ].copy()

    palette = {'Malaysia':'#d62728', 'ASEAN Average':'#2ca02c', 'World':'#1f77b4'}
    fig, ax = plt.subplots(figsize=(11,6))
    sns.lineplot(data=plot_df, x="Year", y="CO2_per_capita", hue="Country Name",
                 marker="o", linewidth=2.5, palette=palette, ax=ax)
    ax.set_ylim(0, 9)
    ax.set_xlabel("Year"); ax.set_ylabel("CO₂ (metric tons per capita)")
    ax.set_title("Malaysia vs ASEAN Average vs World")
    st.pyplot(fig)

    st.markdown(
        "> 观察：自 1990s 后期起，**Malaysia** 的人均排放长期高于 **World** 平均值，"
        "且与 **ASEAN Average** 的差距在 2000s 扩大、近几年逐步收敛。"
    )

elif view.startswith("Bar"):
    latest_year = min(max(df["Year"]), year_range[1])  # 与滑块上限同步
    latest = df[(df["Year"] == latest_year) & (df["Country Name"].isin(ASEAN + ["World"]))].copy()
    latest = latest.sort_values("CO2_per_capita", ascending=False)

    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(data=latest, x="CO2_per_capita", y="Country Name", palette="YlOrRd", ax=ax)
    ax.set_xlabel("CO₂ (metric tons per capita)"); ax.set_ylabel("Country")
    ax.set_title(f"Latest Year Comparison – {latest_year}")
    # 数值标注
    for i, v in enumerate(latest["CO2_per_capita"]):
        ax.text(v + 0.1, i, f"{v:.1f}", va="center")
    st.pyplot(fig)

    st.markdown(
        f"> 观察：在 **{latest_year}** 年，**Malaysia** 位于东盟上游；**Brunei/SG** 仍显著更高，"
        "而 **Cambodia/Myanmar** 明显较低；**World** 作为基线参考。"
    )

else:  # Heatmap
    hm = df[df["Country Name"].isin(ASEAN + ["World"])]
    hm = hm[hm["Year"].between(*year_range)]
    pivot = hm.pivot_table(values="CO2_per_capita", index="Country Name", columns="Year", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(pivot, cmap="YlOrRd", cbar_kws={"label":"CO₂ per capita"}, ax=ax)
    ax.set_title("Heatmap of CO₂ per Capita (ASEAN & World)")
    ax.set_xlabel("Year"); ax.set_ylabel("Country")

    # 高亮 Malaysia & World 外框
    for target, color, lw in [("Malaysia", "black", 2.2), ("World", "#1f77b4", 2.0)]:
        if target in pivot.index:
            i = list(pivot.index).index(target)
            ax.add_patch(plt.Rectangle((0, i), len(pivot.columns), 1, fill=False, lw=lw, edgecolor=color))
    st.pyplot(fig)

    st.markdown(
        "> 观察：热力图显示国家 **横向** 的绝对水平差异；Malaysia 在 2000s 出现显著上升，近年趋稳。"
    )

st.divider()
st.caption("Tip: Use the sidebar to adjust year range or switch visualisations.")
