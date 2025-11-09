# app.py
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from io import StringIO
import re


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

def smart_read(uploaded_file):
    """
    Robust reader for CSV/Excel with World Bank wide-to-long reshape.

    - Excel: read directly
    - CSV: try autodetect delimiter; fallback to common seps; finally skip bad lines
    - If a World Bank-like wide table is detected (year columns like 1960..2023),
      reshape to long format with columns: Country Name, Year, CO2_per_capita.
    """
    name = uploaded_file.name.lower()

    # -------- helper: post-process to our canonical schema --------
    def postprocess_df(df: pd.DataFrame) -> pd.DataFrame:
        # 1) Try to detect wide year columns (e.g., '1960', '1961', ...)
        year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
        if year_cols and ("Country Name" in df.columns or "country" in [c.lower() for c in df.columns]):
            # Normalize id column names that might vary
            id_candidates = ["Country Name", "Country Code", "Indicator Name", "Indicator Code",
                             "Series Name", "Series Code"]
            id_vars = [c for c in id_candidates if c in df.columns]

            # Ensure we at least keep country name
            if "Country Name" not in id_vars:
                # try lower-case variants
                for c in df.columns:
                    if c.lower().strip() in ["country", "country name", "country_name"]:
                        df = df.rename(columns={c: "Country Name"})
                        id_vars = ["Country Name"] + [c for c in id_vars if c != "Country Name"]
                        break

            m = df.melt(id_vars=id_vars, value_vars=year_cols,
                        var_name="Year", value_name="CO2_per_capita")

            # If Indicator Code is present, keep CO2-per-capita series when available
            if "Indicator Code" in m.columns:
                mask = m["Indicator Code"].astype(str).str.contains("EN.GHG.CO2.PC", case=False, na=False)
                if mask.any():
                    m = m[mask]

            # Keep only what we need
            if "Country Name" not in m.columns:
                raise RuntimeError("Cannot find 'Country Name' after reshape.")

            m["Year"] = pd.to_numeric(m["Year"], errors="coerce")
            m["CO2_per_capita"] = pd.to_numeric(m["CO2_per_capita"], errors="coerce")
            m = m.dropna(subset=["Year", "CO2_per_capita"])
            m["Year"] = m["Year"].astype(int)

            return m[["Country Name", "Year", "CO2_per_capita"]]

        # 2) Already long format – try to normalize column names
        mapping = {}
        for c in df.columns:
            lc = c.lower().strip()
            if lc in ["country", "country name", "country_name"]:
                mapping[c] = "Country Name"
            elif lc == "year":
                mapping[c] = "Year"
            elif lc in ["co2_per_capita", "co2 per capita", "co2_pc", "value"]:
                mapping[c] = "CO2_per_capita"
        if mapping:
            df = df.rename(columns=mapping)

        # Numeric/coerce & trim if columns are available
        if {"Country Name", "Year", "CO2_per_capita"}.issubset(df.columns):
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
            df["CO2_per_capita"] = pd.to_numeric(df["CO2_per_capita"], errors="coerce")
            df = df.dropna(subset=["Year", "CO2_per_capita"])
            df["Year"] = df["Year"].astype(int)
        return df

    # -------- Excel: read directly --------
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
        return postprocess_df(df)

    # -------- CSV: read text then parse (handle preface/header rows) --------
    raw = uploaded_file.read().decode("utf-8", errors="ignore")

    # Find the real header row (World Bank files often have notes at the top)
    header_candidates = ["country", "country name", "year", "co2", "value", "indicator code"]
    lines = raw.splitlines()
    header_row = 0
    for i, ln in enumerate(lines[:50]):
        lower = ln.lower()
        if any(k in lower for k in header_candidates):
            header_row = i
            break
    trimmed = "\n".join(lines[header_row:])

    # Helper to reopen the text
    from io import StringIO
    def _io(text): return StringIO(text)

    # Try 1: autodetect sep
    try:
        df = pd.read_csv(_io(trimmed), sep=None, engine="python")
        return postprocess_df(df)
    except Exception:
        pass

    # Try 2: common separators
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(_io(trimmed), sep=sep)
            return postprocess_df(df)
        except Exception:
            continue

    # Try 3: tolerant mode (skip bad lines)
    df = pd.read_csv(_io(trimmed), sep=None, engine="python", on_bad_lines="skip")
    return postprocess_df(df)


def _hex_to_rgb(h):
    h = (h or "#FFFFFF").lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _luma(hex_color):
    r, g, b = _hex_to_rgb(hex_color)
    return (0.2126*r + 0.7152*g + 0.0722*b) / 255.0

GRID = "#E5E7EB" if _luma(BG) >= 0.5 else "#3A3F47"

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
    leg = ax.get_legend()
    if leg:
        if _luma(BG) >= 0.5:
            leg.get_frame().set_facecolor((1, 1, 1, 0.85))
        else:
            leg.get_frame().set_facecolor((0.1, 0.1, 0.1, 0.85))
        leg.get_frame().set_edgecolor(GRID)
        leg.get_frame().set_linewidth(1.0)
        if leg.get_title():
            leg.get_title().set_color(TXT)
        for t in leg.get_texts():
            t.set_color(TXT)

# ----------------------------
# 会话状态：原始/清洗数据 + 流水线
# ----------------------------
if "df_raw" not in st.session_state:   st.session_state.df_raw = None
if "df_clean" not in st.session_state: st.session_state.df_clean = None
if "pipeline" not in st.session_state: st.session_state.pipeline = []  # list of dict steps

# ----------------------------
# 默认数据（如果没上传就用你现有 CSV）
# ----------------------------
DEFAULT_PATH = "data/co2_clean_asean.csv"
def load_default():
    df = pd.read_csv(DEFAULT_PATH)
    df["Year"] = df["Year"].astype(int)
    df["CO2_per_capita"] = pd.to_numeric(df["CO2_per_capita"], errors="coerce")
    df = df.dropna(subset=["CO2_per_capita"])
    return df

# ----------------------------
# 数据域常量 & 可视化函数（复用你原逻辑）
# ----------------------------
ASEAN = [
    "Malaysia","Singapore","Thailand","Indonesia","Vietnam",
    "Philippines","Cambodia","Lao PDR","Myanmar","Brunei Darussalam"
]

def compute_combined(df_base: pd.DataFrame):
    asean_mean = (
        df_base[df_base["Country Name"].isin(ASEAN)]
        .groupby("Year")["CO2_per_capita"]
        .mean()
        .reset_index()
    )
    asean_mean["Country Name"] = "ASEAN Average"
    combined = pd.concat([df_base, asean_mean], ignore_index=True)
    return combined

def line_chart(combined, year_range):
    plot_df = combined[
        combined["Country Name"].isin(["Malaysia", "ASEAN Average", "World"])
        & (combined["Year"].between(*year_range))
    ]
    palette = {
        "Malaysia": PRIMARY,
        "ASEAN Average": "#9b8aff",
        "World": "#6fc3ff"
    }
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    sns.lineplot(
        data=plot_df, x="Year", y="CO2_per_capita",
        hue="Country Name", palette=palette, marker="o", linewidth=2.5, ax=ax
    )
    ax.set_ylim(0, 9)
    ax.set_xlabel("Year"); ax.set_ylabel("CO$_2$ (metric tons per capita)")
    ax.set_title("Malaysia vs ASEAN Average vs World")
    apply_theme(ax); plt.tight_layout(pad=1.5)
    st.pyplot(fig)

def bar_chart(df_base, year_range):
    latest_year = min(max(df_base["Year"]), year_range[1])
    latest = df_base[
        (df_base["Year"] == latest_year) & (df_base["Country Name"].isin(ASEAN + ["World"]))
    ].sort_values("CO2_per_capita", ascending=False)

    fig, ax = plt.subplots(figsize=(8.8, 5))
    cmap = LinearSegmentedColormap.from_list(
        "rank_grad", [PRIMARY, "#8B8FA7", "#2B2F36"], N=len(latest)
    )
    denom = max(1, len(latest) - 1)
    bar_palette = [cmap(i/denom) for i in range(len(latest))]

    sns.barplot(
        data=latest, x="CO2_per_capita", y="Country Name",
        palette=bar_palette, ax=ax
    )
    ax.set_xlabel("CO$_2$ (metric tons per capita)"); ax.set_ylabel("Country")
    ax.set_title(f"Latest Year Comparison – {latest_year}")
    for i, v in enumerate(latest["CO2_per_capita"]):
        ax.text(v + 0.1, i, f"{v:.1f}", va="center", color=TXT)
    apply_theme(ax); plt.tight_layout(pad=1.5)
    st.pyplot(fig)

def heatmap_chart(df_base, year_range):
    hm = df_base[df_base["Country Name"].isin(ASEAN + ["World"])]
    hm = hm[hm["Year"].between(*year_range)]
    pivot = hm.pivot_table(values="CO2_per_capita",
                           index="Country Name", columns="Year", aggfunc="mean")

    latest_year = min(max(df_base["Year"]), year_range[1])
    if latest_year in pivot.columns:
        order = pivot[latest_year].sort_values(ascending=False).index
        pivot = pivot.loc[order]

    vals = pivot.values.astype(float)
    vmin, vmax = np.nanpercentile(vals, [5, 95])

    if _luma(BG) < 0.5:
        cmap = LinearSegmentedColormap.from_list("heat_dark", [SEC, "#3b4252", PRIMARY], N=256)
    else:
        cmap = LinearSegmentedColormap.from_list("heat_light", ["#fff1fa", PRIMARY, "#4a154b"], N=256)

    fig, ax = plt.subplots(figsize=(12, 6))
    h = sns.heatmap(
        pivot, cmap=cmap, vmin=vmin, vmax=vmax,
        cbar_kws={"label": "CO$_2$ per capita"},
        ax=ax, linewidths=0.4, linecolor=GRID, square=False
    )
    cbar = h.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(TXT); cbar.ax.tick_params(colors=TXT)
    if cbar.outline: cbar.outline.set_edgecolor(GRID)

    years = list(pivot.columns)
    step = 2 if len(years) > 15 else 1
    ax.set_xticks(range(0, len(years), step))
    ax.set_xticklabels(years[::step], rotation=0)
    ax.set_xlabel("Year"); ax.set_ylabel("Country")
    ax.set_title("Heatmap of CO$_2$ per Capita (ASEAN & World)")

    for target, color, lw in [("Malaysia", PRIMARY, 2.0), ("World", "#1f77b4", 1.8)]:
        if target in pivot.index:
            i = list(pivot.index).index(target)
            ax.add_patch(plt.Rectangle((0, i), len(pivot.columns), 1, fill=False, lw=lw, edgecolor=color))

    apply_theme(ax)
    st.pyplot(fig)

# ----------------------------
# 顶部标题
# ----------------------------
st.title("CO₂ Emissions per Capita – Malaysia in ASEAN Context")
st.caption("Source: World Bank (EN.GHG.CO2.PC.CE.AR5) – Metric tons per capita")

# ----------------------------
# 页签
# ----------------------------
tab_upload, tab_clean, tab_viz, tab_report = st.tabs(
    ["Upload & Inspect", "Clean & Transform", "Visualise", "Report"]
)

# ----------------------------
# 1) Upload & Inspect
# ----------------------------
with tab_upload:
    st.subheader("Upload a dataset (CSV/Excel) or use the default ASEAN dataset")
    up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

    colA, colB = st.columns([1,1])
    with colA:
        if st.button("Use default ASEAN dataset"):
            st.session_state.df_raw = load_default()
            st.session_state.df_clean = None
            st.session_state.pipeline = [{"step":"load_default", "args": {}}]
            st.success("Loaded built-in dataset.")
    if up is not None:
        try:
            # 用更健壮的读取器
            df_raw = smart_read(up)
            st.session_state.df_raw = df_raw
            st.session_state.df_clean = None
            st.session_state.pipeline = [{"step":"upload", "args":{"filename": up.name}}]
            st.success(f"Uploaded: {up.name}  (rows={len(df_raw)}, cols={len(df_raw.columns)})")
        except Exception as e:
            st.error(f"Failed to read file: {e}")

    df_show = st.session_state.df_clean or st.session_state.df_raw
    if df_show is not None:
        st.markdown("**Preview (head)**")
        st.dataframe(df_show.head(10), use_container_width=True)

        st.markdown("**Schema / dtypes**")
        st.write(df_show.dtypes)

        st.markdown("**Missing values by column**")
        st.write(df_show.isna().sum())

        st.markdown("**Duplicate rows**")
        st.write(df_show.duplicated().sum())

# ----------------------------
# 2) Clean & Transform
# ----------------------------
with tab_clean:
    st.subheader("Step-by-step cleaning with preview and pipeline log")

    # 让用户先选模式（放在 Clean 页签内部）
    clean_mode = st.radio(
        "Cleaning Mode",
        ["Manual", "Auto"],
        captions=["Step-by-step manual cleaning", "Auto-run with default cleaning pipeline"],
        horizontal=True,
        key="clean_mode_radio",
    )

    # “只跑一次”的保护开关
    if "auto_done" not in st.session_state:
        st.session_state.auto_done = False

    # 有数据再处理
    base = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw

    # ------- Auto 模式：仅在切到 Auto 且还没执行过时触发 -------
    if clean_mode == "Auto":
        if st.session_state.df_raw is None and base is None:
            st.info("No data yet. Go to **Upload & Inspect** to load a dataset.")
        elif not st.session_state.auto_done:
            src = st.session_state.df_raw if st.session_state.df_raw is not None else base
            if src is None:
                st.warning("No data to clean. Please upload or load the default dataset in 'Upload & Inspect'.")
                st.stop()
            df = src.copy()


            # 1) 缺失值：数值列填均值
            df = df.fillna(df.mean(numeric_only=True))
            # 2) 去重
            df = df.drop_duplicates()
            # 3) 尝试把 object 转数字（失败忽略）
            for col in df.columns:
                if df[col].dtype == "object":
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except Exception:
                        pass

            st.session_state.df_clean = df
            st.session_state.pipeline.append({
                "step": "auto_clean",
                "args": {"methods": ["fill mean", "drop duplicates", "convert numeric"]}
            })
            st.session_state.auto_done = True
            st.success("Auto cleaning completed.")
        else:
            st.info("Auto cleaning already applied. Switch to Manual if you want step-by-step actions.")

    # 如果切回 Manual，允许再次手动编辑，并可重置 auto_done
    else:
        st.session_state.auto_done = False

    # ------- Manual 工具区 -------
    base = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw
    if base is None:
        st.info("No data yet. Go to **Upload & Inspect** to load a dataset.")
        st.stop()

    df_work = base.copy()
    st.markdown("### Actions")

    act = st.selectbox(
        "Choose an action",
        ["(select)", "Handle missing values", "Remove duplicates", "Cast to numeric", "Min-Max scale"]
    )

    if act == "Handle missing values":
        cols = st.multiselect("Columns to impute", df_work.columns.tolist())
        how = st.radio(
            "Method",
            ["drop rows", "fill mean", "fill median", "fill mode", "forward fill", "backward fill"],
            horizontal=True
        )
        if st.button("Apply", key="apply_impute"):
            before = len(df_work)
            if how == "drop rows":
                df_work = df_work.dropna(subset=cols) if cols else df_work.dropna()
            elif how == "fill mean":
                for c in cols: df_work[c] = df_work[c].fillna(df_work[c].mean())
            elif how == "fill median":
                for c in cols: df_work[c] = df_work[c].fillna(df_work[c].median())
            elif how == "fill mode":
                for c in cols:
                    mode = df_work[c].mode()
                    if not mode.empty: df_work[c] = df_work[c].fillna(mode[0])
            elif how == "forward fill":
                df_work[cols] = df_work[cols].ffill()
            elif how == "backward fill":
                df_work[cols] = df_work[cols].bfill()

            st.session_state.df_clean = df_work
            st.session_state.pipeline.append({"step": "impute", "args": {"cols": cols, "method": how}})
            st.success(f"Applied: {how}. Rows before/after: {before} → {len(df_work)}")

    elif act == "Remove duplicates":
        keep = st.selectbox("Keep", ["first", "last", "False (drop all dup)"])
        keep_arg = {"first": "first", "last": "last", "False (drop all dup)": False}[keep]
        if st.button("Apply", key="apply_dropdup"):
            before = len(df_work)
            df_work = df_work.drop_duplicates(keep=keep_arg)
            st.session_state.df_clean = df_work
            st.session_state.pipeline.append({"step": "drop_duplicates", "args": {"keep": keep_arg}})
            st.success(f"Duplicates removed. Rows: {before} → {len(df_work)}")

    elif act == "Cast to numeric":
        cols = st.multiselect("Columns to cast (errors→NaN)", df_work.columns.tolist())
        if st.button("Apply", key="apply_cast"):
            for c in cols:
                df_work[c] = pd.to_numeric(df_work[c], errors="coerce")
            st.session_state.df_clean = df_work
            st.session_state.pipeline.append({"step": "to_numeric", "args": {"cols": cols}})
            st.success("Cast done.")

    elif act == "Min-Max scale":
        cols = st.multiselect("Columns to scale (0-1)", df_work.select_dtypes(include=np.number).columns.tolist())
        if st.button("Apply", key="apply_scale"):
            for c in cols:
                lo, hi = df_work[c].min(), df_work[c].max()
                if pd.notna(lo) and pd.notna(hi) and hi != lo:
                    df_work[c] = (df_work[c] - lo) / (hi - lo)
            st.session_state.df_clean = df_work
            st.session_state.pipeline.append({"step": "minmax", "args": {"cols": cols}})
            st.success("Scaled.")

    # ------- 导出 -------
    st.markdown("### Export cleaned data")
    export_df = (
        st.session_state.df_clean
        if (st.session_state.df_clean is not None and not st.session_state.df_clean.empty)
        else st.session_state.df_raw
    )
    if export_df is not None:
        csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download cleaned.csv", csv, file_name="cleaned.csv", mime="text/csv")

    # ------- 流水线 -------
    st.markdown("### Pipeline log")
    st.json(st.session_state.pipeline)


# ----------------------------
# 3) Visualise（复用你的图）
# ----------------------------
with tab_viz:
    st.subheader("Visualisations")

    # 这里选择使用哪个 DataFrame 来画图：优先用清洗后的
    if st.session_state.df_clean is not None and not st.session_state.df_clean.empty:
        df_base = st.session_state.df_clean
    elif st.session_state.df_raw is not None and not st.session_state.df_raw.empty:
        df_base = st.session_state.df_raw
    else:
        df_base = load_default()


    # 保障核心列存在
    required_cols = {"Year", "Country Name", "CO2_per_capita"}
    if not required_cols.issubset(set(df_base.columns)):
        st.error("The selected dataset must contain columns: Year, Country Name, CO2_per_capita")
    else:
        # 年份范围
        ymin, ymax = int(df_base["Year"].min()), int(df_base["Year"].max())
        year_range = st.slider("Year range", min_value=ymin, max_value=ymax, value=(max(1990, ymin), ymax), step=1)

        # 选择视图
        labels = {
            "line": "**Line:**  Malaysia vs ASEAN vs World",
            "bar":  "**Bar:**   Latest Year (ASEAN + World)",
            "heat": "**Heatmap:** ASEAN & World (1990–2023)",
        }
        view = st.radio("Visualisation", options=["line","bar","heat"], format_func=lambda k: labels[k], horizontal=True)

        # 计算 combined（含 ASEAN Average）
        combined = compute_combined(df_base)

        if view == "line":
            line_chart(combined, year_range)
            st.markdown(
                "**Observation (Malaysia-focused):** Since the late 1990s, Malaysia’s per-capita CO$_2$ has stayed above the world average; "
                "the gap with ASEAN narrowed after the 2010s as regional averages rose and Malaysia stabilized."
            )
        elif view == "bar":
            bar_chart(df_base, year_range)
            latest_year_bar = min(max(df_base["Year"]), year_range[1])
            st.markdown(
                f"**Observation:** In **{latest_year_bar}**, Malaysia ranks in the upper tier of ASEAN; Brunei / Singapore remain notably higher, "
                "while Cambodia / Myanmar are much lower; World is shown as a baseline."
            )
        else:
            heatmap_chart(df_base, year_range)
            st.markdown(
                "**Observation:** Heatmap is sorted by the most recent year to highlight relative levels; "
                "Malaysia rose in the 2000s and has stabilized in recent years."
            )

# ----------------------------
# 4) Report（自动方法摘要）
# ----------------------------
with tab_report:
    st.subheader("Methods summary (auto-generated)")
    if len(st.session_state.pipeline)==0:
        st.info("No pipeline steps yet. Perform some cleaning in the **Clean & Transform** tab.")
    else:
        # 生成简单 methods 文本
        lines = ["# Data Processing Methods",
                 "",
                 "This dataset was processed inside the system as follows:"]
        for i, step in enumerate(st.session_state.pipeline, start=1):
            lines.append(f"{i}. **{step['step']}** — params: `{step['args']}`")
        md = "\n".join(lines)
        st.markdown(md)

        # 导出 markdown
        st.download_button(
            "Download methods.md",
            md.encode("utf-8"),
            file_name="methods.md",
            mime="text/markdown"
        )

st.divider()
st.caption("Tip: Upload → Clean → Visualise → Report. The pipeline is reproducible and exportable.")
