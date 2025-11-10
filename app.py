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

def _hex_to_rgb(h):
    h = (h or "#FFFFFF").lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _luma(hex_color):
    r, g, b = _hex_to_rgb(hex_color)
    return (0.2126*r + 0.7152*g + 0.0722*b) / 255.0

GRID = "#E5E7EB" if _luma(BG) >= 0.5 else "#3A3F47"

# Matplotlib / Seaborn 同步 theme
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
# ---- Brand helpers (GLOBAL) ----
IS_DARK = (_luma(BG) < 0.5)

def brand_palette(n=6):
    # 你的粉紫主题调色
    base = [PRIMARY, "#9b8aff", "#6f6f7a", "#d9b3ff", "#b38dff", "#ff99f3"]
    return base[:max(1, n)]

def new_fig(w=7.2, h=4.2):
    # 统一较小图幅，避免霸屏
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax

# （可选）全局设定 seaborn 默认调色为粉紫系
sns.set_palette(brand_palette(6))

# ----------------------------
# 智能读取器：CSV/Excel + 世行宽表 reshape
# ----------------------------
def smart_read(uploaded_file):
    """
    Robust reader for CSV/Excel with World Bank wide-to-long reshape.

    - Excel: read directly
    - CSV: autodetect delimiter; fallback to common seps; skip bad lines
    - If a World Bank-like wide table is detected (year columns like 1960..2023),
      reshape to long with: Country Name, Year, CO2_per_capita
    """
    name = uploaded_file.name.lower()

    def postprocess_df(df: pd.DataFrame) -> pd.DataFrame:
        # 1) 识别宽表的年份列
        year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
        if year_cols and ("Country Name" in df.columns or "country" in [c.lower() for c in df.columns]):
            id_candidates = ["Country Name", "Country Code", "Indicator Name", "Indicator Code",
                             "Series Name", "Series Code"]
            id_vars = [c for c in id_candidates if c in df.columns]

            if "Country Name" not in id_vars:
                for c in df.columns:
                    if str(c).lower().strip() in ["country", "country name", "country_name"]:
                        df = df.rename(columns={c: "Country Name"})
                        id_vars = ["Country Name"] + [x for x in id_vars if x != "Country Name"]
                        break

            m = df.melt(id_vars=id_vars, value_vars=year_cols,
                        var_name="Year", value_name="CO2_per_capita")

            if "Indicator Code" in m.columns:
                mask = m["Indicator Code"].astype(str).str.contains("EN.GHG.CO2.PC", case=False, na=False)
                if mask.any():
                    m = m[mask]

            if "Country Name" not in m.columns:
                raise RuntimeError("Cannot find 'Country Name' after reshape.")

            m["Year"] = pd.to_numeric(m["Year"], errors="coerce")
            m["CO2_per_capita"] = pd.to_numeric(m["CO2_per_capita"], errors="coerce")
            m = m.dropna(subset=["Year", "CO2_per_capita"])
            m["Year"] = m["Year"].astype(int)

            return m[["Country Name", "Year", "CO2_per_capita"]]

        # 2) 已是长表：尝试统一列名
        mapping = {}
        for c in df.columns:
            lc = str(c).lower().strip()
            if lc in ["country", "country name", "country_name"]:
                mapping[c] = "Country Name"
            elif lc == "year":
                mapping[c] = "Year"
            elif lc in ["co2_per_capita", "co2 per capita", "co2_pc", "value"]:
                mapping[c] = "CO2_per_capita"
        if mapping:
            df = df.rename(columns=mapping)

        # 数值化 & 去缺
        if {"Country Name", "Year", "CO2_per_capita"}.issubset(df.columns):
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
            df["CO2_per_capita"] = pd.to_numeric(df["CO2_per_capita"], errors="coerce")
            df = df.dropna(subset=["Year", "CO2_per_capita"])
            df["Year"] = df["Year"].astype(int)
        return df

    # Excel
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
        return postprocess_df(df)

    # CSV：剪掉前言行，自动识别表头
    raw = uploaded_file.read().decode("utf-8", errors="ignore")
    header_candidates = ["country", "country name", "year", "co2", "value", "indicator code"]
    lines = raw.splitlines()
    header_row = 0
    for i, ln in enumerate(lines[:50]):
        lower = ln.lower()
        if any(k in lower for k in header_candidates):
            header_row = i
            break
    trimmed = "\n".join(lines[header_row:])

    def _io(text): return StringIO(text)

    try:
        df = pd.read_csv(_io(trimmed), sep=None, engine="python")
        return postprocess_df(df)
    except Exception:
        pass
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(_io(trimmed), sep=sep)
            return postprocess_df(df)
        except Exception:
            continue
    df = pd.read_csv(_io(trimmed), sep=None, engine="python", on_bad_lines="skip")
    return postprocess_df(df)

# ----------------------------
# 会话状态
# ----------------------------
if "df_raw" not in st.session_state:    st.session_state.df_raw = None
if "df_clean" not in st.session_state:  st.session_state.df_clean = None
if "pipeline" not in st.session_state:  st.session_state.pipeline = []
if "generic_view" not in st.session_state: st.session_state.generic_view = None
if "last_upload_name" not in st.session_state: st.session_state.last_upload_name = None  # NEW

# ----------------------------
# 默认数据（你的 CO₂ CSV）
# ----------------------------
DEFAULT_PATH = "data/co2_clean_asean.csv"
def load_default():
    df = pd.read_csv(DEFAULT_PATH)
    df["Year"] = df["Year"].astype(int)
    df["CO2_per_capita"] = pd.to_numeric(df["CO2_per_capita"], errors="coerce")
    df = df.dropna(subset=["CO2_per_capita"])
    return df

# ----------------------------
# CO₂ 专用可视化工具
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
            st.session_state.last_upload_name = "__DEFAULT__"  # NEW
            st.success("Loaded built-in dataset.")
    # Only process when a NEW file is chosen
    if up is not None and st.session_state.last_upload_name != up.name:  # NEW
        try:
            df_raw = smart_read(up)
            st.session_state.df_raw = df_raw
            st.session_state.df_clean = None
            st.session_state.pipeline = [{"step":"upload", "args":{"filename": up.name}}]
            st.session_state.last_upload_name = up.name  # NEW
            st.success(f"Uploaded: {up.name}  (rows={len(df_raw)}, cols={len(df_raw.columns)})")
        except Exception as e:
            st.error(f"Failed to read file: {e}")

    df_show = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw
    if df_show is not None:
        st.markdown("**Preview (head)**")
        st.dataframe(df_show.head(10), use_container_width=True)

        st.markdown("**Schema / dtypes**")
        st.write(df_show.dtypes)

        st.markdown("**Missing values by column**")
        st.write(df_show.isna().sum())

        st.markdown("**Duplicate rows**")
        st.write(df_show.duplicated().sum())

        # ====== 通用：列映射器（当没有 CO₂ 专用列时使用）======
        required = {"Year", "Country Name", "CO2_per_capita"}
        if not required.issubset(df_show.columns):
            st.markdown("### Column Mapper（Universal Data Adaptation）")
            st.info("The current data does not contain column names specific to CO₂. You can map your own columns to universal roles (time/grouping/values) for universal visualization.")

            cols = df_show.columns.tolist()

            # 自动猜测时间列
            time_candidates = [c for c in cols if np.issubdtype(df_show[c].dtype, np.datetime64)]
            if not time_candidates:
                for c in cols:
                    if df_show[c].dtype == "object":
                        try:
                            pd.to_datetime(df_show[c], errors="raise", infer_datetime_format=True)
                            time_candidates.append(c)
                        except Exception:
                            pass

            col_time = st.selectbox("Time/Year（Optional）", ["(none)"] + time_candidates + [c for c in cols if c not in time_candidates])
            col_cat  = st.selectbox("Category/Group（Optional）", ["(none)"] + cols)
            # 只列出可数值化的列
            numeric_like = []
            for c in cols:
                if pd.api.types.is_numeric_dtype(df_show[c]):
                    numeric_like.append(c)
                else:
                    try:
                        pd.to_numeric(df_show[c], errors="raise")
                        numeric_like.append(c)
                    except Exception:
                        pass
            col_val  = st.selectbox("Value（Numeric column(s), required）", numeric_like if numeric_like else ["(none)"])

            if st.button("Create mapped view"):
                mapped = df_show.copy()
                rename = {}
                if col_time != "(none)": rename[col_time] = "Year"
                if col_cat  != "(none)": rename[col_cat]  = "Category"
                if col_val  != "(none)": rename[col_val]  = "Value"
                mapped = mapped.rename(columns=rename)

                # 解析 Year
                if "Year" in mapped.columns:
                    try:
                        mapped["Year"] = pd.to_datetime(mapped["Year"], errors="coerce", infer_datetime_format=True)
                    except Exception:
                        pass

                # 数值化 Value
                if "Value" in mapped.columns:
                    mapped["Value"] = pd.to_numeric(mapped["Value"], errors="coerce")

                st.session_state.generic_view = mapped
                st.success("A universal mapping view has been generated. Use it in the 'Generic Visualise' section on the Visualise page.")

# ----------------------------
# 2) Clean & Transform
# ----------------------------
with tab_clean:
    st.subheader("Step-by-step cleaning with preview and pipeline log")

    # 模式选择（放在 Clean 页里）
    clean_mode = st.radio(
        "Cleaning Mode",
        ["Manual", "Auto"],
        captions=["Step-by-step manual cleaning", "Auto-run with default cleaning pipeline"],
        horizontal=True,
        key="clean_mode_radio",
    )

    # “只跑一次”的保护
    if "auto_done" not in st.session_state:
        st.session_state.auto_done = False

    # 基础数据
    base = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw

    # Auto：仅首次切到 Auto 时执行
    if clean_mode == "Auto":
        if st.session_state.df_raw is None and base is None:
            st.info("No data yet. Go to **Upload & Inspect** to load a dataset.")
        elif not st.session_state.auto_done:
            src = st.session_state.df_raw if st.session_state.df_raw is not None else base
            if src is None:
                st.warning("No data to clean. Please upload or load the default dataset in 'Upload & Inspect'.")
                st.stop()
            df = src.copy()
            # 1) 缺失值（数值列填均值）
            df = df.fillna(df.mean(numeric_only=True))
            # 2) 去重
            df = df.drop_duplicates()
            # 3) object → numeric（失败忽略）
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
            st.info("Auto cleaning already applied. Switch to Manual for step-by-step actions.")
    else:
        st.session_state.auto_done = False

    # Manual 工具区
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

    # 导出
    st.markdown("### Export cleaned data")
    export_df = (
        st.session_state.df_clean
        if (st.session_state.df_clean is not None and not st.session_state.df_clean.empty)
        else st.session_state.df_raw
    )
    if export_df is not None:
        csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download cleaned.csv", csv, file_name="cleaned.csv", mime="text/csv")

    # 流水线
    st.markdown("### Pipeline log")
    st.json(st.session_state.pipeline)

# ----------------------------
# 3) Visualise
# ----------------------------
with tab_viz:
    st.subheader("Visualisations")

    # 选择用于绘图的数据源：优先 clean → raw → default
    if st.session_state.df_clean is not None and not st.session_state.df_clean.empty:
        df_base = st.session_state.df_clean
    elif st.session_state.df_raw is not None and not st.session_state.df_raw.empty:
        df_base = st.session_state.df_raw
    else:
        df_base = load_default()

    # CO₂ 专用列是否存在
    required_cols = {"Year", "Country Name", "CO2_per_capita"}
    has_co2_view = required_cols.issubset(df_base.columns)

    if not has_co2_view:
        st.info("No CO₂-specific columns were detected. You can still use the **Generic Visualise** feature below for arbitrary data visualization, or first use the **Column Mapper** on the Upload page for column mapping.")

    # 仅当具备 CO₂ 列时，展示你的三张图
    if has_co2_view:
        ymin, ymax = int(df_base["Year"].min()), int(df_base["Year"].max())
        year_range = st.slider("Year range", min_value=ymin, max_value=ymax, value=(max(1990, ymin), ymax), step=1)

        labels = {
            "line": "**Line:**  Malaysia vs ASEAN vs World",
            "bar":  "**Bar:**   Latest Year (ASEAN + World)",
            "heat": "**Heatmap:** ASEAN & World (1990–2023)",
        }
        view = st.radio("Visualisation", options=["line","bar","heat"], format_func=lambda k: labels[k], horizontal=True)

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

    # ====== 通用可视化（任何数据）======
st.markdown("---")
with st.expander("Generic Visualise (any dataset)"):
    gv = st.session_state.get("generic_view", None)
    use_generic = st.toggle(
        "Priority is given to using the view generated by Column Mapper (if available).",
        value=True if gv is not None else False
    )
    plot_df = gv if (use_generic and gv is not None) else df_base.copy()

    # 自动把名为 Year 或含 year 的列转成 datetime 类型
    for c in plot_df.columns:
        if "year" in c.lower():
            try:
                plot_df[c] = pd.to_datetime(plot_df[c].astype(str), errors="coerce", format="%Y")
            except Exception:
                pass


    st.write("**A preview of the data currently used for plotting:**")
    st.dataframe(plot_df.head(5), use_container_width=True)

    # 自动识别列类型
    num_cols = plot_df.select_dtypes(include="number").columns.tolist()
    dt_cols  = [c for c in plot_df.columns if np.issubdtype(plot_df[c].dtype, np.datetime64)]
    cat_cols = [c for c in plot_df.columns if c not in num_cols + dt_cols]

    chart = st.selectbox(
        "Chart Type",
        ["Histogram", "Bar (agg)", "Line (time)", "Scatter", "Correlation Heatmap"]
    )

    if chart == "Histogram":
        if not num_cols:
            st.warning("No numeric columns detected. Please specify a Value in the Column Mapper or select a different chart type.")
        else:
            col  = st.selectbox("Numeric column(s)", num_cols)
            bins = st.slider("Bins", 5, 80, 30)
            fig, ax = new_fig()  # 小尺寸
            edge = "white" if IS_DARK else "#2b2b2b"
            ax.hist(
                plot_df[col].dropna(),
                bins=bins,
                color=PRIMARY,              # 粉色主色
                edgecolor=edge,
                linewidth=0.8,
                alpha=0.95
            )
            ax.set_title(f"Histogram – {col}")
            ax.set_xlabel(col); ax.set_ylabel("Count")
            apply_theme(ax); st.pyplot(fig)

    elif chart == "Bar (agg)":
        if not num_cols or not (cat_cols or dt_cols):
            st.warning("A grouping column (category or time) and a numeric column are required.")
        else:
            x   = st.selectbox("Grouping Column (X)", cat_cols + dt_cols)
            y   = st.selectbox("Numeric Column (Y)", num_cols)
            agg = st.selectbox("Aggregation Method", ["mean","sum","median"])
            data = getattr(plot_df.groupby(x)[y], agg)().reset_index()

            fig, ax = new_fig()
            if x in dt_cols:
                data = data.sort_values(x)
                sns.lineplot(
                    data=data, x=x, y=y, marker="o", ax=ax,
                    color=PRIMARY   # 单系列用主色
                )
                ax.set_title(f"{agg.title()} {y} over {x}")
            else:
                sns.barplot(
                    data=data, x=y, y=x, ax=ax,
                    palette=brand_palette(5)  # 多系列用品牌调色
                )
                ax.set_title(f"{agg.title()} {y} by {x}")
            apply_theme(ax); st.pyplot(fig)

    elif chart == "Line (time)":
        if not dt_cols or not num_cols:
            st.warning("A datetime column and a numeric column are required. You can map a time column to 'Year' in the Column Mapper (automatic parsing will be attempted).")
        else:
            t = st.selectbox("Time Column", dt_cols)
            y = st.selectbox("Numeric Column (Y)", num_cols)
            g = st.selectbox("Grouping (Optional)", ["(none)"] + cat_cols)
            data = plot_df.dropna(subset=[t, y]).sort_values(t)

            fig, ax = new_fig()
            if g != "(none)":
                sns.lineplot(
                    data=data, x=t, y=y, hue=g, marker="o", ax=ax,
                    palette=brand_palette(6)   # 分组线条走粉紫系列
                )
            else:
                sns.lineplot(
                    data=data, x=t, y=y, marker="o", ax=ax,
                    color=PRIMARY
                )
            ax.set_title(f"{y} over {t}")
            apply_theme(ax); st.pyplot(fig)

    elif chart == "Scatter":
        if len(num_cols) < 2:
            st.warning("At least two numeric columns are required.")
        else:
            x = st.selectbox("X (Numeric)", num_cols, key="sc_x")
            y = st.selectbox("Y (Numeric)", num_cols, key="sc_y")
            g = st.selectbox("Color Grouping (Optional)", ["(none)"] + cat_cols, key="sc_g")

            fig, ax = new_fig()
            if g != "(none)":
                sns.scatterplot(
                    data=plot_df, x=x, y=y, hue=g, ax=ax,
                    palette=brand_palette(6)
                )
            else:
                sns.scatterplot(
                    data=plot_df, x=x, y=y, ax=ax,
                    color=PRIMARY
                )
            ax.set_title(f"{y} vs {x}")
            apply_theme(ax); st.pyplot(fig)

    else:  # Correlation Heatmap
        nums = plot_df.select_dtypes(include="number")
        if nums.shape[1] < 2:
            st.warning("A correlation heatmap requires at least two numeric columns.")
        else:
            corr = nums.corr(numeric_only=True)
            fig, ax = new_fig(7.5, 4.6)
            # 粉紫梯度
            cmap_corr = LinearSegmentedColormap.from_list(
                "corr_pink", ["#fff1fa", "#9b8aff", PRIMARY], N=256
            )
            sns.heatmap(
                corr, annot=True, fmt=".2f", cmap=cmap_corr, ax=ax,
                cbar_kws={"label": "Correlation"}
            )
            ax.set_title("Correlation Heatmap")
            apply_theme(ax); st.pyplot(fig)


# ----------------------------
# 4) Report（自动方法摘要）
# ----------------------------
with tab_report:
    st.subheader("Methods summary (auto-generated)")
    if len(st.session_state.pipeline)==0:
        st.info("No pipeline steps yet. Perform some cleaning in the **Clean & Transform** tab.")
    else:
        lines = ["# Data Processing Methods", "", "This dataset was processed inside the system as follows:"]
        for i, step in enumerate(st.session_state.pipeline, start=1):
            lines.append(f"{i}. **{step['step']}** — params: `{step['args']}`")
        md = "\n".join(lines)
        st.markdown(md)

        st.download_button(
            "Download methods.md",
            md.encode("utf-8"),
            file_name="methods.md",
            mime="text/markdown"
        )

st.divider()
st.caption("Tip: Upload → Clean → Visualise → Report. The pipeline is reproducible and exportable.")
