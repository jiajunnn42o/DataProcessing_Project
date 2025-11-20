# app.py — dataset-agnostic visualisation; all your functions kept intact
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from io import StringIO
import re
import plotly.express as px
import plotly.io as pio
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster
import json

# ----------------------------
# 页面设置
# ----------------------------
st.set_page_config(
    page_title="Dataset Explorer",
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
    base = [PRIMARY, "#9b8aff", "#6f6f7a", "#d9b3ff", "#b38dff", "#ff99f3"]
    return base[:max(1, n)]

def new_fig(w=7.2, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax

sns.set_palette(brand_palette(6))

# Plotly theme + color-blind palette
CB_SAFE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AB"
]
pio.templates.default = "plotly_dark" if IS_DARK else "plotly_white"

# ----------------------------
# 辅助增强：重复值指标 & 安全自动数值转换 & 读CSV缓存
# ----------------------------
def duplicate_metrics(df, subset=None):
    mask_first = df.duplicated(subset=subset, keep="first")
    mask_last  = df.duplicated(subset=subset, keep="last")
    mask_all   = df.duplicated(subset=subset, keep=False)
    return {
        "rows_marked_duplicate (keep='first')": int(mask_first.sum()),
        "rows_marked_duplicate (keep='last')" : int(mask_last.sum()),
        "rows_in_duplicated_groups (all)"     : int(mask_all.sum())
    }, mask_first, mask_all

@st.cache_data(show_spinner=False)
def _read_csv_fast(text):
    return pd.read_csv(StringIO(text), sep=None, engine="python")

def _safe_autocast_numeric(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            s = df[col].astype(str).str.replace(r"[,\s]", "", regex=True)
            frac_numeric = s.str.match(r"^-?\d+(\.\d+)?$").mean()
            if frac_numeric > 0.9:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ----------------------------
# 智能读取器：CSV/Excel + 世行宽表 reshape
# ----------------------------
def smart_read(uploaded_file):
    """
    Robust reader for CSV/Excel with World Bank wide-to-long reshape.
    """
    name = uploaded_file.name.lower()

    def postprocess_df(df: pd.DataFrame) -> pd.DataFrame:
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

        # long form unify
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

        if {"Country Name", "Year", "CO2_per_capita"}.issubset(df.columns):
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
            df["CO2_per_capita"] = pd.to_numeric(df["CO2_per_capita"], errors="coerce")
            df = df.dropna(subset=["Year", "CO2_per_capita"])
            df["Year"] = df["Year"].astype(int)
        return df

    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
        return postprocess_df(df)

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

    try:
        df = _read_csv_fast(trimmed)
        return postprocess_df(df)
    except Exception:
        pass
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(StringIO(trimmed), sep=sep)
            return postprocess_df(df)
        except Exception:
            continue
    df = pd.read_csv(StringIO(trimmed), sep=None, engine="python", on_bad_lines="skip")
    return postprocess_df(df)

# ----------------------------
# 会话状态
# ----------------------------
if "df_raw" not in st.session_state:    st.session_state.df_raw = None
if "df_clean" not in st.session_state:  st.session_state.df_clean = None
if "pipeline" not in st.session_state:  st.session_state.pipeline = []
if "generic_view" not in st.session_state: st.session_state.generic_view = None
if "last_upload_name" not in st.session_state: st.session_state.last_upload_name = None
if "roles" not in st.session_state:     st.session_state.roles = {"time": None, "category": None, "value": None}
if "auto_done" not in st.session_state: st.session_state.auto_done = False

# ----------------------------
# 默认数据（你的 CO₂ CSV） — 保留，但仅在无数据时使用
# ----------------------------
DEFAULT_PATH = "data/co2_clean_asean.csv"
def load_default():
    df = pd.read_csv(DEFAULT_PATH)
    df["Year"] = df["Year"].astype(int)
    df["CO2_per_capita"] = pd.to_numeric(df["CO2_per_capita"], errors="coerce")
    df = df.dropna(subset=["CO2_per_capita"])
    return df

# ----------------------------
# CO₂ 专用可视化工具（原样保留）
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
    palette = {"Malaysia": PRIMARY, "ASEAN Average": "#9b8aff", "World": "#6fc3ff"}
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
    plt.close(fig)

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
    plt.close(fig)

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
    plt.close(fig)

# ----------------------------
# 顶部标题（动态：根据上传文件名）
# ----------------------------
page_name = st.session_state.last_upload_name or "Dataset Explorer"
st.title(f"{page_name}")
st.caption("Upload any CSV/Excel. The app adapts to your schema. Cleaning → Visualise → Report.")

# ----------------------------
# 页签
# ----------------------------
# ============ 两栏布局：左 = 上传 & 清洗；右 = 可视化 & 报告 ============
col_left, col_right = st.columns([1, 1.2], gap="large")

# 左边：Upload & Inspect + Clean & Transform
with col_left:
    tab_upload, tab_clean = st.tabs(["Upload & Inspect", "Clean & Transform"])
# ----------------------------
# 1) Upload & Inspect
# ----------------------------
    with tab_upload:
        st.subheader("Upload a dataset (CSV/Excel) or use the sample")
        up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

        colA, colB = st.columns([1,1])
        with colA:
            if st.button("Use sample dataset (CO₂)"):
                st.session_state.df_raw = load_default()
                st.session_state.df_clean = None
                st.session_state.pipeline = [{"step":"load_default", "args": {}}]
                st.session_state.last_upload_name = "__SAMPLE__ co2_clean_asean.csv"
                st.success("Loaded sample dataset.")
        if up is not None and st.session_state.last_upload_name != up.name:
            try:
                df_raw = smart_read(up)
                st.session_state.df_raw = df_raw
                st.session_state.df_clean = None
                st.session_state.pipeline = [{"step":"upload", "args":{"filename": up.name}}]
                st.session_state.last_upload_name = up.name
                if len(df_raw) > 500_000:
                    st.warning(f"Large dataset detected: {len(df_raw):,} rows. Some operations may be slow; visualisations may use only subsets.")
                st.success(f"Uploaded: {up.name}  (rows={len(df_raw)}, cols={len(df_raw.columns)})")
            except Exception as e:
                st.error(f"Failed to read file: {e}")

        df_show = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw

        # --- Status ribbon (enhanced)
        if df_show is not None:
            st.info(
                f"Active dataset: {'cleaned' if st.session_state.df_clean is not None else 'raw'} | "
                f"Rows: {len(df_show):,} | Columns: {len(df_show.columns):,} | "
                f"Pipeline steps: {len(st.session_state.pipeline)}"
            )

        if df_show is not None:
            st.markdown("**Data preview (first 10 rows)**")
            st.dataframe(df_show.head(10), use_container_width=True)

            st.markdown("**Missing values by column**")
            st.write(df_show.isna().sum())

            # Quick data profile
            with st.expander("Quick data profile"):
                num = df_show.select_dtypes(include="number")
                cat = df_show.select_dtypes(exclude="number")
                if not num.empty:
                    st.markdown("**Numeric summary**")
                    st.dataframe(num.describe().T)
                if not cat.empty:
                    st.markdown("**Categorical unique counts**")
                    st.dataframe(cat.nunique().rename("n_unique"))

            # ---------- ENHANCED DUPLICATES SECTION ----------
            st.markdown("### Duplicate Analysis")
            metrics, dup_first_mask, dup_all_mask = duplicate_metrics(df_show, subset=None)

            st.markdown(
                f"""
                <div style="
                    background-color:{SEC};
                    border: 1px solid {GRID};
                    border-radius: 12px;
                    padding: 18px 24px;
                    margin: 6px 0 12px 0;
                    box-shadow: 0px 2px 6px rgba(0,0,0,0.15);
                ">
                    <h4 style="margin-bottom:8px; color:{TXT};">Duplicate Rows Summary</h4>
                    <ul style="margin-top:8px;color:{TXT}">
                    <li>Rows flagged (keep='first'): <b style="color:{PRIMARY}">{metrics["rows_marked_duplicate (keep='first')"]:,}</b></li>
                    <li>Rows flagged (keep='last') : <b style="color:{PRIMARY}">{metrics["rows_marked_duplicate (keep='last')"]:,}</b></li>
                    <li>Rows in duplicated groups : <b style="color:{PRIMARY}">{metrics["rows_in_duplicated_groups (all)"]:,}</b></li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.expander("Find & filter duplicates by columns"):
                cols_with_dups = [
                    c for c in df_show.columns
                    if pd.Series(df_show[c]).duplicated(keep=False).any()
                ]
                cols_pick = st.multiselect(
                    "Columns to consider for duplicates (leave empty = all columns)",
                    options=df_show.columns.tolist(),
                    default=cols_with_dups,
                )

                keep_mode = st.selectbox(
                    "Mark which occurrence as duplicate",
                    ["first", "last", "none (mark all)"],
                    help="Matches pandas.duplicated(keep=...) behaviour.",
                )
                keep_arg = {"first": "first", "last": "last", "none (mark all)": False}[keep_mode]

                if st.button("Analyse duplicates", key="btn_analyse_dups"):
                    subset_arg = cols_pick if len(cols_pick) > 0 else None
                    dup_mask = df_show.duplicated(subset=subset_arg, keep=keep_arg)

                    st.info(f"Rows flagged as duplicates: **{int(dup_mask.sum()):,}**")
                    st.dataframe(df_show[dup_mask].head(200), use_container_width=True)

                    if subset_arg:
                        cols = subset_arg if isinstance(subset_arg, list) else [subset_arg]
                        grp = (
                            df_show.loc[dup_mask, cols]
                            .groupby(cols, dropna=False)
                            .size()
                            .reset_index(name="count")
                            .sort_values("count", ascending=False, kind="mergesort")
                        )
                        st.markdown("**Duplicate groups (top 50)**")
                        st.dataframe(grp.head(50), use_container_width=True)

                    csv_dups = df_show[dup_mask].to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download duplicate rows (CSV)",
                        csv_dups,
                        file_name="duplicates_filtered.csv",
                        mime="text/csv",
                        key="dl_dups",
                    )
            # ---------- END ENHANCED DUPLICATES SECTION ----------

            # ====== Column Mapper（当没有特定列时使用）======
            required = {"Year", "Country Name", "CO2_per_capita"}
            if not required.issubset(df_show.columns):
                st.markdown("### Column Mapper（Universal Data Adaptation）")
                st.info("Map your own columns to enable generic visualisation with time/category/value semantics.")

                cols = df_show.columns.tolist()

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

                    if "Year" in mapped.columns:
                        try:
                            mapped["Year"] = pd.to_datetime(mapped["Year"], errors="coerce", infer_datetime_format=True)
                        except Exception:
                            pass
                    if "Value" in mapped.columns:
                        mapped["Value"] = pd.to_numeric(mapped["Value"], errors="coerce")

                    st.session_state.generic_view = mapped
                    st.session_state.roles = {
                        "time": None if col_time=="(none)" else "Year",
                        "category": None if col_cat=="(none)" else "Category",
                        "value": None if col_val=="(none)" else "Value",
                    }
                    st.success("A universal mapping view has been generated. Use it in the Visualise tab → Generic Visualise.")

            # Export / import roles mapping
            with st.expander("Save or load column-role mapping (roles.json)"):
                roles_json = json.dumps(st.session_state.roles)
                st.download_button(
                    "Download roles.json",
                    roles_json.encode("utf-8"),
                    file_name="roles.json",
                    mime="application/json",
                    key="dl_roles",
                )
                roles_file = st.file_uploader("Import roles.json", type=["json"], key="roles_in")
                if roles_file is not None:
                    try:
                        st.session_state.roles = json.loads(roles_file.read().decode("utf-8"))
                        st.success("Roles mapping imported. Visualisation defaults will follow these roles.")
                    except Exception as e:
                        st.error(f"Failed to import roles: {e}")

    # ----------------------------
    # 2) Clean & Transform (unchanged logic, but safer auto-cast)
    # ----------------------------
    with tab_clean:
        st.subheader("Step-by-step cleaning with preview and pipeline log")

        clean_mode = st.radio(
            "Cleaning Mode",
            ["Manual", "Auto"],
            captions=["Step-by-step manual cleaning", "Auto-run with default cleaning pipeline"],
            horizontal=True,
            key="clean_mode_radio",
        )

        base = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw

        if clean_mode == "Auto":
            if st.session_state.df_raw is None and base is None:
                st.info("No data yet. Go to **Upload & Inspect** to load a dataset.")
            elif not st.session_state.auto_done:
                src = st.session_state.df_raw if st.session_state.df_raw is not None else base
                if src is None:
                    st.warning("No data to clean. Please upload or load the default dataset in 'Upload & Inspect'.")
                else:
                    df = src.copy()
                    df = df.fillna(df.mean(numeric_only=True))
                    df = df.drop_duplicates()
                    df = _safe_autocast_numeric(df)

                    st.session_state.df_clean = df
                    st.session_state.pipeline.append({
                        "step": "auto_clean",
                        "args": {"methods": ["fill mean", "drop duplicates", "convert numeric (safe)"]}
                    })
                    st.session_state.auto_done = True
                    st.success("Auto cleaning completed.")
            else:
                st.info("Auto cleaning already applied. Switch to Manual for step-by-step actions.")
        else:
            st.session_state.auto_done = False

        base = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw
        if base is None:
            st.info("No data yet. Go to **Upload & Inspect** to load a dataset.")
        else:
            df_work = base.copy()
            st.markdown("### Actions")

            act = st.selectbox(
                "Choose an action",
                ["(select)", "Filter rows (keep/remove)", "Handle missing values",
                "Remove duplicates", "Cast to numeric", "Min-Max scale",
                "Clip outliers (IQR)", "Standardize text categories"]
            )

            if act == "Filter rows (keep/remove)":
                try:
                    default_idx = next(
                        i for i, c in enumerate(df_work.columns)
                        if str(c).lower().strip() in ["country", "country name", "country_name", "countryname", "country_code"]
                    )
                except StopIteration:
                    default_idx = 0

                col = st.selectbox("Column to filter", df_work.columns.tolist(), index=default_idx)

                uniq_vals = pd.Series(df_work[col].astype(str).unique())
                uniq_vals = uniq_vals.sort_values(kind="mergesort").tolist()

                use_asean_preset = st.checkbox("Quick preset: Select ASEAN countries")
                preselected = []
                if use_asean_preset:
                    target = set(ASEAN)
                    preselected = [v for v in uniq_vals if v in target]
                    if not preselected:
                        st.info("No ASEAN names matched in this column. You can still select manually.")

                picked = st.multiselect("Pick values", options=uniq_vals, default=preselected)

                pasted = st.text_area("Or paste values (comma/semicolon/newline separated)",
                                    placeholder="Malaysia, Singapore, Thailand\nVietnam")
                if pasted.strip():
                    extra = [s.strip() for s in re.split(r"[,;\n]", pasted) if s.strip()]
                    picked = sorted(set(picked) | set(extra))

                mode = st.radio("Mode", ["Keep only selected", "Remove selected"], horizontal=True)

                if st.button("Apply", key="apply_filter_values"):
                    if not picked:
                        st.warning("Please select at least one value to proceed.")
                    else:
                        before = len(df_work)
                        mask = df_work[col].astype(str).isin(set(picked))
                        if mode == "Keep only selected":
                            df_work = df_work[mask];  mode_key = "keep"
                        else:
                            df_work = df_work[~mask]; mode_key = "remove"

                        st.session_state.df_clean = df_work
                        st.session_state.pipeline.append({
                            "step": "filter_values",
                            "args": {
                                "column": col,
                                "mode": mode_key,
                                "values": (picked[:20] + (["..."] if len(picked) > 20 else [])),
                                "rows_before": before,
                                "rows_after": len(df_work)
                            }
                        })
                        st.success(f"Filtered ({mode}). Rows: {before} → {len(df_work)}")

            elif act == "Handle missing values":
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
                            mode_val = df_work[c].mode()
                            if not mode_val.empty: df_work[c] = df_work[c].fillna(mode_val[0])
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

            elif act == "Clip outliers (IQR)":
                cols = st.multiselect("Numeric columns to clip", df_work.select_dtypes(include=np.number).columns.tolist())
                if st.button("Apply", key="apply_iqr"):
                    before_stats = df_work[cols].describe().to_dict() if cols else {}
                    for c in cols:
                        q1, q3 = df_work[c].quantile([0.25, 0.75])
                        iqr = q3 - q1
                        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
                        df_work[c] = df_work[c].clip(lo, hi)
                    st.session_state.df_clean = df_work
                    st.session_state.pipeline.append({"step": "clip_iqr", "args": {"cols": cols, "before": before_stats}})
                    st.success("Outliers clipped using IQR method.")

            elif act == "Standardize text categories":
                cols = st.multiselect("Text columns to standardize", df_work.select_dtypes(include="object").columns.tolist())
                if st.button("Apply", key="apply_std_text"):
                    for c in cols:
                        df_work[c] = (
                            df_work[c].astype(str)
                            .str.strip()
                            .str.replace(r"\s+", " ", regex=True)
                            .str.normalize("NFKC")
                        )
                    st.session_state.df_clean = df_work
                    st.session_state.pipeline.append({"step": "standardize_text", "args": {"cols": cols}})
                    st.success("Text categories standardized (trim + whitespace normalization).")

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


# 右边：Visualise + Report
with col_right:
    tab_viz, tab_report = st.tabs(["Results", "Report"])
    # ----------------------------
    # 3) Visualise (dataset-agnostic by default)
    # ----------------------------
    with tab_viz:
        st.subheader("Results")

        # Pick the active dataframe
        if st.session_state.df_clean is not None and not st.session_state.df_clean.empty:
            df_base = st.session_state.df_clean
        elif st.session_state.df_raw is not None and not st.session_state.df_raw.empty:
            df_base = st.session_state.df_raw
        else:
            # 还没有任何 dataset（没上传 & 没按 sample 按钮）
            df_base = None

        if df_base is None:
                st.info("No dataset yet. Please upload a CSV/Excel or click 'Use sample dataset (CO₂)' on the left.")
        else:
            # Choose data source for plotting (Column Mapper if provided)
            gv = st.session_state.get("generic_view", None)
            use_generic = st.toggle("Use Column Mapper view if available", value=True if gv is not None else False)
            plot_df = gv if (use_generic and gv is not None) else df_base.copy()

            # Detect column types
            num_cols = plot_df.select_dtypes(include="number").columns.tolist()
            dt_cols  = [c for c in plot_df.columns if np.issubdtype(plot_df[c].dtype, np.datetime64)]
            cat_cols = [c for c in plot_df.columns if c not in num_cols + dt_cols]

            # ---- INTERACTIVE MODE ----
            st.markdown("### Interactive mode (Plotly / Leaflet)")
            inter_mode = st.radio("Pick interactive engine", ["Auto", "Plotly", "Leaflet (map)"], horizontal=True)

            # Color-blind friendly toggle
            cb = st.toggle("Color-blind friendly palette", value=False)
            palette_seq = CB_SAFE if cb else None

            # Helper: detect geo possibilities
            def _has_latlon(cols):
                lc = [str(c).lower() for c in cols]
                return ("lat" in lc or "latitude" in lc) and ("lon" in lc or "long" in lc or "lng" in lc or "longitude" in lc)

            def _country_col(cols):
                for c in cols:
                    lc = str(c).lower()
                    if any(k in lc for k in ["country", "nation", "location", "iso", "region"]):
                        return c
                return None

            # Recommend chart
            recommended = None
            if _has_latlon(plot_df.columns):
                recommended = "map_latlon"
            elif _country_col(plot_df.columns):
                recommended = "map_country" if num_cols else None
            elif dt_cols and num_cols:
                recommended = "line"
            elif len(num_cols) >= 2:
                recommended = "scatter"
            elif num_cols:
                recommended = "hist"
            elif cat_cols and num_cols:
                recommended = "bar"

            # auto route to map if appropriate
            if inter_mode in ["Auto", "Plotly"]:
                if inter_mode == "Auto" and recommended in ["map_latlon", "map_country"]:
                    inter_mode = "Leaflet (map)"

            role = st.session_state.roles

            # --- Plotly path (with Line index option added) ---
            if inter_mode in ["Auto", "Plotly"]:
                st.write("**Preview of current plotting data:**")
                st.dataframe(plot_df.head(5), use_container_width=True)

                chart_options = ["Line (time)", "Line (index)", "Bar (agg)", "Scatter", "Histogram", "Correlation Heatmap"]
                default_idx = 0 if recommended == "line" else (2 if recommended == "bar" else (3 if recommended == "scatter" else (4 if recommended == "hist" else 0)))
                chart = st.selectbox("Chart Type (interactive)", chart_options, index=default_idx)

                if chart == "Line (time)":
                    x_candidates = dt_cols + cat_cols + num_cols
                    if not num_cols or not x_candidates:
                        st.warning("You need at least one numeric column for Y and one column for X.")
                    else:
                        pref_x = role.get("time") or role.get("category")
                        pref_y = role.get("value")
                        x_index = x_candidates.index(pref_x) if pref_x in x_candidates else 0
                        y_index = num_cols.index(pref_y) if pref_y in num_cols else 0

                        x = st.selectbox("X Column", x_candidates, index=x_index)
                        y = st.selectbox("Y (Numeric Column)", num_cols, index=y_index)
                        color = st.selectbox("Grouping/Color (optional)", ["(none)"] + [c for c in plot_df.columns if c not in [x, y]])
                        hover = st.multiselect("Extra hover fields", [c for c in plot_df.columns if c not in [x, y]])
                        if plot_df[x].dtype == "object":
                            try:
                                plot_df[x] = pd.to_datetime(plot_df[x], errors="coerce", infer_datetime_format=True)
                            except Exception:
                                pass
                        fig = px.line(
                            plot_df.sort_values(x),
                            x=x, y=y,
                            color=None if color == "(none)" else color,
                            markers=True, hover_data=hover,
                            title=f"{y} over {x}",
                            color_discrete_sequence=palette_seq
                        )
                        fig.update_traces(hovertemplate=f"{x}=%{{x}}<br>{y}=%{{y:.3f}}<extra></extra>")
                        st.plotly_chart(fig, use_container_width=True)

                elif chart == "Line (index)":
                    if not num_cols:
                        st.warning("At least one numeric column is required.")
                    else:
                        pref_y = role.get("value")
                        y_index = num_cols.index(pref_y) if pref_y in num_cols else 0
                        y = st.selectbox("Numeric column (Y)", num_cols, index=y_index)
                        df_idx = plot_df.reset_index()
                        idx = df_idx.columns[0]
                        fig = px.line(df_idx, x=idx, y=y, markers=True, title=f"{y} over row index",
                                    color_discrete_sequence=palette_seq)
                        fig.update_traces(hovertemplate=f"Index=%{{x}}<br>{y}=%{{y:.3f}}<extra></extra>")
                        st.plotly_chart(fig, use_container_width=True)

                elif chart == "Bar (agg)":
                    if not num_cols or not (cat_cols or dt_cols):
                        st.warning("A grouping column (category or time) and a numeric column are required.")
                    else:
                        pref_x = role.get("category") or role.get("time")
                        pref_y = role.get("value")
                        group_candidates = cat_cols + dt_cols
                        x_index = group_candidates.index(pref_x) if pref_x in group_candidates else 0
                        y_index = num_cols.index(pref_y) if pref_y in num_cols else 0

                        x = st.selectbox("Grouping Column (X)", group_candidates, index=x_index)
                        y = st.selectbox("Numeric Column (Y)", num_cols, index=y_index)
                        agg = st.selectbox("Aggregation", ["mean","sum","median"])
                        df_agg = getattr(plot_df.groupby(x)[y], agg)().reset_index()
                        if x in dt_cols:
                            fig = px.line(df_agg.sort_values(x), x=x, y=y, markers=True,
                                        title=f"{agg.title()} of {y} over {x}",
                                        color_discrete_sequence=palette_seq)
                            fig.update_traces(hovertemplate=f"{x}=%{{x}}<br>{y}=%{{y:.3f}}<extra></extra>")
                        else:
                            fig = px.bar(df_agg, x=y, y=x, orientation="h",
                                        title=f"{agg.title()} of {y} by {x}",
                                        color_discrete_sequence=palette_seq)
                            fig.update_traces(hovertemplate=f"{x}=%{{y}}<br>{y}=%{{x:.3f}}<extra></extra>")
                        st.plotly_chart(fig, use_container_width=True)

                elif chart == "Scatter":
                    if len(num_cols) < 2:
                        st.warning("At least two numeric columns are required.")
                    else:
                        pref_y = role.get("value")
                        y_index = num_cols.index(pref_y) if pref_y in num_cols else 1 if len(num_cols) > 1 else 0
                        x = st.selectbox("X (numeric)", num_cols, key="px_sc_x")
                        y = st.selectbox("Y (numeric)", num_cols, index=y_index, key="px_sc_y")
                        color = st.selectbox("Color (optional)", ["(none)"] + cat_cols, key="px_sc_c")
                        size  = st.selectbox("Bubble size (optional)", ["(none)"] + num_cols, key="px_sc_s")
                        hover = st.multiselect("Extra hover fields", [c for c in plot_df.columns if c not in [x, y]])
                        fig = px.scatter(
                            plot_df, x=x, y=y,
                            color=None if color=="(none)" else color,
                            size=None if size=="(none)" else size,
                            hover_data=hover, trendline=None, title=f"{y} vs {x}",
                            color_discrete_sequence=palette_seq
                        )
                        fig.update_traces(hovertemplate=f"{x}=%{{x:.3f}}<br>{y}=%{{y:.3f}}<extra></extra>")
                        st.plotly_chart(fig, use_container_width=True)

                elif chart == "Histogram":
                    if not num_cols:
                        st.warning("No numeric columns detected.")
                    else:
                        pref_y = role.get("value")
                        col_index = num_cols.index(pref_y) if pref_y in num_cols else 0
                        col = st.selectbox("Numeric column", num_cols, index=col_index)
                        bins = st.slider("Bins", 5, 80, 30)
                        fig = px.histogram(plot_df, x=col, nbins=bins,
                                        title=f"Histogram — {col}",
                                        color_discrete_sequence=palette_seq)
                        fig.update_traces(hovertemplate=f"{col}=%{{x}}<br>Count=%{{y}}<extra></extra>")
                        st.plotly_chart(fig, use_container_width=True)

                else:  # Correlation Heatmap
                    nums = plot_df.select_dtypes(include="number")
                    if nums.shape[1] < 2:
                        st.warning("A correlation heatmap requires at least two numeric columns.")
                    else:
                        corr = nums.corr(numeric_only=True)
                        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap",
                                        color_continuous_scale=["#fff1fa", "#9b8aff", PRIMARY])
                        st.plotly_chart(fig, use_container_width=True)

            # --- LEAFLET MAP path (auto or forced) ---
            if inter_mode == "Leaflet (map)":
                st.write("**Map preview (Leaflet via folium):**")
                cols = list(plot_df.columns)
                lat_candidates = [c for c in cols if str(c).lower() in ["lat","latitude"]]
                lon_candidates = [c for c in cols if str(c).lower() in ["lon","long","lng","longitude"]]
                country_col = _country_col(cols)

                if lat_candidates and lon_candidates:
                    lat_col = st.selectbox("Latitude column", lat_candidates)
                    lon_col = st.selectbox("Longitude column", lon_candidates)
                    popup_col = st.selectbox("Popup/label column (optional)", ["(none)"] + [c for c in cols if c not in [lat_col, lon_col]])
                    center_lat = plot_df[lat_col].dropna().astype(float).mean() if not plot_df[lat_col].dropna().empty else 3.1390
                    center_lon = plot_df[lon_col].dropna().astype(float).mean() if not plot_df[lon_col].dropna().empty else 101.6869
                    m = folium.Map(location=[center_lat, center_lon], tiles="OpenStreetMap", zoom_start=4)
                    cluster = MarkerCluster().add_to(m)
                    for _, row in plot_df.dropna(subset=[lat_col, lon_col]).iterrows():
                        popup_text = None if popup_col=="(none)" else str(row[popup_col])
                        folium.CircleMarker(
                            location=[float(row[lat_col]), float(row[lon_col])],
                            radius=4, weight=1, fill=True, popup=popup_text
                        ).add_to(cluster)
                    st_folium(m, use_container_width=True, returned_objects=[])

                elif country_col and num_cols:
                    target_metric = st.selectbox("Metric (numeric)", num_cols)
                    mode = st.selectbox("Country reference", ["auto (names)", "ISO-2", "ISO-3"])
                    locmode = {"auto (names)":"country names", "ISO-2":"ISO-2", "ISO-3":"ISO-3"}[mode]
                    fig = px.choropleth(
                        plot_df.dropna(subset=[country_col]),
                        locations=country_col, locationmode=locmode,
                        color=target_metric, hover_name=country_col,
                        title=f"{target_metric} by Country",
                        color_continuous_scale="RdPu"
                    )
                    fig.update_geos(fitbounds="locations", visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No lat/lon or country columns detected—switch to Plotly charts or map your columns with the Column Mapper.")

            # ====== Optional: keep your original CO₂/ASEAN charts if the schema matches ======
            required_cols = {"Year", "Country Name", "CO2_per_capita"}
            has_co2_view = required_cols.issubset(df_base.columns)
            if has_co2_view:
                with st.expander("Legacy CO₂/ASEAN charts (from your original code)"):
                    ymin, ymax = int(df_base["Year"].min()), int(df_base["Year"].max())
                    year_range = st.slider("Year range", min_value=ymin, max_value=ymax, value=(max(1990, ymin), ymax), step=1)

                    labels = {
                        "line": "Line: Country vs ASEAN vs World",
                        "bar":  "Bar: Latest Year (ASEAN + World)",
                        "heat": "Heatmap: ASEAN & World",
                    }
                    view = st.radio("Chart", options=["line","bar","heat"], format_func=lambda k: labels[k], horizontal=True)
                    combined = compute_combined(df_base)
                    if view == "line":   line_chart(combined, year_range)
                    elif view == "bar":  bar_chart(df_base, year_range)
                    else:                heatmap_chart(df_base, year_range)

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
st.caption("Upload any dataset → Clean → Visualise. Generic visualisations adapt to your columns; CO₂ charts appear only if applicable.")
