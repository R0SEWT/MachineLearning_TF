# app.py ────────────────────────────────────────────────────────────────
# GUI Streamlit – Perú C-Inversiones  ·  XGBoost (extensible)
# Cumple heurísticas básicas de Nielsen                                   »

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json, pathlib, datetime as dt
from io import StringIO

# ─── 0 · Config general ────────────────────────────────────────────────
st.set_page_config(
    page_title="🚀 Recomendador Cripto",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ‼️ Para subir archivos >200 MB, arranca con:
#    streamlit run app.py --server.maxUploadSize=1024

# ─── 1 · Rutas por defecto ─────────────────────────────────────────────
BASE      = pathlib.Path(__file__).resolve().parent.parent
DATA_FP   = BASE / "data" / "crypto_ohlc_join_hourly_full.csv"
MODEL_DIR = BASE / "models"
XGB_PATH  = MODEL_DIR / "xgboost_optuna_optimized_20250710_022712.model"
XGB_CFG   = MODEL_DIR / "xgb_optuna_best_params_20250708_070015.json"

def dedup(seq):
    seen = set(); out = []
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

# ─── 2 · Carga modelo XGBoost (caché) ──────────────────────────────────
@st.cache_resource(show_spinner=True)
def load_xgboost():
    booster = xgb.Booster()
    booster.load_model(str(XGB_PATH))
    feats = None
    if XGB_CFG.exists():
        with open(XGB_CFG, encoding="utf-8") as f:
            cfg = json.load(f)
        for cand in [
            cfg.get("feature_names"),
            cfg.get("features"),
            cfg.get("learner", {}).get("feature_names"),
            cfg.get("learner", {})
               .get("gradient_booster", {})
               .get("model", {})
               .get("feature_names"),
        ]:
            if cand:
                feats = cand; break
    if not feats and getattr(booster, "feature_names", None):
        feats = booster.feature_names
    return booster, dedup(feats or [])

xgb_model, xgb_feats = load_xgboost()

# ─── 3 · Configuración del modelo ──────────────────────────────────────
MODELS = {
    "XGBoost": {
        "model":      xgb_model,
        "features":   xgb_feats,
        "cat_feats":  ["narrative", "cluster_id"],
    },
}

# ─── 4 · Lectura y preprocesado ───────────────────────────────────────
def read_default_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_FP, low_memory=False)
    df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    # quedarnos solo con la fila más reciente por símbolo
    return df.sort_values("date")\
             .groupby("symbol", as_index=False)\
             .tail(1)

def read_uploaded_df(uploaded: StringIO) -> pd.DataFrame:
    try:
        df = pd.read_csv(uploaded, low_memory=False)
    except Exception as e:
        raise ValueError(f"No se pudo leer el CSV: {e}")
    required = {"symbol","timestamp","narrative","market_cap","volume"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {', '.join(missing)}")
    df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df.sort_values("date")\
             .groupby("symbol", as_index=False)\
             .tail(1)

def base_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    # narrativa
    X["narrative"] = X.get("narrative","unknown")\
                      .astype(str).fillna("unknown")
    # cluster_id como texto
    if "cluster_id" not in X.columns:
        X["cluster_id"] = "-1"
    X["cluster_id"] = pd.to_numeric(X["cluster_id"],
                                    errors="coerce")\
                       .fillna(-1).astype(int).astype(str)
    # numéricos críticos
    for c in ("price","market_cap","volume"):
        if c in X.columns:
            X[c] = pd.to_numeric(X[c],
                                 errors="coerce")\
                     .fillna(0.0)
    return X

def encode_align(X: pd.DataFrame,
                 feat_list: list[str],
                 cat_cols: list[str]) -> pd.DataFrame:
    X2 = X.copy()
    # asegurar categorías mínimas
    for c in cat_cols:
        if c not in X2.columns:
            X2[c] = "unknown"
    # one‐hot
    X_enc = pd.get_dummies(X2,
                           columns=cat_cols,
                           dtype=float)
    # rellenar y reordenar
    for col in feat_list:
        if col not in X_enc.columns:
            X_enc[col] = 0.0
    return X_enc[feat_list]

# ─── 5 · Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Dataset")
    up_file   = st.file_uploader("Subir CSV propio (opcional)", type="csv")
    top_k     = st.slider("Top N sugerencias", 5, 50, 20)
    narrative = st.selectbox("Narrativa",
                             ["Todas","ai","gaming","meme","rwa"])
    st.divider()
    if st.button("❓ Ayuda"):
        st.info(
            "1) (Opcional) sube tu CSV con OHLC+features.\n"
            "2) Elige narrativa.\n"
            "3) Explora ranking y métricas; descarga.",
            icon="💡",
        )

# ─── 6 · Leer dataset ──────────────────────────────────────────────────
try:
    raw_df = read_uploaded_df(up_file) if up_file else read_default_df()
except ValueError as e:
    st.error(str(e), icon="❌")
    st.stop()

# ─── 7 · Filtrar narrativa ─────────────────────────────────────────────
df_f = raw_df if narrative=="Todas" else raw_df.query("narrative==@narrative")
if df_f.empty:
    st.warning("Sin datos para la narrativa elegida.", icon="⚠️")
    st.stop()

# ─── 8 · Predecir ──────────────────────────────────────────────────────
X_base = base_preprocess(df_f)
info   = MODELS["XGBoost"]
X_enc  = encode_align(X_base,
                      info["features"],
                      info["cat_feats"])
dmat   = xgb.DMatrix(X_enc.values,
                     feature_names=info["features"])
preds  = info["model"].predict(dmat)

df_f    = df_f.assign(exp_ret_30d=preds)\
               .sort_values("exp_ret_30d", ascending=False)
top_df  = df_f.head(top_k)

# ─── 9 · Métricas rápidas ─────────────────────────────────────────────
st.markdown("### 📊 Estadísticas del subconjunto")
c1, c2, c3 = st.columns(3)
c1.metric("Activos analizados",         f"{len(df_f):,}")
c2.metric("Market Cap medio",           f"${df_f['market_cap'].mean():,.0f}")
c3.metric("Retorno esperado medio (30d)", f"{df_f['exp_ret_30d'].mean():+.2%}")

# ───10 · Tabla principal ──────────────────────────────────────────────
basic = ["symbol","narrative","market_cap","exp_ret_30d"]
adv   = [c for c in df_f.columns if c not in basic+["date","timestamp"]]
showc = basic + adv

st.markdown(f"## 🏅 Top {top_k} – XGBoost")
st.caption(f"Actualizado: {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC}")
st.dataframe(
    top_df[showc]
         .style.format({"market_cap":"{:,.0f}",
                        "exp_ret_30d":"{:+.2%}"}),
    use_container_width=True, hide_index=True
)

# ───11 · Descarga CSV ─────────────────────────────────────────────────
with st.popover("⬇️ Descargar"):
    if st.checkbox("Confirmo descarga"):
        st.download_button(
            "Descargar archivo",
            top_df.to_csv(index=False).encode(),
            f"recomendaciones_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M}.csv",
            mime="text/csv"
        )

# ───12 · Gráficas ─────────────────────────────────────────────────────
st.markdown("### 📈 Retorno esperado vs Capitalización de Mercado")
st.scatter_chart(
    top_df,
    x="market_cap",
    y="exp_ret_30d",
    size="volume",
    color="narrative",
    height=420,
)

# Nueva gráfica: Retorno promedio esperado por narrativa
st.markdown("### 📊 Retorno promedio esperado por narrativa")
narr_avg = df_f.groupby("narrative")["exp_ret_30d"]\
               .mean()\
               .sort_values(ascending=False) * 100
st.bar_chart(narr_avg, height=320)

st.caption("© 2025 Perú C-Inversiones · demo académica")
