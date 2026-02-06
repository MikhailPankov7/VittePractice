# -*- coding: utf-8 -*-
"""
Science Events Web App

Endpoints:
- GET  /                    -> page_index
- GET  /events              -> page_events
- POST /events/predict       -> api_events_predict
- GET  /submission          -> page_submission
- POST /api/submission/predict -> api_submission_predict
- GET  /focus               -> page_focus
- POST /api/focus/check      -> api_focus_check
- POST /api/focus/check_by_id -> api_focus_check_by_id (если появится индекс)
- GET  /methodology         -> page_methodology
- POST /api/assistant/report -> api_assistant_report
"""

import json
import os
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import joblib

from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import HTTPException

# ---------------------------------------------------------------------
# Compatibility helper for joblib unpickle
# ---------------------------------------------------------------------
def _ravel(x):
    return np.ravel(x)

# ---------------------------------------------------------------------
# Optional .env
# ---------------------------------------------------------------------
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None

# ---------------------------------------------------------------------
# Optional OpenAI client (ProxyAPI OpenAI-compatible)
# ---------------------------------------------------------------------
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None

BASE_DIR = Path(__file__).resolve().parent

ARTIFACTS_DIR = BASE_DIR / "artifacts_stage7"
if not ARTIFACTS_DIR.exists():
    ARTIFACTS_DIR = BASE_DIR / "artifacts"

TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")

if load_dotenv is not None:
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        load_dotenv(str(env_path))

# ---------------------------------------------------------------------
# Helpers: load
# ---------------------------------------------------------------------
def _load_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if default is None:
        default = {}
    try:
        if not path.exists():
            return dict(default)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        app.logger.exception("Failed to load json %s: %s", path, e)
        return dict(default)


def _safe_joblib_load(path: Path):
    try:
        if not path.exists():
            return None
        return joblib.load(path)
    except Exception as e:
        app.logger.exception("Failed to load joblib %s: %s", path, e)
        return None


def _coerce_pdna_to_none(v: Any) -> Any:
    if v is pd.NA:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v

# ---------------------------------------------------------------------
# Introspection: expected columns
# ---------------------------------------------------------------------
def _normalize_cols(cols_sel: Any) -> List[str]:
    if cols_sel is None:
        return []
    if isinstance(cols_sel, str):
        return [cols_sel]
    if isinstance(cols_sel, (list, tuple, set, np.ndarray, pd.Index)):
        return [str(x) for x in list(cols_sel)]
    return []
  
def _has_vectorizer(obj: Any) -> bool:
    name = obj.__class__.__name__.lower()
    if "tfidfvectorizer" in name or "countvectorizer" in name:
        return True
    return False

def _guess_kind(transformer: Any, name: str) -> str:
    """
    kind: "num" | "cat" | "text"
    """
    lname = (name or "").lower()
    if "text" in lname or "tfidf" in lname:
        return "text"
    if "cat" in lname or "onehot" in lname:
        return "cat"
    if "num" in lname or "scaler" in lname:
        return "num"

    # transformer maybe Pipeline with steps
    steps = []
    if hasattr(transformer, "steps"):
        try:
            steps = [s for (_, s) in transformer.steps]
        except Exception:
            steps = []

    for s in steps:
        if _has_vectorizer(s):
            return "text"
        cls = s.__class__.__name__.lower()
        if "onehotencoder" in cls:
            return "cat"
        if "standardscaler" in cls or "minmaxscaler" in cls:
            return "num"

    # single трансформер
    if transformer is None:
        return "cat"
    if _has_vectorizer(transformer):
        return "text"
    cls = transformer.__class__.__name__.lower()
    if "onehotencoder" in cls:
        return "cat"
    return "num"


def _find_column_transformer(model: Any) -> Optional[Any]:
    """Seek ColumnTransformer inside Pipeline."""
    if model is None:
        return None
    # Pipeline?
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if step.__class__.__name__ == "ColumnTransformer":
                return step
    # сам ColumnTransformer?
    if model.__class__.__name__ == "ColumnTransformer":
        return model
    return None

def _model_input_spec(model: Any) -> Tuple[List[str], Dict[str, str]]:
    """
    Return:
      cols: list of expected columns
      kinds: dict col -> kind ("num"/"cat"/"text")
    """
    cols_set = set()
    kinds: Dict[str, str] = {}

    if model is None:
        return [], {}

    ct = _find_column_transformer(model)
    if ct is not None:
        tr_list = None
        if hasattr(ct, "transformers_"):
            tr_list = ct.transformers_
        elif hasattr(ct, "transformers"):
            tr_list = ct.transformers

        if tr_list:
            for name, trans, cols_sel in tr_list:
                if name == "remainder":
                    continue
                if isinstance(trans, str) and trans == "drop":
                    continue
                cols_norm = _normalize_cols(cols_sel)
                if not cols_norm:
                    continue
                kind = _guess_kind(trans, str(name))
                for c in cols_norm:
                    cols_set.add(c)
                    kinds[c] = kind

    # fallback: feature_names_in_
    if not cols_set and hasattr(model, "feature_names_in_"):
        try:
            for c in list(model.feature_names_in_):
                cols_set.add(str(c))
                kinds[str(c)] = "cat"
        except Exception:
            pass

    return sorted(cols_set), kinds


def _default_for_kind(kind: str) -> Any:
    if kind == "num":
        return 0.0
    return ""


def _coerce_value(kind: str, val: Any) -> Any:
    if kind == "num":
        if val in (None, "", "None"):
            return 0.0
        try:
            return float(val)
        except Exception:
            return 0.0
    # cat/text
    if val in (None, "None"):
        return ""
    return str(val)


def _build_text_all_from_row(row: Dict[str, Any]) -> str:
    parts = []
    for key in [
        "title_clean", "track_title", "venue", "city",
        "university", "country", "role",
        "format"
    ]:
        v = row.get(key, "")
        if v is None:
            continue
        s = str(v).strip()
        if s:
            parts.append(s)
    return " ".join(parts).strip()


def _apply_computed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add/count calculable columns, if they are inside df.
    """
    # has_tracks_clean = (tracks_count_clean > 0)
    if "has_tracks_clean" in df.columns:
        base = 0.0
        if "tracks_count_clean" in df.columns:
            try:
                base = float(df.loc[0, "tracks_count_clean"] or 0.0)
            except Exception:
                base = 0.0
        df.loc[0, "has_tracks_clean"] = 1.0 if base > 0 else 0.0

    # text_all = concat of important text fields
    if "text_all" in df.columns:
        row = df.iloc[0].to_dict()
        df.loc[0, "text_all"] = _build_text_all_from_row(row)

    return df


def _build_X_for_model(
    model: Any,
    raw: Dict[str, Any],
    fallback_schema: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    cols, kinds = _model_input_spec(model)

    if not cols:
        # fallback
        if not fallback_schema:
            fallback_schema = {}
        cols = list(fallback_schema.keys())
        kinds = dict(fallback_schema)

    row: Dict[str, Any] = {}
    for c in cols:
        kind = kinds.get(c, "cat")
        row[c] = _coerce_value(kind, raw.get(c, None))

    df = pd.DataFrame([row])

    # calculate derived features
    df = _apply_computed_columns(df)

    # ok if computed created new columns and model expects it
    # if model expects column and it's not appear, add default
    for c in cols:
        if c not in df.columns:
            df[c] = _default_for_kind(kinds.get(c, "cat"))

    return df


# ---------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------
METRICS_PATH = ARTIFACTS_DIR / "metrics_stage7.json"

SUBMISSION_MODEL_PATH = ARTIFACTS_DIR / "submission_decision_model.joblib"
SUBMISSION_THRESHOLD_PATH = ARTIFACTS_DIR / "submission_decision_threshold.json"

EVENT_MODEL_PATH = ARTIFACTS_DIR / "event_regcount_forecast_model.joblib"
EVENT_CONFIG_PATH = ARTIFACTS_DIR / "event_regcount_forecast_config.json"

FOCUS_RULE_PATH = ARTIFACTS_DIR / "focus_rule.json"

METRICS: Dict[str, Any] = _load_json(METRICS_PATH, default={})
SUBMISSION_THRESHOLD: Dict[str, Any] = _load_json(SUBMISSION_THRESHOLD_PATH, default={"thr": 0.65})
EVENT_FORECAST_CONFIG: Dict[str, Any] = _load_json(EVENT_CONFIG_PATH, default={})
FOCUS_RULE: Dict[str, Any] = _load_json(FOCUS_RULE_PATH, default={})

SUBMISSION_MODEL = _safe_joblib_load(SUBMISSION_MODEL_PATH)
EVENT_MODEL = _safe_joblib_load(EVENT_MODEL_PATH)

# fallback schemas
FALLBACK_EVENT_SCHEMA = EVENT_FORECAST_CONFIG.get("schema_cols")
if not isinstance(FALLBACK_EVENT_SCHEMA, dict) or not FALLBACK_EVENT_SCHEMA:
    FALLBACK_EVENT_SCHEMA = {
        "title_clean": "text",
        "city": "cat",
        "venue": "cat",
        "format": "cat",
        "is_online_flag": "num",
        "cost_is_free": "num",
        "cost_mentions_fee": "num",
        "event_month": "num",
        "event_dow": "num",
        "description_len": "num",
        "event_duration_days": "num",
        "tracks_count_clean": "num",
        "n_links": "num",
        "n_calendar": "num",
        "n_external": "num",
        "n_pages": "num",
        "deadline_days_before_event": "num",
        "has_tracks_clean": "num",
    }

FALLBACK_SUB_SCHEMA = {
    "text_all": "text",

    "title_clean": "text",
    "track_title": "text",

    "university": "cat",
    "role": "cat",
    "country": "cat",
    "city": "cat",
    "venue": "cat",
    "format": "cat",

    "is_online_flag": "num",
    "cost_is_free": "num",
    "cost_mentions_fee": "num",
    "event_month": "num",
    "event_dow": "num",
    "description_len": "num",

    "n_reviews": "num",
    "score_mean": "num",
    "score_std": "num",
    "score_min": "num",
    "score_max": "num",
    "conf_mean": "num",
}

# ---------------------------------------------------------------------
# ProxyAPI assistant
# ---------------------------------------------------------------------
def _get_ai_client() -> Optional[Any]:
    if OpenAI is None:
        return None
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if not api_key or not base_url:
        return None
    try:
        return OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        app.logger.exception("Failed to init OpenAI client: %s", e)
        return None

def _ask_assistant(prompt: str) -> str:
    client = _get_ai_client()
    if client is None:
        return "ИИ-ассистент не настроен. Проверьте .env (OPENAI_API_KEY / OPENAI_BASE_URL / MODEL_NAME)."

    model_name = os.environ.get("MODEL_NAME", "gpt-4o").strip() or "gpt-4o"

    system = (
        "Ты — помощник сервиса Science Events.\n"
        "Пиши коротко, по делу, человеческим языком.\n"
        "Если данных мало — скажи, какие поля лучше заполнить.\n"
        "Если это прогноз — напомни, что это ориентир.\n"
    )
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        msg = resp.choices[0].message
        return msg.content if msg and msg.content else ""
    except Exception as e:
        app.logger.exception("ProxyAPI request failed: %s", e)
        return "Не удалось получить ответ ассистента (ошибка запроса)."


# ---------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------
@app.errorhandler(Exception)
def handle_any_error(e):
    if isinstance(e, HTTPException):
        return e
    app.logger.exception("Unhandled error: %s", e)
    return "Internal Server Error", 500


@app.get("/favicon.ico")
def favicon():
    return ("", 204)


# ---------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------
@app.get("/")
def page_index():
    thr = float(SUBMISSION_THRESHOLD.get("thr", 0.65))
    return render_template(
        "index.html",
        metrics=METRICS,
        submission_threshold=thr,
        has_submission_model=SUBMISSION_MODEL is not None,
        has_event_model=EVENT_MODEL is not None,
    )


@app.get("/events")
def page_events():
    return render_template(
        "events.html",
        metrics=METRICS,
        has_event_model=EVENT_MODEL is not None,
        event_config=EVENT_FORECAST_CONFIG,
    )


@app.get("/submission")
def page_submission():
    thr = float(SUBMISSION_THRESHOLD.get("thr", 0.65))
    return render_template(
        "submission.html",
        metrics=METRICS,
        submission_threshold=thr,
        has_submission_model=SUBMISSION_MODEL is not None,
    )


@app.get("/focus")
def page_focus():
    return render_template(
        "focus.html",
        metrics=METRICS,
        focus_rule=FOCUS_RULE,
    )


@app.get("/methodology")
def page_methodology():
    return render_template(
        "methodology.html",
        metrics=METRICS,
        has_ai=(_get_ai_client() is not None),
    )


# ---------------------------------------------------------------------
# API: Events forecast
# ---------------------------------------------------------------------
@app.post("/events/predict")
def api_events_predict():
    if EVENT_MODEL is None:
        return jsonify({"ok": False, "error": "model_not_loaded"})

    raw_in = request.json or {}
    raw = {k: _coerce_pdna_to_none(v) for k, v in raw_in.items()}

    try:
        X = _build_X_for_model(EVENT_MODEL, raw, fallback_schema=FALLBACK_EVENT_SCHEMA)
        pred = EVENT_MODEL.predict(X)
        y_hat = float(np.asarray(pred).reshape(-1)[0])
        y_hat = max(0.0, y_hat)

        baseline = None
        for key in ["baseline", "baseline_mean", "train_mean", "reg_count_mean"]:
            if key in EVENT_FORECAST_CONFIG:
                try:
                    baseline = float(EVENT_FORECAST_CONFIG.get(key))
                    break
                except Exception:
                    baseline = None

        return jsonify({
            "ok": True,
            "reg_count_pred": y_hat,
            "baseline": baseline,
            "note": "Оценка ожидаемого числа регистраций (ориентир)"
        })
    except Exception as e:
        app.logger.exception("Event predict failed: %s", e)
        return jsonify({"ok": False, "error": "predict_failed", "detail": str(e)})


# ---------------------------------------------------------------------
# API: Submission predict
# ---------------------------------------------------------------------
@app.post("/api/submission/predict")
def api_submission_predict():
    raw_in = request.json or {}
    raw = {k: _coerce_pdna_to_none(v) for k, v in raw_in.items()}

    thr = float(SUBMISSION_THRESHOLD.get("thr", 0.65))

    # даже если модели нет — fallback, чтобы UI работал
    if SUBMISSION_MODEL is None:
        # лёгкая эвристика
        score_mean = float(raw.get("score_mean") or 0.0) if raw.get("score_mean") not in ("", None) else 0.0
        n_reviews = float(raw.get("n_reviews") or 0.0) if raw.get("n_reviews") not in ("", None) else 0.0
        conf_mean = float(raw.get("conf_mean") or 0.0) if raw.get("conf_mean") not in ("", None) else 0.0

        z = (score_mean - 5.5) * 1.2 + (n_reviews - 1.0) * 0.35 + (conf_mean - 3.0) * 0.15
        proba = 1.0 / (1.0 + math.exp(-z))
        label = int(proba >= thr)

        return jsonify({"ok": True, "proba": float(proba), "thr": thr, "label": label, "note": "fallback_rule"})

    try:
        X = _build_X_for_model(SUBMISSION_MODEL, raw, fallback_schema=FALLBACK_SUB_SCHEMA)

        # Если модель ожидает text_all — мы его уже собрали в _apply_computed_columns,
        # но на всякий случай:
        if "text_all" in X.columns and not str(X.loc[0, "text_all"] or "").strip():
            X.loc[0, "text_all"] = _build_text_all_from_row(X.iloc[0].to_dict())

        proba = SUBMISSION_MODEL.predict_proba(X)[:, 1]
        p = float(np.asarray(proba).reshape(-1)[0])
        label = int(p >= thr)

        return jsonify({"ok": True, "proba": p, "thr": thr, "label": label, "note": "model"})
    except Exception as e:
        app.logger.exception("Submission predict failed: %s", e)
        return jsonify({"ok": False, "error": "predict_failed", "detail": str(e)})


# backward compatibility
@app.post("/submission/predict")
def api_submission_predict_compat():
    return api_submission_predict()


# ---------------------------------------------------------------------
# API: Focus rule
# ---------------------------------------------------------------------
@app.post("/api/focus/check")
def api_focus_check():
    raw = request.json or {}
    city = str(raw.get("city", "") or "")
    venue = str(raw.get("venue", "") or "")
    title = str(raw.get("title_clean", "") or "")

    venue_key = str(FOCUS_RULE.get("venue_keyword", "МУ имени С. Ю. Витте"))
    city_key = str(FOCUS_RULE.get("city_keyword", "Москва"))

    is_focus = (venue_key.lower() in venue.lower()) and (city_key.lower() in city.lower())

    reasons = []
    reasons.append("площадка совпадает" if venue_key.lower() in venue.lower() else "площадка не совпадает")
    reasons.append("город совпадает" if city_key.lower() in city.lower() else "город не совпадает")
    if title.strip():
        reasons.append("название учтено как контекст")

    return jsonify({
        "ok": True,
        "is_focus": bool(is_focus),
        "reason": ", ".join(reasons),
        "rule": {"venue_keyword": venue_key, "city_keyword": city_key},
    })

@app.post("/api/focus/check_by_id")
def api_focus_check_by_id():
    payload = request.json or {}
    event_id = str(payload.get("event_id", "") or "").strip()
    if not event_id:
        return jsonify({"ok": False, "error": "empty_event_id"})
    return jsonify({"ok": False, "error": "events_index_not_found", "detail": "Нет индекса событий в artifacts (events_index.*)"})

# backward compatibility
@app.post("/focus/check")
def api_focus_check_compat():
    return api_focus_check()

# ---------------------------------------------------------------------
# API: Assistant report
# ---------------------------------------------------------------------
@app.post("/api/assistant/report")
def api_assistant_report():
    payload = request.json or {}

    ctx = payload.get("context", "")
    if isinstance(ctx, (dict, list)):
        ctx_text = json.dumps(ctx, ensure_ascii=False, indent=2)
    else:
        ctx_text = str(ctx or "").strip()

    if not ctx_text:
        return jsonify({"ok": False, "error": "empty_context"})

    prompt = (
        "Поясни результат пользователю простыми словами (1–2 абзаца):\n"
        f"{ctx_text[:6000]}\n\n"
        "Скажи: что означает результат, что повлияло, и что стоит проверить/дозаполнить."
    )

    report = _ask_assistant(prompt)
    return jsonify({"ok": True, "report": report})

# ---------------------------------------------------------------------
# Local run
# ---------------------------------------------------------------------
if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    app.run(host=host, port=port, debug=False)
