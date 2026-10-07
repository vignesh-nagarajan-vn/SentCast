"""Sentiment scorers (PLAN.md section 4). Each maps a list of texts to scores; higher is more positive."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import INTERIM, RAW, load_config
from .data import FILES

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
LABELS = ["negative", "neutral", "positive"]


def normalise(text: str) -> str:
    """Lower-case and reduce to letters, digits and single spaces, like a URL headline."""
    return _NON_ALNUM.sub(" ", text.lower()).strip()


# --- lexicons -------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _lm_lists() -> tuple[frozenset, frozenset]:
    lm = pd.read_csv(RAW / FILES["lm"], usecols=["Word", "Negative", "Positive"], keep_default_na=False)
    words = lm["Word"].astype(str).str.upper()
    return frozenset(words[lm["Positive"] > 0]), frozenset(words[lm["Negative"] > 0])


def score_lm(texts: list[str]) -> np.ndarray:
    """Loughran-McDonald net tone: (pos - neg) / (pos + neg); 0 if no listed word."""
    pos, neg = _lm_lists()
    out = np.zeros(len(texts))
    for i, t in enumerate(texts):
        tokens = normalise(t).upper().split()
        p = sum(1 for w in tokens if w in pos)
        n = sum(1 for w in tokens if w in neg)
        if p + n:
            out[i] = (p - n) / (p + n)
    return out


def score_vader(texts: list[str]) -> np.ndarray:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    return np.array([sia.polarity_scores(t)["compound"] for t in texts])


# --- FinBERT --------------------------------------------------------------------------

def score_finbert(texts: list[str], progress: bool = True) -> np.ndarray:
    """P(positive) - P(negative) from ProsusAI/finbert."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cfg = load_config()["scorers"]["finbert"]
    torch.manual_seed(load_config()["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model"], revision=cfg["revision"]).to(device).eval()
    label2id = {v.lower(): k for k, v in model.config.id2label.items()}
    order = np.argsort([len(t) for t in texts])
    out = np.zeros(len(texts))
    bs = cfg["batch_size"]
    with torch.inference_mode():
        for start in range(0, len(texts), bs):
            idx = order[start:start + bs]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=cfg["max_length"],
                      return_tensors="pt").to(device)
            prob = torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy()
            out[idx] = prob[:, label2id["positive"]] - prob[:, label2id["negative"]]
            if progress and (start // bs) % 200 == 0:
                print(f"  finbert {start}/{len(texts)}", flush=True)
    return out


def finbert_labels(texts: list[str]) -> list[str]:
    """Predicted class per text, for the labelled-data check in E02."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cfg = load_config()["scorers"]["finbert"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model"], revision=cfg["revision"]).to(device).eval()
    out: list[str] = []
    with torch.inference_mode():
        for start in range(0, len(texts), cfg["batch_size"]):
            enc = tok(texts[start:start + cfg["batch_size"]], padding=True, truncation=True,
                      max_length=cfg["max_length"], return_tensors="pt").to(device)
            pred = model(**enc).logits.argmax(dim=-1).cpu().numpy()
            out.extend(model.config.id2label[int(p)].lower() for p in pred)
    return out


# --- simple trained classifier --------------------------------------------------------

def load_phrasebank() -> pd.DataFrame:
    cfg = load_config()["scorers"]["clf"]
    z = zipfile.ZipFile(io.BytesIO((RAW / FILES["phrasebank"]).read_bytes()))
    rows = []
    for line in z.read(cfg["train_file"]).decode("latin-1").splitlines():
        if "@" in line:
            text, label = line.rsplit("@", 1)
            rows.append((text.strip(), label.strip().lower()))
    return pd.DataFrame(rows, columns=["text", "label"])


def _clf_pipeline():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    cfg = load_config()["scorers"]["clf"]
    return make_pipeline(
        TfidfVectorizer(preprocessor=normalise, ngram_range=tuple(cfg["ngram_range"]), min_df=cfg["min_df"],
                        sublinear_tf=True),
        LogisticRegression(C=cfg["C"], max_iter=2000),
    )


@lru_cache(maxsize=1)
def _clf_fitted():
    pb = load_phrasebank()
    return _clf_pipeline().fit(pb["text"].tolist(), pb["label"].tolist())


def score_clf(texts: list[str]) -> np.ndarray:
    """P(positive) - P(negative) from the TF-IDF logistic regression."""
    model = _clf_fitted()
    prob = model.predict_proba(texts)
    classes = list(model.classes_)
    return prob[:, classes.index("positive")] - prob[:, classes.index("negative")]


def clf_cross_validation() -> dict:
    """Stratified k-fold accuracy and macro-F1 of the classifier on the Financial PhraseBank."""
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold

    cfg = load_config()
    pb = load_phrasebank()
    X, y = np.array(pb["text"]), np.array(pb["label"])
    skf = StratifiedKFold(n_splits=cfg["scorers"]["clf"]["cv_folds"], shuffle=True, random_state=cfg["seed"])
    acc, f1 = [], []
    for tr, te in skf.split(X, y):
        m = _clf_pipeline().fit(X[tr].tolist(), y[tr].tolist())
        pred = m.predict(X[te].tolist())
        acc.append(accuracy_score(y[te], pred))
        f1.append(f1_score(y[te], pred, average="macro"))
    return {"n": len(pb), "folds": len(acc), "accuracy_mean": float(np.mean(acc)), "accuracy_sd": float(np.std(acc)),
            "macro_f1_mean": float(np.mean(f1)), "macro_f1_sd": float(np.std(f1)),
            "class_counts": pb["label"].value_counts().to_dict()}


def labels_from_scores(scores: np.ndarray, threshold: float = 0.05) -> list[str]:
    return ["positive" if s > threshold else "negative" if s < -threshold else "neutral" for s in scores]


# --- LLM scorer (runs only with an API key) --------------------------------------------

# USD per million tokens (input, output). Needed to enforce the cost cap; a model that is
# not listed here is refused rather than run without a cap.
LLM_PRICES_USD_PER_MTOK = {"claude-haiku-4-5": (1.00, 5.00), "claude-sonnet-5-5": (2.00, 10.00)}

_LLM_SYSTEM = (
    "You label the sentiment of news headlines about the US economy and financial markets. "
    "The headlines are lower-case and without punctuation. For each headline give one integer: "
    "-2 very negative, -1 negative, 0 neutral or unclear, 1 positive, 2 very positive, judged as news "
    "for the economic or market outlook. Judge only the wording of the headline. Do not use any "
    "knowledge of events after it was written. Return the labels in the order given."
)
_LLM_SCHEMA = {
    "type": "object",
    "properties": {"labels": {"type": "array", "items": {"type": "integer", "enum": [-2, -1, 0, 1, 2]}}},
    "required": ["labels"],
    "additionalProperties": False,
}


def llm_available() -> bool:
    return bool(os.environ.get(load_config()["scorers"]["llm"]["env_key"]))


_LLM_MAX_TOKENS = 1024
_LLM_USAGE_KEYS = ("requests", "input_tokens", "output_tokens", "cost_usd", "discarded_batches", "errors")


def llm_configs() -> list[dict]:
    """Settings of the planned LLM scorer, then of the second model if one is configured."""
    cfg = dict(load_config()["scorers"]["llm"])
    second = cfg.pop("second", None)
    return [cfg] + ([{**cfg, **second}] if second else [])


def _llm_cache_paths(cfg: dict):
    """Label cache and running usage total, keyed by model, prompt and effort so a changed request starts afresh."""
    key = _LLM_SYSTEM + json.dumps(_LLM_SCHEMA, sort_keys=True) + str(cfg.get("effort"))
    tag = hashlib.sha256(key.encode()).hexdigest()[:8]
    return INTERIM / f"llm_{cfg['model']}_{tag}.parquet", INTERIM / f"llm_{cfg['model']}_{tag}.usage.json"


def _llm_request(client, cfg: dict, batch: list[str]) -> dict:
    """One request for one batch of headlines. Returns token counts and the labels, or None if unusable.

    No fallback model is requested: a batch the model declines is discarded and counted, so every label
    in an index comes from the one model named in the config.
    """
    import anthropic

    prompt = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(batch))
    output_config = {"format": {"type": "json_schema", "schema": _LLM_SCHEMA}}
    if cfg.get("effort"):                        # models that always think: effort sets how much
        output_config["effort"] = cfg["effort"]
    try:
        resp = client.messages.create(
            model=cfg["model"], max_tokens=cfg.get("max_tokens", _LLM_MAX_TOKENS), system=_LLM_SYSTEM,
            messages=[{"role": "user", "content": prompt}], output_config=output_config,
        )
    except (anthropic.BadRequestError, anthropic.AuthenticationError, anthropic.PermissionDeniedError,
            anthropic.NotFoundError):
        raise                                    # a wrong key, model or request fails every batch: stop
    except anthropic.RateLimitError:
        return {"error": True}
    except anthropic.APIStatusError:
        return {"error": True}
    except anthropic.APIConnectionError:
        return {"error": True}
    u = resp.usage
    out = {"input_tokens": u.input_tokens + (u.cache_creation_input_tokens or 0) + (u.cache_read_input_tokens or 0),
           "output_tokens": u.output_tokens, "labels": None}
    if resp.stop_reason != "end_turn":
        return out
    try:
        labels = json.loads(next(b.text for b in resp.content if b.type == "text"))["labels"]
    except (StopIteration, ValueError, KeyError):
        return out
    if len(labels) == len(batch):
        out["labels"] = labels
    return out


def score_llm(texts: list[str], cfg: dict | None = None) -> tuple[np.ndarray, dict]:
    """Label texts with one configured LLM (the planned one unless cfg names another). Returns (scores in [-1, 1], nan where unlabelled; usage report).

    Each distinct text is labelled once. Labels and the running cost are kept under data/interim, so a
    rerun asks only for what is missing and the cap applies to the total spent across runs. Stops before
    any round of requests whose worst-case cost would take spending past the cap.
    """
    import anthropic

    cfg = llm_configs()[0] if cfg is None else cfg
    if cfg["model"] not in LLM_PRICES_USD_PER_MTOK:
        raise RuntimeError(f"no price on file for {cfg['model']}; cannot enforce the cost cap")
    price_in, price_out = LLM_PRICES_USD_PER_MTOK[cfg["model"]]
    cap = float(cfg["max_cost_usd"])
    cache_path, usage_path = _llm_cache_paths(cfg)
    max_tokens = cfg.get("max_tokens", _LLM_MAX_TOKENS)
    cache = pd.read_parquet(cache_path).set_index("text")["score"].to_dict() if cache_path.exists() else {}
    usage = json.loads(usage_path.read_text()) if usage_path.exists() else {}
    usage = {k: usage.get(k, 0) for k in _LLM_USAGE_KEYS}
    stopped = False

    def save() -> None:
        INTERIM.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"text": list(cache), "score": list(cache.values())}).to_parquet(cache_path, index=False)
        usage_path.write_text(json.dumps(usage, indent=1))

    todo = [t for t in pd.unique(pd.Series(texts, dtype=object)) if t not in cache]
    bs, workers = cfg["batch_size"], cfg["workers"]
    batches = [todo[i:i + bs] for i in range(0, len(todo), bs)]
    client = anthropic.Anthropic(max_retries=8)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for r, start in enumerate(range(0, len(batches), workers)):
                wave = batches[start:start + workers]
                # characters / 2 overstates the token count of English text, so this bound is safe
                worst = sum(((sum(len(t) + 6 for t in b) + len(_LLM_SYSTEM)) / 2 * price_in
                             + max_tokens * price_out) / 1e6 for b in wave)
                if usage["cost_usd"] + worst > cap:
                    stopped = True
                    break
                for batch, res in zip(wave, pool.map(lambda b: _llm_request(client, cfg, b), wave)):
                    if res.get("error"):
                        usage["errors"] += 1
                        continue
                    usage["requests"] += 1
                    usage["input_tokens"] += res["input_tokens"]
                    usage["output_tokens"] += res["output_tokens"]
                    usage["cost_usd"] += (res["input_tokens"] * price_in + res["output_tokens"] * price_out) / 1e6
                    if res["labels"] is None:
                        usage["discarded_batches"] += 1
                        continue
                    cache.update(zip(batch, (v / 2.0 for v in res["labels"])))
                if r % 50 == 49:
                    save()
                    print(f"   {cfg['model']}: {len(cache)} texts labelled, USD {usage['cost_usd']:.2f}", flush=True)
    finally:
        save()
    out = np.array([cache.get(t, np.nan) for t in texts], dtype=float)
    report = {"model": cfg["model"], "cap_usd": cap, **usage, "cost_usd": round(usage["cost_usd"], 6),
              "texts": len(texts), "labelled": int(np.isfinite(out).sum()), "requested_this_run": len(todo),
              "stopped_at_cap": stopped}
    return out, report


SCORERS = {"lm": score_lm, "vader": score_vader, "finbert": score_finbert, "clf": score_clf}
