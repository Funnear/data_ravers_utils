"""
text_features.py

Reusable utilities and sklearn-compatible transformer(s) for extracting
hand-crafted text features (length, punctuation, POS ratios, etc.).

This module is project-agnostic so it can be reused across NLP projects.
"""

from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import spacy
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------------------------
# Punctuation patterns (Spanish-aware)
# ---------------------------------------------------------------------------

PUNCT_PATTERNS = {
    "num_commas": r",",
    "num_periods": r"\.",
    "num_semicolons": r";",
    "num_colons": r":",
    "num_qmarks": r"\?|¿",          # ? and inverted ¿
    "num_emarks": r"!|¡",           # ! and inverted ¡
    "num_ellipses": r"\.{3,}",      # sequences of 3+ dots
    "num_dashes": r"—|–|-",         # em dash, en dash, hyphen
    "num_quotes": r"['\"“”‘’«»]",   # various quote types
    "num_parens": r"[()]",          # parentheses
}


def _count_pattern(text: str, pattern: str) -> int:
    """Safe regex count that handles non-string input."""
    if not isinstance(text, str):
        return 0
    return len(re.findall(pattern, text))

# ---------------------------------------------------------------------------
# SpaCy feature columns
# ---------------------------------------------------------------------------

SPACY_FEATURE_COLUMNS = [
    "num_sents",
    "avg_tokens_per_sent",
    "verb_ratio",
    "adj_ratio",
    "adv_ratio",
    "pron_ratio",
    "num_ents",
    "num_ent_person",
    "num_ent_org",
    "num_ent_loc",
]

# ---------------------------------------------------------------------------
# SpaCy loading helper
# ---------------------------------------------------------------------------

def load_spanish_model(model_name: str = "es_core_news_sm") -> spacy.Language:
    """
    Load and return a Spanish spaCy model.

    Parameters
    ----------
    model_name : str
        Name of the spaCy model to load.

    Raises
    ------
    OSError
        If the model is not installed.

    """
    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise OSError(
            f"SpaCy model '{model_name}' is not installed. "
            f"Run: python -m spacy download {model_name}"
        ) from exc


# ---------------------------------------------------------------------------
# Core sklearn-style transformer
# ---------------------------------------------------------------------------

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts hand-crafted features from raw text.

    Features implemented so far
    ---------------------------
    Length-based
    ~~~~~~~~~~~~
    - word_count
    - char_count
    - log_length
    - is_outlier_length  (based on word_count quantile from training data)

    Punctuation-based
    ~~~~~~~~~~~~~~~~~
    - num_commas, num_periods, num_semicolons, num_colons,
      num_qmarks, num_emarks, num_ellipses, num_dashes,
      num_quotes, num_parens, num_punct_total

    SpaCy-based (if use_spacy=True)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - num_sents
    - avg_tokens_per_sent
    - verb_ratio, adj_ratio, adv_ratio, pron_ratio
    - num_ents, num_ent_person, num_ent_org, num_ent_loc
    """

    def __init__(
        self,
        use_spacy: bool = True,
        spacy_model: str = "es_core_news_sm",
        length_outlier_quantile: float = 0.95,
    ) -> None:
        self.use_spacy = use_spacy
        self.spacy_model = spacy_model
        self.length_outlier_quantile = length_outlier_quantile

        self._nlp: Optional[spacy.Language] = None
        self.length_outlier_threshold_: Optional[float] = None

    # ------------------------------------------------------------------ #
    # sklearn interface
    # ------------------------------------------------------------------ #

    def fit(self, X: Iterable[str], y: Optional[Iterable] = None):
        """
        Fit the transformer.

        - Optionally loads the spaCy model (for future features).
        - Computes the sentence-length outlier threshold from training data.
        """
        texts = list(X)

        if self.use_spacy and self._nlp is None:
            self._nlp = load_spanish_model(self.spacy_model)

        # compute length-based outlier threshold
        if self.length_outlier_quantile is not None:
            word_counts = pd.Series(texts).astype(str).str.split().str.len()
            self.length_outlier_threshold_ = float(
                word_counts.quantile(self.length_outlier_quantile)
            )

        return self

    def transform(self, X: Iterable[str]) -> pd.DataFrame:
        """
        Transform an iterable of texts into a feature DataFrame.

        Parameters
        ----------
        X : iterable of str
            Raw text samples.

        Returns
        -------
        pandas.DataFrame
            DataFrame with one row per input text and feature columns.
        """
        texts = list(X)
        df = pd.DataFrame({"text": texts}).astype({"text": "string"})

        # ------------------------------------------------------------------
        # Length-based features
        # ------------------------------------------------------------------
        df["word_count"] = df["text"].str.split().str.len()
        df["char_count"] = df["text"].str.len()
        df["log_length"] = np.log1p(df["word_count"])

        if self.length_outlier_quantile is not None and self.length_outlier_threshold_ is not None:
            df["is_outlier_length"] = df["word_count"] > self.length_outlier_threshold_
        else:
            df["is_outlier_length"] = False

        # ------------------------------------------------------------------
        # Punctuation-based features
        # ------------------------------------------------------------------
        for col, pattern in PUNCT_PATTERNS.items():
            df[col] = df["text"].apply(lambda s: _count_pattern(s, pattern))

        df["num_punct_total"] = df[list(PUNCT_PATTERNS.keys())].sum(axis=1)

        # ------------------------------------------------------------------
        # SpaCy-based features (optional)
        # ------------------------------------------------------------------
        if self.use_spacy and self._nlp is not None:
            records = []

            for doc in self._nlp.pipe(df["text"].tolist(), batch_size=64):
                tokens = [t for t in doc if not t.is_space]
                n_tokens = len(tokens) if tokens else 1

                sents = list(doc.sents)
                n_sents = len(sents) if sents else 1

                # POS counts
                num_verbs = sum(t.pos_ == "VERB" for t in tokens)
                num_adjs  = sum(t.pos_ == "ADJ"  for t in tokens)
                num_advs  = sum(t.pos_ == "ADV"  for t in tokens)
                num_pron  = sum(t.pos_ == "PRON" for t in tokens)

                # Named entities
                ents = list(doc.ents)
                num_ents = len(ents)
                num_ent_person = sum(ent.label_ == "PER" for ent in ents)
                num_ent_org    = sum(ent.label_ == "ORG" for ent in ents)
                num_ent_loc    = sum(ent.label_ in {"LOC", "GPE"} for ent in ents)

                records.append({
                    "num_sents": n_sents,
                    "avg_tokens_per_sent": n_tokens / n_sents,
                    "verb_ratio": num_verbs / n_tokens,
                    "adj_ratio":  num_adjs  / n_tokens,
                    "adv_ratio":  num_advs  / n_tokens,
                    "pron_ratio": num_pron  / n_tokens,
                    "num_ents": num_ents,
                    "num_ent_person": num_ent_person,
                    "num_ent_org":    num_ent_org,
                    "num_ent_loc":    num_ent_loc,
                })

            spacy_df = pd.DataFrame(records, index=df.index)
            df = pd.concat([df, spacy_df], axis=1)

        else:
            # If spaCy is disabled or not loaded, fill spaCy feature columns with zeros
            for col in SPACY_FEATURE_COLUMNS:
                df[col] = 0.0

        # We drop the raw text column here; the transformer is for numeric features
        # Drop text column
        out = df.drop(columns=["text"])

        # Ensure numeric dtype (important for sparse hstack!)
        out = out.astype(float)

        # Return numpy array
        return out.to_numpy()
    
    def get_feature_names(self):
        """Return feature names in the same order as transform()."""

        names = [
            # length-based features
            "word_count",
            "char_count",
            "log_length",
            "is_outlier_length",
        ]

        # punctuation features (10 columns from PUNCT_PATTERNS)
        names.extend(list(PUNCT_PATTERNS.keys()))

        # total punctuation
        names.append("num_punct_total")

        # spaCy features (10 more)
        if self.use_spacy:
            names.extend(SPACY_FEATURE_COLUMNS)
        else:
            # even when spaCy disabled, the transform() still adds them as zeros
            names.extend(SPACY_FEATURE_COLUMNS)

        return names

