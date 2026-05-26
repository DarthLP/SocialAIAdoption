"""
Script summary:
Wordfish Poisson scaling (Slapin & Proksch 2008) for subreddit×bin political documents.

Functionality:
- Event-time bins anchored at ban date t* (daily or 7-day blocks from t*).
- DTM construction via political_lexicon.tokenize with stopwords and token pruning.
- WordfishScaler (adapted from SemScale wfcode/scaler.py, pure NumPy).
- Panel helpers: center_lang_pre, extremity, dispersion, axis words.

How to apply/run:
- Used by scripts/diagnostics/prepare_wordfish.py; not a standalone CLI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from src.political_lexicon import tokenize

ITALY_TOPIC_FAMILIES = frozenset({"it_political", "it_others"})
PURE_NUMERIC_RE = re.compile(r"^\d+$")
PROVENANCE_HEADER_RE = re.compile(r"^#")


@dataclass
class WordfishFitResult:
    """Function summary: container for one Wordfish fit."""

    doc_ids: List[str]
    theta: np.ndarray
    beta: Dict[str, float]
    vocabulary: List[str]
    objective_final: float
    objective_history: List[float]
    converged: bool
    sign_flipped: bool
    one_sided_beta: bool = False
    beta_neg_fraction: float = float("nan")
    beta_pos_fraction: float = float("nan")


@dataclass
class DocumentRecord:
    """Function summary: one subreddit×bin or author×bin political document."""

    doc_id: str
    subreddit: str
    topic_family: str
    primary_lexicon: str
    bin_start: str
    time_bin: str
    n_days_in_bin: int
    n_tokens: int
    tokens: List[str] = field(repr=False)
    author: str = ""
    n_words_proxy: int = 0


def parse_anchor_date(anchor: str) -> date:
    """Function summary: parse YYYY-MM-DD ban anchor date.

    Parameters:
    - anchor: ISO date string.

    Returns:
    - date object.
    """
    return datetime.strptime(anchor.strip()[:10], "%Y-%m-%d").date()


def days_from_anchor(date_str: str, anchor: date) -> int:
    """Function summary: calendar days from anchor to date_utc.

    Parameters:
    - date_str: YYYY-MM-DD.
    - anchor: ban anchor date.

    Returns:
    - Integer day offset (negative = pre-ban).
    """
    d = datetime.strptime(date_str.strip()[:10], "%Y-%m-%d").date()
    return (d - anchor).days


def bin_start_for_day(date_str: str) -> str:
    """Function summary: daily bin label equals calendar date.

    Parameters:
    - date_str: YYYY-MM-DD.

    Returns:
    - bin_start string.
    """
    return date_str.strip()[:10]


def bin_start_for_week(date_str: str, anchor: date, weekly_days: int = 7) -> str:
    """Function summary: left edge of event-week bin containing date_str.

    Parameters:
    - date_str: YYYY-MM-DD.
    - anchor: t* ban date (bin boundary).
    - weekly_days: block width in days.

    Returns:
    - bin_start as YYYY-MM-DD (left edge of [k*W, (k+1)*W) block from t*).
    """
    offset = days_from_anchor(date_str, anchor)
    bin_idx = offset // weekly_days
    left = anchor + timedelta(days=int(bin_idx) * weekly_days)
    return left.isoformat()


def load_stopwords(path: Path) -> Set[str]:
    """Function summary: load lowercase stopwords from a text file (skip # headers).

    Parameters:
    - path: stopwords file path.

    Returns:
    - Set of stopword strings.
    """
    if not path.is_file():
        return set()
    out: Set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or PROVENANCE_HEADER_RE.match(line):
            continue
        out.add(line.lower())
    return out


def filter_token(tok: str, min_token_len: int) -> bool:
    """Function summary: keep token for DTM if alphabetic and long enough.

    Parameters:
    - tok: raw token.
    - min_token_len: minimum length.

    Returns:
    - True if token should be kept.
    """
    if len(tok) < min_token_len:
        return False
    if PURE_NUMERIC_RE.match(tok):
        return False
    if not any(ch.isalpha() for ch in tok):
        return False
    return True


def tokenize_document(
    text: str,
    stopwords: Set[str],
    min_token_len: int,
) -> List[str]:
    """Function summary: tokenize and prune one document body.

    Parameters:
    - text: raw comment concatenation.
    - stopwords: lowercase stopword set.
    - min_token_len: minimum token length.

    Returns:
    - List of filtered tokens.
    """
    return [
        t
        for t in tokenize(text)
        if filter_token(t, min_token_len) and t not in stopwords
    ]


def build_vocabulary_and_matrix(
    doc_tokens: Sequence[Sequence[str]],
    min_doc_freq: int,
    top_freq_drop_n: int,
    max_vocab_terms: int,
) -> Tuple[np.ndarray, List[str]]:
    """Function summary: build document-term count matrix with vocabulary pruning.

    Parameters:
    - doc_tokens: per-document token lists.
    - min_doc_freq: minimum document frequency for a term.
    - top_freq_drop_n: drop this many highest global-frequency terms.
    - max_vocab_terms: cap vocabulary size after pruning.

    Returns:
    - Tuple (occurrences matrix n_docs×n_terms, vocabulary list).
    """
    n_docs = len(doc_tokens)
    if n_docs == 0:
        return np.zeros((0, 0), dtype=np.float64), []

    doc_freq: Dict[str, int] = {}
    term_total: Dict[str, int] = {}
    for tokens in doc_tokens:
        seen: Set[str] = set()
        for t in tokens:
            term_total[t] = term_total.get(t, 0) + 1
            if t not in seen:
                doc_freq[t] = doc_freq.get(t, 0) + 1
                seen.add(t)

    vocab = [t for t, df in doc_freq.items() if df >= min_doc_freq]
    if not vocab:
        return np.zeros((n_docs, 0), dtype=np.float64), []

    vocab.sort(key=lambda w: (-term_total[w], w))
    if top_freq_drop_n > 0 and len(vocab) > top_freq_drop_n:
        drop = set(vocab[:top_freq_drop_n])
        vocab = [w for w in vocab if w not in drop]
    if max_vocab_terms > 0 and len(vocab) > max_vocab_terms:
        vocab = vocab[:max_vocab_terms]

    word_to_idx = {w: i for i, w in enumerate(vocab)}
    mat = np.zeros((n_docs, len(vocab)), dtype=np.float64)
    for i, tokens in enumerate(doc_tokens):
        for t in tokens:
            j = word_to_idx.get(t)
            if j is not None:
                mat[i, j] += 1.0
    return mat, vocab


class WordfishScaler:
    """Function summary: Wordfish Poisson scaling (adapted from SemScale, pure NumPy)."""

    def __init__(self, occurrences: np.ndarray) -> None:
        """Function summary: initialize scaler from document-term counts.

        Parameters:
        - occurrences: n_docs × n_words count matrix.
        """
        self.occurrences = np.asarray(occurrences, dtype=np.float64)
        self.num_docs, self.num_words = self.occurrences.shape
        self.alpha_docs = np.zeros(self.num_docs)
        self.theta_docs = np.zeros(self.num_docs)
        self.beta_words = np.zeros(self.num_words)
        self.psi_words = np.zeros(self.num_words)
        self.log_expectations = np.zeros((self.num_docs, self.num_words))
        self.objective_history: List[float] = []

    def initialize(self) -> None:
        """Function summary: SVD-based initialization of alpha, theta, beta, psi."""
        if self.num_docs == 0 or self.num_words == 0:
            return
        avg = np.average(self.occurrences, axis=0)
        avg = np.maximum(avg, 1e-8)
        self.psi_words = np.log(avg)
        counts = np.sum(self.occurrences, axis=1)
        counts = np.maximum(counts, 1e-8)
        self.alpha_docs = np.log(counts / counts[0])
        # SemScale: SVD on transposed log-occurrence matrix (words × docs).
        matrix = (
            np.log(np.maximum(self.occurrences, 1e-8)).T
            - np.repeat(self.psi_words[:, np.newaxis], self.num_docs, axis=1)
            - np.repeat(self.alpha_docs[np.newaxis, :], self.num_words, axis=0)
        )
        u, _s, vt = np.linalg.svd(matrix, full_matrices=False)
        self.beta_words = u[:, 0]
        self.theta_docs = vt[0, :]

    def normalize_positions(self) -> None:
        """Function summary: fix alpha[0]=0 and standardize theta across documents."""
        if self.num_docs == 0:
            return
        self.alpha_docs[0] = 0.0
        mu = float(np.mean(self.theta_docs))
        sd = float(np.std(self.theta_docs))
        if sd > 0:
            self.theta_docs = (self.theta_docs - mu) / sd

    def objective(self) -> float:
        """Function summary: negative log-likelihood of Poisson Wordfish model.

        Returns:
        - Scalar objective (lower is better).
        """
        self.log_expectations = self._log_expectation()
        exp_le = np.exp(np.clip(self.log_expectations, -30.0, 30.0))
        ll = np.sum(self.occurrences * self.log_expectations - exp_le)
        return float(-ll)

    def _log_expectation(self) -> np.ndarray:
        """Function summary: log expected counts per doc-word."""
        return (
            self.alpha_docs[:, np.newaxis]
            + self.psi_words[np.newaxis, :]
            + np.outer(self.theta_docs, self.beta_words)
        )

    def gradients_words(self) -> Tuple[np.ndarray, np.ndarray]:
        """Function summary: beta and psi gradients."""
        exp_le = np.exp(np.clip(self.log_expectations, -30.0, 30.0))
        psi_grads = np.sum(exp_le - self.occurrences, axis=0)
        beta_grads = np.sum(
            (exp_le - self.occurrences)
            * self.theta_docs[:, np.newaxis],
            axis=0,
        )
        return beta_grads, psi_grads

    def gradients_docs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Function summary: alpha and theta gradients."""
        exp_le = np.exp(np.clip(self.log_expectations, -30.0, 30.0))
        alpha_grads = np.sum(exp_le - self.occurrences, axis=1)
        theta_grads = np.sum(
            (exp_le - self.occurrences) * self.beta_words[np.newaxis, :],
            axis=1,
        )
        return alpha_grads, theta_grads

    def train(
        self,
        learning_rate: float,
        num_iters: int,
        check_convergence: bool = True,
        min_improvement: float = 1.0e-4,
    ) -> bool:
        """Function summary: gradient descent training loop.

        Parameters:
        - learning_rate: step size.
        - num_iters: maximum iterations.
        - check_convergence: compare last two logged objectives.
        - min_improvement: required objective decrease at end.

        Returns:
        - True if convergence criterion met.
        """
        self.normalize_positions()
        self.objective_history = [self.objective()]
        plateau = 0
        for i in range(num_iters):
            alpha_grads, theta_grads = self.gradients_docs()
            self.alpha_docs -= alpha_grads * learning_rate / max(self.num_words, 1)
            self.theta_docs -= theta_grads * learning_rate / max(self.num_words, 1)
            self.normalize_positions()

            beta_grads, psi_grads = self.gradients_words()
            self.beta_words -= beta_grads * learning_rate / max(self.num_docs, 1)
            self.psi_words -= psi_grads * learning_rate / max(self.num_docs, 1)

            if (i + 1) % 100 == 0 or i == num_iters - 1:
                obj = self.objective()
                self.objective_history.append(obj)
                if len(self.objective_history) >= 2:
                    prev_obj = self.objective_history[-2]
                    if (prev_obj - obj) < min_improvement:
                        plateau += 1
                    else:
                        plateau = 0
                    if check_convergence and plateau >= 3:
                        break

        converged = True
        if check_convergence and len(self.objective_history) >= 2:
            prev_obj = self.objective_history[-2]
            final_obj = self.objective_history[-1]
            converged = (prev_obj - final_obj) >= min_improvement
        return converged


def fit_wordfish(
    occurrences: np.ndarray,
    vocabulary: List[str],
    doc_ids: List[str],
    train_iters: int,
    learning_rate: float,
    convergence_cfg: Dict[str, Any],
) -> WordfishFitResult:
    """Function summary: run full Wordfish fit on a DTM.

    Parameters:
    - occurrences: count matrix.
    - vocabulary: term list aligned with columns.
    - doc_ids: document identifiers per row.
    - train_iters: max iterations.
    - learning_rate: step size.
    - convergence_cfg: check_final_objective, min_objective_improvement.

    Returns:
    - WordfishFitResult with theta and beta.
    """
    scaler = WordfishScaler(occurrences)
    scaler.initialize()
    check = bool(convergence_cfg.get("check_final_objective", True))
    min_imp = float(convergence_cfg.get("min_objective_improvement", 1.0e-4))
    converged = scaler.train(
        learning_rate=learning_rate,
        num_iters=train_iters,
        check_convergence=check,
        min_improvement=min_imp,
    )
    beta = {vocabulary[i]: float(scaler.beta_words[i]) for i in range(len(vocabulary))}
    return WordfishFitResult(
        doc_ids=doc_ids,
        theta=scaler.theta_docs.copy(),
        beta=beta,
        vocabulary=vocabulary,
        objective_final=float(scaler.objective_history[-1]) if scaler.objective_history else float("nan"),
        objective_history=scaler.objective_history,
        converged=converged,
        sign_flipped=False,
    )


def cap_document_tokens(
    tokens: List[str],
    max_tokens: int,
    seed: int,
) -> Tuple[List[str], int, bool]:
    """Function summary: subsample tokens to a ceiling for comparable document size.

    Parameters:
    - tokens: full token list.
    - max_tokens: maximum tokens to keep.
    - seed: RNG seed for reproducible subsampling.

    Returns:
    - Tuple (capped tokens, n_tokens_raw, was_truncated).
    """
    n_raw = len(tokens)
    if max_tokens <= 0 or n_raw <= max_tokens:
        return tokens, n_raw, False
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_raw, size=max_tokens, replace=False)
    capped = [tokens[int(i)] for i in sorted(idx)]
    return capped, n_raw, True


def beta_pole_diagnostics(
    beta: Dict[str, float],
    opposite_frac_threshold: float = 0.05,
) -> Dict[str, Any]:
    """Function summary: one-sided pole diagnostic on raw beta (not mean-centered).

    Parameters:
    - beta: word weights from fit.
    - opposite_frac_threshold: minority-sign fraction below which axis is one-sided.

    Returns:
    - Dict with beta_neg_fraction, beta_pos_fraction, one_sided_beta.
    """
    vals = [float(v) for v in beta.values() if not np.isnan(float(v))]
    if not vals:
        return {
            "beta_neg_fraction": float("nan"),
            "beta_pos_fraction": float("nan"),
            "one_sided_beta": True,
        }
    n = len(vals)
    n_neg = sum(1 for v in vals if v < 0)
    n_pos = sum(1 for v in vals if v > 0)
    neg_frac = n_neg / n
    pos_frac = n_pos / n
    minority = min(neg_frac, pos_frac)
    return {
        "beta_neg_fraction": neg_frac,
        "beta_pos_fraction": pos_frac,
        "one_sided_beta": minority < opposite_frac_threshold,
    }


class WordfishScalerV2:
    """Function summary: Wordfish with alternating conditional MLE blocks (scipy)."""

    def __init__(self, occurrences: np.ndarray) -> None:
        """Function summary: initialize from document-term matrix.

        Parameters:
        - occurrences: n_docs x n_words counts.
        """
        self.occurrences = np.asarray(occurrences, dtype=np.float64)
        self.num_docs, self.num_words = self.occurrences.shape
        self.alpha_docs = np.zeros(self.num_docs)
        self.theta_docs = np.zeros(self.num_docs)
        self.beta_words = np.zeros(self.num_words)
        self.psi_words = np.zeros(self.num_words)
        self.log_expectations = np.zeros((self.num_docs, self.num_words))
        self.objective_history: List[float] = []

    def initialize(self) -> None:
        """Function summary: SVD initialization (same as legacy WordfishScaler)."""
        if self.num_docs == 0 or self.num_words == 0:
            return
        avg = np.average(self.occurrences, axis=0)
        avg = np.maximum(avg, 1e-8)
        self.psi_words = np.log(avg)
        counts = np.sum(self.occurrences, axis=1)
        counts = np.maximum(counts, 1e-8)
        self.alpha_docs = np.log(counts / counts[0])
        matrix = (
            np.log(np.maximum(self.occurrences, 1e-8)).T
            - np.repeat(self.psi_words[:, np.newaxis], self.num_docs, axis=1)
            - np.repeat(self.alpha_docs[np.newaxis, :], self.num_words, axis=0)
        )
        u, _s, vt = np.linalg.svd(matrix, full_matrices=False)
        self.beta_words = u[:, 0].copy()
        self.theta_docs = vt[0, :].copy()

    def _standardize_theta(self) -> None:
        """Function summary: pin θ scale (mean 0, sd 1) for identification."""
        if self.num_docs == 0:
            return
        mu = float(np.mean(self.theta_docs))
        sd = float(np.std(self.theta_docs))
        if sd > 0:
            self.theta_docs = (self.theta_docs - mu) / sd

    def _fix_alpha0(self) -> None:
        """Function summary: fix α₀ = 0 location normalization."""
        if self.num_docs > 0:
            self.alpha_docs[0] = 0.0

    def objective(self) -> float:
        """Function summary: negative log-likelihood."""
        self.log_expectations = self._log_expectation()
        exp_le = np.exp(np.clip(self.log_expectations, -30.0, 30.0))
        ll = np.sum(self.occurrences * self.log_expectations - exp_le)
        return float(-ll)

    def _log_expectation(self) -> np.ndarray:
        """Function summary: log expected counts."""
        return (
            self.alpha_docs[:, np.newaxis]
            + self.psi_words[np.newaxis, :]
            + np.outer(self.theta_docs, self.beta_words)
        )

    def _grad_words(self) -> np.ndarray:
        """Function summary: gradient w.r.t. concatenated [psi, beta]."""
        exp_le = np.exp(np.clip(self.log_expectations, -30.0, 30.0))
        psi_g = np.sum(exp_le - self.occurrences, axis=0)
        beta_g = np.sum(
            (exp_le - self.occurrences) * self.theta_docs[:, np.newaxis],
            axis=0,
        )
        return np.concatenate([psi_g, beta_g])

    def _grad_docs(self) -> np.ndarray:
        """Function summary: gradient w.r.t. concatenated [alpha, theta]."""
        exp_le = np.exp(np.clip(self.log_expectations, -30.0, 30.0))
        alpha_g = np.sum(exp_le - self.occurrences, axis=1)
        theta_g = np.sum(
            (exp_le - self.occurrences) * self.beta_words[np.newaxis, :],
            axis=1,
        )
        return np.concatenate([alpha_g, theta_g])

    def _set_words(self, x: np.ndarray) -> None:
        """Function summary: unpack word-parameter vector."""
        self.psi_words[:] = x[: self.num_words]
        self.beta_words[:] = x[self.num_words :]

    def _set_docs(self, x: np.ndarray) -> None:
        """Function summary: unpack document-parameter vector."""
        self.alpha_docs[:] = x[: self.num_docs]
        self.theta_docs[:] = x[self.num_docs :]
        self._standardize_theta()
        self._fix_alpha0()

    def train_alternating(
        self,
        max_cycles: int = 40,
        min_improvement: float = 1e-4,
    ) -> bool:
        """Function summary: alternating scipy L-BFGS on word then document blocks.

        Parameters:
        - max_cycles: outer alternation count.
        - min_improvement: stop when objective improvement below threshold.

        Returns:
        - True if final improvement met min_improvement.
        """
        from scipy.optimize import minimize

        self.initialize()
        self._standardize_theta()
        self._fix_alpha0()
        self.objective_history = [self.objective()]

        x_words0 = np.concatenate([self.psi_words, self.beta_words])
        x_docs0 = np.concatenate([self.alpha_docs, self.theta_docs])

        for _ in range(max_cycles):
            def word_obj(xw: np.ndarray) -> float:
                self._set_words(xw)
                return self.objective()

            def word_jac(xw: np.ndarray) -> np.ndarray:
                self._set_words(xw)
                return self._grad_words()

            res_w = minimize(
                word_obj,
                x_words0,
                jac=word_jac,
                method="L-BFGS-B",
                options={"maxiter": 80, "ftol": 1e-8},
            )
            x_words0 = res_w.x
            self._set_words(x_words0)

            def doc_obj(xd: np.ndarray) -> float:
                self._set_docs(xd)
                return self.objective()

            def doc_jac(xd: np.ndarray) -> np.ndarray:
                self._set_docs(xd)
                return self._grad_docs()

            res_d = minimize(
                doc_obj,
                x_docs0,
                jac=doc_jac,
                method="L-BFGS-B",
                options={"maxiter": 80, "ftol": 1e-8},
            )
            x_docs0 = res_d.x
            self._set_docs(x_docs0)

            obj = self.objective()
            self.objective_history.append(obj)
            if len(self.objective_history) >= 2:
                if (self.objective_history[-2] - obj) < min_improvement:
                    break

        converged = True
        if len(self.objective_history) >= 2:
            converged = (self.objective_history[-2] - self.objective_history[-1]) >= min_improvement
        return converged


def fit_wordfish_v2(
    occurrences: np.ndarray,
    vocabulary: List[str],
    doc_ids: List[str],
    convergence_cfg: Dict[str, Any],
) -> WordfishFitResult:
    """Function summary: Wordfish fit via alternating conditional MLE (v2).

    Parameters:
    - occurrences: count matrix.
    - vocabulary: term list.
    - doc_ids: document ids per row.
    - convergence_cfg: max_cycles, min_objective_improvement, opposite_frac_threshold.

    Returns:
    - WordfishFitResult with pole diagnostics on raw beta.
    """
    scaler = WordfishScalerV2(occurrences)
    max_cycles = int(convergence_cfg.get("max_cycles", 40))
    min_imp = float(convergence_cfg.get("min_objective_improvement", 1.0e-4))
    converged = scaler.train_alternating(max_cycles=max_cycles, min_improvement=min_imp)
    beta = {vocabulary[i]: float(scaler.beta_words[i]) for i in range(len(vocabulary))}
    diag = beta_pole_diagnostics(
        beta,
        opposite_frac_threshold=float(convergence_cfg.get("opposite_frac_threshold", 0.05)),
    )
    return WordfishFitResult(
        doc_ids=doc_ids,
        theta=scaler.theta_docs.copy(),
        beta=beta,
        vocabulary=vocabulary,
        objective_final=float(scaler.objective_history[-1]) if scaler.objective_history else float("nan"),
        objective_history=scaler.objective_history,
        converged=converged,
        sign_flipped=False,
        one_sided_beta=bool(diag["one_sided_beta"]),
        beta_neg_fraction=float(diag["beta_neg_fraction"]),
        beta_pos_fraction=float(diag["beta_pos_fraction"]),
    )


def apply_sign_anchor(
    result: WordfishFitResult,
    doc_meta: Sequence[Dict[str, str]],
    anchor_subreddit: str,
) -> WordfishFitResult:
    """Function summary: flip theta and beta if anchor subreddit mean theta is negative.

    Parameters:
    - result: fit result to adjust.
    - doc_meta: per-doc dicts with subreddit key, aligned with doc_ids.
    - anchor_subreddit: reference forum name.

    Returns:
    - Updated WordfishFitResult (sign_flipped flag set).
    """
    sub_to_theta: Dict[str, List[float]] = {}
    for i, doc_id in enumerate(result.doc_ids):
        sub = doc_meta[i].get("subreddit", "")
        sub_to_theta.setdefault(sub, []).append(float(result.theta[i]))
    anchor_vals = sub_to_theta.get(anchor_subreddit, [])
    if not anchor_vals:
        return result
    if float(np.mean(anchor_vals)) >= 0:
        return result
    flipped_beta = {w: -b for w, b in result.beta.items()}
    return WordfishFitResult(
        doc_ids=result.doc_ids,
        theta=-result.theta,
        beta=flipped_beta,
        vocabulary=result.vocabulary,
        objective_final=result.objective_final,
        objective_history=result.objective_history,
        converged=result.converged,
        sign_flipped=True,
    )


def top_axis_words(
    beta: Dict[str, float],
    n_top: int = 25,
) -> List[Tuple[str, float, str]]:
    """Function summary: return top positive and negative beta words.

    Parameters:
    - beta: word weights.
    - n_top: words per tail.

    Returns:
    - List of (word, beta, sign) with sign in {pos, neg}.
    """
    items = sorted(beta.items(), key=lambda x: x[1])
    neg = [(w, b, "neg") for w, b in items[:n_top]]
    pos = [(w, b, "pos") for w, b in items[-n_top:][::-1]]
    return pos + neg


def compute_center_lang_pre(
    bin_starts: Sequence[str],
    thetas: Sequence[float],
    anchor_date: str,
) -> float:
    """Function summary: mean theta over pre-ban forum-bins.

    Parameters:
    - bin_starts: bin_start per row.
    - thetas: theta values.
    - anchor_date: t* (pre-ban if bin_start < t*).

    Returns:
    - Scalar center or NaN if no pre-ban rows.
    """
    vals = [
        float(t)
        for bs, t in zip(bin_starts, thetas)
        if bs < anchor_date
    ]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def zscore_preban(
    extremities: Sequence[float],
    bin_starts: Sequence[str],
    anchor_date: str,
) -> Tuple[float, float]:
    """Function summary: mean and sd of extremity on pre-ban bins for z-scoring.

    Parameters:
    - extremities: per-row extremity.
    - bin_starts: bin labels.
    - anchor_date: t*.

    Returns:
    - (mean, sd); sd=1.0 if undefined.
    """
    pre = [float(e) for e, bs in zip(extremities, bin_starts) if bs < anchor_date]
    if len(pre) < 2:
        return float("nan"), float("nan")
    mu = float(np.mean(pre))
    sd = float(np.std(pre, ddof=0))
    if sd <= 0:
        sd = 1.0
    return mu, sd


def family_dispersion(thetas: Sequence[float]) -> Dict[str, float]:
    """Function summary: var/iqr/range across forums in one family-bin.

    Parameters:
    - thetas: theta values (len >= 1).

    Returns:
    - Dict with dispersion_var, dispersion_iqr, dispersion_range (NaN if n<2).
    """
    arr = np.asarray(list(thetas), dtype=float)
    n = len(arr)
    if n < 2:
        return {
            "dispersion_var": float("nan"),
            "dispersion_iqr": float("nan"),
            "dispersion_range": float("nan"),
        }
    q75, q25 = np.percentile(arr, [75, 25])
    return {
        "dispersion_var": float(np.var(arr, ddof=0)),
        "dispersion_iqr": float(q75 - q25),
        "dispersion_range": float(np.max(arr) - np.min(arr)),
    }


def _rolling_prior_extremity_mean(
    grp: pd.DataFrame,
    row_idx: int,
    window_days: int,
    time_bin: str,
) -> float:
    """Function summary: mean extremity over prior bins within window W.

    Parameters:
    - grp: sorted subreddit panel rows.
    - row_idx: index in grp for current row.
    - window_days: calendar days (day bins) or prior row count (week bins).
    - time_bin: day or week.

    Returns:
    - Prior mean or NaN if no qualifying prior bins.
    """
    if row_idx <= 0:
        return float("nan")
    current = grp.iloc[row_idx]
    cur_bs = str(current["bin_start"])
    if time_bin == "day":
        cur_date = parse_anchor_date(cur_bs)
        vals: List[float] = []
        for j in range(row_idx):
            prior = grp.iloc[j]
            delta = (cur_date - parse_anchor_date(str(prior["bin_start"]))).days
            if 0 < delta <= window_days:
                vals.append(float(prior["extremity"]))
        if not vals:
            return float("nan")
        return float(np.mean(vals))
    prior_rows = grp.iloc[max(0, row_idx - window_days) : row_idx]
    if prior_rows.empty:
        return float("nan")
    return float(prior_rows["extremity"].astype(float).mean())


EN_LEXICON_ALIASES = frozenset({"eu", "uk", "us"})


def normalize_lexicon_code(code: str) -> str:
    """Function summary: map eu/uk/us shard codes to en for language assignment.

    Parameters:
    - code: primary_lexicon value.

    Returns:
    - Normalized it, en, or de (or original if unknown).
    """
    c = str(code).strip().lower()
    if c in EN_LEXICON_ALIASES:
        return "en"
    return c


def assign_primary_language(
    langs_seen: Set[str],
    priority: Sequence[str],
) -> str:
    """Function summary: assign one fit language via it > de > en priority.

    Parameters:
    - langs_seen: normalized lexicon codes with political comments.
    - priority: ordered language codes.

    Returns:
    - Assigned primary_lexicon for Wordfish fit.
    """
    norm = {normalize_lexicon_code(x) for x in langs_seen}
    for lang in priority:
        if lang in norm:
            return lang
    return priority[-1] if priority else "en"


def compute_change_outcomes(
    ext_df: pd.DataFrame,
    anchor_date: str,
    window_days: int,
    group_col: str = "subreddit",
) -> pd.DataFrame:
    """Function summary: add change and change_z from rolling prior extremity.

    Parameters:
    - ext_df: extremity panel with group_col, bin_start, time_bin, primary_lexicon, extremity.
    - anchor_date: ban date t*.
    - window_days: rolling window (days for day bins, prior rows for week/window bins).
    - group_col: subreddit or author for within-entity rolling mean.

    Returns:
    - Copy of ext_df with change and change_z columns.
    """
    out = ext_df.copy()
    change = pd.Series(index=out.index, dtype=float)
    group_cols = [group_col, "primary_lexicon", "time_bin"]
    for _key, grp in out.groupby(group_cols, sort=False):
        grp = grp.sort_values("bin_start")
        tbin = str(grp["time_bin"].iloc[0])
        for i, idx in enumerate(grp.index):
            row = grp.loc[idx]
            prior_mean = _rolling_prior_extremity_mean(
                grp.reset_index(drop=True), i, window_days, tbin
            )
            ext = float(row["extremity"])
            if np.isnan(prior_mean) or np.isnan(ext):
                change.loc[idx] = float("nan")
            else:
                change.loc[idx] = ext - prior_mean
    out["change"] = change

    change_z = pd.Series(index=out.index, dtype=float)
    for (_lang, _tbin), grp in out.groupby(
        ["primary_lexicon", "time_bin"], sort=False
    ):
        pre_change = [
            float(c)
            for c, bs in zip(grp["change"], grp["bin_start"])
            if str(bs) < anchor_date and not np.isnan(float(c))
        ]
        if len(pre_change) < 2:
            mu, sd = float("nan"), float("nan")
        else:
            mu = float(np.mean(pre_change))
            sd = float(np.std(pre_change, ddof=0))
            if sd <= 0:
                sd = 1.0
        for idx, row in grp.iterrows():
            c = float(row["change"])
            if np.isnan(mu) or np.isnan(c):
                change_z.loc[idx] = float("nan")
            else:
                change_z.loc[idx] = (c - mu) / sd
    out["change_z"] = change_z
    return out


def add_date_utc_column(df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: prompt-facing date_utc alias (day bins only).

    Parameters:
    - df: panel with bin_start and time_bin.

    Returns:
    - DataFrame with date_utc column.
    """
    out = df.copy()
    out["date_utc"] = out.apply(
        lambda r: r["bin_start"] if str(r.get("time_bin", "")) == "day" else "",
        axis=1,
    )
    return out


def add_placebo_flags(
    df: pd.DataFrame,
    placebo_launch_date: str,
    ban_anchor_date: str,
) -> pd.DataFrame:
    """Function summary: pre/post placebo window flags for fake cutoff analysis.

    Parameters:
    - df: panel with bin_start.
    - placebo_launch_date: fake launch (e.g. 2023-03-16).
    - ban_anchor_date: real ban t*.

    Returns:
    - DataFrame with pre_placebo and post_placebo columns.
    """
    out = df.copy()
    bs = out["bin_start"].astype(str)
    out["pre_placebo"] = (bs < placebo_launch_date).astype(int)
    out["post_placebo"] = ((bs >= placebo_launch_date) & (bs < ban_anchor_date)).astype(int)
    return out


def build_placebo_window_summary(ext_df: pd.DataFrame) -> pd.DataFrame:
    """Function summary: mean outcomes by language, time_bin, IT, post_placebo.

    Parameters:
    - ext_df: extremity panel with placebo flags and z-scores.

    Returns:
    - Aggregated summary table for null-check narrative.
    """
    cols = ["extremity_z", "change_z", "extremity", "change"]
    present = [c for c in cols if c in ext_df.columns]
    if not present:
        return pd.DataFrame()
    agg = {c: "mean" for c in present}
    agg["subreddit"] = "count"
    summary = (
        ext_df.groupby(
            ["primary_lexicon", "time_bin", "IT", "post_placebo"],
            dropna=False,
        )
        .agg(agg)
        .reset_index()
    )
    summary = summary.rename(columns={"subreddit": "n_forum_bins"})
    return summary
