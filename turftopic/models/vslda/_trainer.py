"""
This is a highly simplified NumPy only implemenation taken from here: https://github.com/izumi-lab/vSLDA/blob/main/src/models/vmf_sentence_lda.py
"""

import math
import warnings
from typing import Sequence

import numpy as np
from scipy.special import digamma as _digamma
from scipy.special import ive
from scipy.special import ive as _ive
from tqdm import trange


def calculate_avg_ll_vmf_from_encoded(
    encoded_corpus: Sequence[np.ndarray],
    topic_assignments: Sequence[Sequence[int]],
    topic_means: np.ndarray,
    kappa_per_topic: np.ndarray,
) -> float:
    """Compute average vMF log-likelihood from pre-encoded unit vectors."""
    d_dim = float(topic_means.shape[1])
    v = d_dim / 2.0 - 1.0
    kappa_safe = np.clip(
        np.asarray(kappa_per_topic, dtype=np.float64), 1e-12, None
    )
    ive_val = np.maximum(ive(v, kappa_safe), 1e-300)
    log_c = (
        v * np.log(kappa_safe)
        - (d_dim / 2.0) * math.log(2.0 * math.pi)
        - (np.log(ive_val) + kappa_safe)
    )
    scaled_topic_means = np.asarray(kappa_per_topic, dtype=np.float64)[
        :, None
    ] * np.asarray(
        topic_means,
        dtype=np.float64,
    )
    total_ll = 0.0
    total_count = 0
    for enc, doc_topics in zip(encoded_corpus, topic_assignments):
        arr = np.asarray(enc)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        topics = np.asarray(doc_topics, dtype=np.int64)
        if topics.size == 0:
            continue
        doc_ll, doc_count = accumulate_doc_average_log_likelihood(
            encoded_doc=arr,
            assignments=topics,
            log_c_per_topic=log_c,
            scaled_topic_means=scaled_topic_means,
        )
        if doc_count <= 0:
            continue
        total_ll += float(doc_ll)
        total_count += int(doc_count)
    if total_count == 0:
        return float("nan")

    return total_ll / total_count


def sample_doc_topic_assignments(
    assignments: np.ndarray,
    counts: np.ndarray,
    log_lik_doc: np.ndarray,
    alpha: np.ndarray,
    uniforms: np.ndarray,
) -> None:
    num_topics = int(counts.shape[0])
    for i in range(assignments.shape[0]):
        old_k = int(assignments[i])
        if 0 <= old_k < num_topics:
            counts[old_k] -= 1

        log_post = np.log(counts.astype(np.float64) + alpha) + log_lik_doc[i]
        log_post -= log_post.max()
        post = np.exp(log_post)
        post_sum = post.sum()
        if not np.isfinite(post_sum) or post_sum <= 0.0:
            new_k = min(int(uniforms[i] * num_topics), num_topics - 1)
        else:
            threshold = float(uniforms[i]) * float(post_sum)
            cumsum = 0.0
            new_k = num_topics - 1
            for k in range(num_topics):
                cumsum += float(post[k])
                if threshold <= cumsum:
                    new_k = k
                    break

        assignments[i] = new_k
        counts[new_k] += 1


def accumulate_doc_assignment_statistics(
    encoded_doc: np.ndarray,
    assignments: np.ndarray,
    log_mixture_weights: np.ndarray,
    scaled_component_means: np.ndarray,
    nk: np.ndarray,
    nk_comp: np.ndarray,
    r: np.ndarray,
) -> None:
    num_topics = int(nk.shape[0])
    num_components = int(nk_comp.shape[1])
    limit = min(int(encoded_doc.shape[0]), int(assignments.shape[0]))
    if num_components == 1:
        for i in range(limit):
            k = int(assignments[i])
            if k < 0 or k >= num_topics:
                continue
            nk[k] += 1.0
            nk_comp[k, 0] += 1.0
            r[k, 0] += encoded_doc[i]
        return

    for i in range(limit):
        k = int(assignments[i])
        if k < 0 or k >= num_topics:
            continue
        x = encoded_doc[i]
        log_resp = log_mixture_weights[k] + scaled_component_means[k] @ x
        log_resp -= log_resp.max()
        resp = np.exp(log_resp)
        resp_sum = float(resp.sum())
        if (not np.isfinite(resp_sum)) or resp_sum <= 0.0:
            resp = np.full(
                num_components, 1.0 / num_components, dtype=np.float64
            )
        else:
            resp /= resp_sum
        nk[k] += 1.0
        nk_comp[k] += resp
        r[k] += resp[:, None] * x[None, :]


def accumulate_doc_average_log_likelihood(
    encoded_doc: np.ndarray,
    assignments: np.ndarray,
    log_c_per_topic: np.ndarray,
    scaled_topic_means: np.ndarray,
) -> tuple[float, int]:
    num_topics = int(log_c_per_topic.shape[0])
    total_ll = 0.0
    total_count = 0
    limit = min(int(encoded_doc.shape[0]), int(assignments.shape[0]))
    for i in range(limit):
        k = int(assignments[i])
        if k < 0 or k >= num_topics:
            continue
        total_ll += float(
            log_c_per_topic[k] + scaled_topic_means[k] @ encoded_doc[i]
        )
        total_count += 1
    return total_ll, total_count


def mean_resultant_length(kappa, dim: int) -> np.ndarray:
    """A_M(kappa) = I_{M/2}(kappa) / I_{M/2-1}(kappa) via exponentially scaled Bessel functions."""
    order = float(dim) / 2.0 - 1.0
    kappa_arr = np.asarray(kappa, dtype=np.float64)
    return _ive(order + 1.0, kappa_arr) / _ive(order, kappa_arr)


def solve_kappa_newton(
    rbar: float,
    dim: int,
    kappa0: float,
    *,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> float:
    """Exact maximum-likelihood concentration: the root of A_M(kappa) = rbar.

    Newton's method from ``kappa0`` (normally the Banerjee et al. 2005 approximation) with
    A_M'(kappa) = 1 - A_M(kappa)^2 - (M - 1) A_M(kappa) / kappa (Sra 2012). Two or three
    iterations reach machine precision at the embedding dimensions used here.
    """
    kappa = float(kappa0)
    if not np.isfinite(kappa) or kappa <= 0.0:
        return kappa
    a = float(mean_resultant_length(kappa, dim))
    residual = a - float(rbar)
    for _ in range(max_iter):
        if abs(residual) <= tol:
            break
        derivative = 1.0 - a * a - (float(dim) - 1.0) * a / kappa
        candidate = kappa - residual / derivative
        if not np.isfinite(candidate) or candidate <= 0.0:
            candidate = 0.5 * kappa
        kappa = float(candidate)
        a = float(mean_resultant_length(kappa, dim))
        residual = a - float(rbar)
    return kappa


class VMFLDATrainer:
    """Trainer for vMF-Sentence-LDA.

    - If num_components == 1: single vMF per topic.
    - If num_components  > 1: mixture of vMFs per topic.
    """

    @staticmethod
    def _should_run_periodic_step(
        *,
        iteration: int,
        total_iterations: int,
        every: int,
    ) -> bool:
        return ((iteration + 1) % every == 0) or (
            iteration == total_iterations - 1
        )

    def __init__(
        self,
        embeddings: list[np.ndarray],
        num_topics: int,
        alpha: float | Sequence[float] | None = None,
        kappa: float = 10.0,
        num_components: int = 1,
        max_kappa: float = 10_000.0,
        pre_normalize_transform: str = "none",
        whitening_eps: float = 1e-5,
        algorithm_variant: str | None = None,
        saem_burn_in: int | None = 5,
        saem_decay: float = 1.0,
        kappa_solver: str = "newton",
        random_state: int | None = None,
    ):
        """
        Args:
        num_topics:
            Number of topics K.
        alpha:
            Dirichlet hyperparameter.
            - None: initialize with standard symmetric 50/K.
            - float: symmetric Dirichlet with that value.
            - sequence length K: asymmetric Dirichlet.
        kappa:
            Default κ value used as a fallback when estimation is unstable.
        num_components:
            Number of vMF components per topic C.
            C=1: single vMF per topic.
            C>1: mixture of vMFs per topic.
        max_kappa:
            Finite positive upper bound applied to learned κ values.
        pre_normalize_transform:
            Embedding transform applied after encode and before L2 normalization.
            One of: {"none", "mean_center", "whitening"}.
        whitening_eps:
            Positive stabilizer used only when pre_normalize_transform="whitening".
        saem_burn_in:
            Stochastic-approximation EM (Kuhn & Lavielle 2004): after this many outer
            iterations T_0 the E-step statistics (n_k, r_k) are averaged with step size
            gamma_t = (t - T_0)^(-saem_decay) and alpha is frozen (the default, T_0 = 5).
            None keeps the plain MCEM (each M-step uses the statistics of its own
            E-step, gamma == 1).
        saem_decay:
            Exponent a in (1/2, 1] of the step size (a > 1/2 gives sum gamma^2 < inf).
        kappa_solver:
            "newton" (default): the exact root of A_M(kappa) = rbar, started from the
            closed-form approximation of Banerjee et al. (2005); "banerjee": that
            approximation itself.
        """
        self.num_topics = int(num_topics)
        self.num_documents = len(embeddings)
        self.embedding_size = embeddings[0].shape[1]
        self.embedding_dtype = embeddings[0].dtype
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)
        self.alpha = self._init_alpha(alpha)
        self.kappa_default = float(kappa)
        self.max_kappa = float(max_kappa)
        if not np.isfinite(self.max_kappa) or self.max_kappa <= 0.0:
            raise ValueError("max_kappa must be finite and > 0")
        if not np.isfinite(self.kappa_default) or self.kappa_default <= 0.0:
            raise ValueError("kappa must be finite and > 0")
        if self.kappa_default > self.max_kappa:
            raise ValueError("kappa must be <= max_kappa")
        self.saem_burn_in = None if saem_burn_in is None else int(saem_burn_in)
        if self.saem_burn_in is not None and self.saem_burn_in < 0:
            raise ValueError("saem_burn_in must be >= 0 or None")
        self.saem_decay = float(saem_decay)
        if not (0.5 < self.saem_decay <= 1.0):
            raise ValueError("saem_decay must lie in (1/2, 1]")
        self.kappa_solver = str(kappa_solver)
        if self.kappa_solver not in {"banerjee", "newton"}:
            raise ValueError("kappa_solver must be 'banerjee' or 'newton'")
        self._sa_state: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.saem_gamma: list[float] = []
        self.saem_objective: list[float] = []
        self._last_estimated_kappa_max: float | None = None
        self._last_clipped_topic_ids: list[int] = []
        self.algorithm_variant = (
            None if algorithm_variant is None else str(algorithm_variant)
        )
        # Number of mixture components per topic (C)
        self.num_components = int(num_components)
        if self.num_components <= 0:
            raise ValueError("num_components must be >= 1")

        # Topic counts n_k (for compatibility / inspection)
        self.topic_counts = np.zeros(self.num_topics, dtype=np.int32)
        # Topic-by-document counts n_{k,d}
        self.topic_counts_per_doc = np.zeros(
            (self.num_topics, self.num_documents), dtype=np.int32
        )
        # Topic assignment per sentence: topic_assignments[d][i] = topic id of i-th sentence in document d
        self.topic_assignments: list[np.ndarray] = []
        # Sum of sentence vectors per topic s_k = Σ x_{d,i}
        # (for compatibility; in mixture case we also maintain per-component sums separately in M-step)
        self.sum_topic_vectors = np.zeros(
            (self.num_topics, self.embedding_size),
            dtype=self.embedding_dtype,
        )
        # Effective topic mean directions μ_k (unit vectors, "average" over mixture components)
        self.topic_means = np.zeros(
            (self.num_topics, self.embedding_size),
            dtype=self.embedding_dtype,
        )
        # Mixture parameters (C components per topic)
        # mixture_weights[k, c] = π_{k|c}
        self.mixture_weights = np.full(
            (self.num_topics, self.num_components),
            1.0 / self.num_components,
            dtype=np.float64,
        )
        # component_means[k, c, :] = μ_{k|c}
        self.component_means = np.zeros(
            (self.num_topics, self.num_components, self.embedding_size),
            dtype=self.embedding_dtype,
        )
        # Topic concentration parameters κ_k (shared across components of topic k)
        self.kappa_per_topic = np.full(
            self.num_topics, self.kappa_default, dtype=np.float64
        )
        # Cache for log C_M(κ_k) to avoid recomputation in the E-step
        self._log_c_per_topic = np.zeros(self.num_topics, dtype=np.float64)
        self._log_mixture_weights = np.zeros(
            (self.num_topics, self.num_components), dtype=np.float64
        )
        self._scaled_topic_means = np.zeros(
            (self.num_topics, self.embedding_size), dtype=np.float64
        )
        self._scaled_component_means = np.zeros(
            (self.num_topics, self.num_components, self.embedding_size),
            dtype=np.float64,
        )
        # History of average log-likelihood
        self.average_ll: list[float] = []
        # First initialize as single-vMF per topic (to get reasonable μ_k, κ_k, z)
        self._initialize_single_vmf(embeddings)
        # Then, if C>1, initialize mixture bases around μ_k
        if self.num_components > 1:
            self._init_mixture_from_topic_means()
        self._refresh_density_caches()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _init_alpha(self, alpha: float | Sequence[float] | None) -> np.ndarray:
        """
        Initialize asymmetric Dirichlet alpha vector.
        Standard default is alpha_k = 50 / K when alpha is None.
        """
        if alpha is None:
            val = 50.0 / float(self.num_topics)
            return np.full(self.num_topics, val, dtype=np.float64)
        if isinstance(alpha, (int, float, np.floating, np.integer)):
            return np.full(self.num_topics, float(alpha), dtype=np.float64)

        arr = np.asarray(alpha, dtype=np.float64).reshape(-1)
        if arr.size != self.num_topics:
            raise ValueError(
                f"alpha must be length {self.num_topics}, got {arr.size}"
            )
        if np.any(arr <= 0.0) or not np.all(np.isfinite(arr)):
            raise ValueError("alpha must be positive and finite")
        return arr.copy()

    def _update_alpha_fixed_point(
        self, max_iter: int = 100, tol: float = 1e-5, min_alpha: float = 1e-3
    ) -> bool:
        """
        Update asymmetric alpha via Minka's fixed-point iteration.
        Returns True if converged within tol, False otherwise.
        """
        counts = np.asarray(self.topic_counts_per_doc, dtype=np.float64)
        if counts.ndim != 2 or counts.shape[0] != self.num_topics:
            return False
        doc_lengths = counts.sum(axis=0)
        if doc_lengths.size == 0:
            return False

        eps = float(min_alpha)
        alpha = np.asarray(self.alpha, dtype=np.float64).copy()
        alpha = np.clip(alpha, eps, None)

        for _ in range(max_iter):
            alpha0 = float(alpha.sum())
            if not np.isfinite(alpha0) or alpha0 <= 0.0:
                break

            denom = np.sum(_digamma(doc_lengths + alpha0) - _digamma(alpha0))
            if not np.isfinite(denom) or denom <= 0.0:
                break

            numer = np.sum(
                _digamma(counts + alpha[:, None]) - _digamma(alpha)[:, None],
                axis=1,
            )
            if not np.all(np.isfinite(numer)):
                break

            alpha_new = alpha * (numer / denom)
            alpha_new = np.clip(alpha_new, eps, None)

            rel_change = np.max(np.abs(alpha_new - alpha) / (alpha + eps))
            alpha = alpha_new
            if rel_change < tol:
                self.alpha = alpha
                return True

        self.alpha = alpha
        return False

    def _log_vmf_normalization_const(self, kappa: np.ndarray) -> np.ndarray:
        """
        Compute log C_M(kappa) for the vMF distribution.

        log C_M(kappa) = (M/2 - 1) * log kappa - (M/2) * log(2π) - log I_{M/2-1}(kappa)
        """
        kappa = np.asarray(kappa, dtype=np.float64)
        kappa_safe = np.clip(kappa, 1e-12, None)
        d_dim = float(self.embedding_size)
        v = d_dim / 2.0 - 1.0

        ive_val = np.maximum(_ive(v, kappa_safe), 1e-300)
        log_iv = np.log(ive_val) + kappa_safe
        return (
            v * np.log(kappa_safe)
            - (d_dim / 2.0) * math.log(2.0 * math.pi)
            - log_iv
        )

    def _refresh_log_c_cache(self) -> None:
        """Refresh cached log C_M(κ_k) values after κ updates."""
        self._log_c_per_topic = self._log_vmf_normalization_const(
            self.kappa_per_topic
        )

    def _refresh_density_caches(self) -> None:
        """Refresh topic-side caches used by E-step and inference."""
        self._refresh_log_c_cache()
        self._scaled_topic_means = (
            self.kappa_per_topic[:, None] * self.topic_means
        )
        self._scaled_component_means = (
            self.kappa_per_topic[:, None, None] * self.component_means
        )
        self._log_mixture_weights = np.log(self.mixture_weights + 1e-12)

    def _random_topic_vector(self, n_sentences: int) -> np.ndarray:
        return self.rng.integers(
            self.num_topics,
            size=int(n_sentences),
            dtype=np.int32,
        )

    def _update_topic_mean_and_kappa_single(self, embeddings, k: int) -> None:
        """
        Update mean direction μ_k and concentration κ_k for topic k
        in the single-vMF (C=1) case.

        Uses:
            - s_k (sum of vectors) to compute μ_k
            - Banerjee-style approximation to estimate κ_k
        """
        s_k = self.sum_topic_vectors[k]
        n_k = int(self.topic_counts[k])

        norm_s = np.linalg.norm(s_k)
        if norm_s == 0.0 or n_k == 0:
            v = self.rng.randn(self.embedding_size)
            v /= np.linalg.norm(v) + 1e-12
            self.topic_means[k] = np.asarray(v, dtype=self.embedding_dtype)
            self.kappa_per_topic[k] = self.kappa_default
            return

        # Mean direction μ_k
        self.topic_means[k] = np.asarray(
            s_k / (norm_s + 1e-12),
            dtype=self.embedding_dtype,
        )

        # If only one point is assigned, kappa estimation is unstable; use default
        if n_k <= 1:
            self.kappa_per_topic[k] = self.kappa_default
            return

        # Mean resultant length l_k = ||s_k|| / n_k
        l_k = norm_s / (n_k + 1e-12)
        l_k = float(np.clip(l_k, 1e-6, 1.0 - 1e-6))

        d = self.embedding_size
        # Approximate κ from l_k (single vMF; standard approximation)
        numerator = l_k * (d - l_k**2)
        denominator = 1.0 - l_k**2
        kappa_est = numerator / (denominator + 1e-12)

        self.kappa_per_topic[k] = self._sanitize_kappa_estimate(kappa_est)

    def _sanitize_kappa_estimate(
        self,
        estimate: float,
        *,
        topic_id: int | None = None,
        record_diagnostics: bool = False,
    ) -> float:
        """Return a valid bounded concentration and optionally record clipping."""

        value = float(estimate)
        if not np.isfinite(value) or value <= 0.0:
            return self.kappa_default
        if record_diagnostics:
            if (
                self._last_estimated_kappa_max is None
                or value > self._last_estimated_kappa_max
            ):
                self._last_estimated_kappa_max = value
            if value > self.max_kappa and topic_id is not None:
                self._last_clipped_topic_ids.append(int(topic_id))
        return min(value, self.max_kappa)

    def _initialize_single_vmf(self, embeddings) -> None:
        """
        Initialize vMF-LDA with a single vMF per topic:
            - Randomly assign each sentence in each document to a topic.
            - Compute s_k, μ_k, κ_k from the assignments.
        This serves as a good starting point even when num_components > 1.
        """
        self.topic_assignments = []
        self.topic_counts[:] = 0
        self.topic_counts_per_doc[:] = 0
        self.sum_topic_vectors[:] = 0.0
        self.topic_means[:] = 0.0
        for d, enc in enumerate(embeddings):
            n_sent = enc.shape[0]
            topics = self._random_topic_vector(n_sent)
            self.topic_assignments.append(topics)
            if topics.size > 0:
                self.topic_counts_per_doc[:, d] = np.bincount(
                    topics,
                    minlength=self.num_topics,
                ).astype(np.int32, copy=False)
            for x, k in zip(enc, topics):
                if x.shape[0] != self.embedding_size:
                    continue
                self.topic_counts[k] += 1
                self.sum_topic_vectors[k] += x
        for k in range(self.num_topics):
            self._update_topic_mean_and_kappa_single(embeddings, k)
            if self.num_components == 1:
                self.component_means[k, 0] = self.topic_means[k]
                self.mixture_weights[k, 0] = 1.0
        self._refresh_density_caches()

    def _init_mixture_from_topic_means(self, embeddings) -> None:
        """
        Initialize mixture components (C > 1) around the single-vMF topic means.

        - π_{k|c} is set to 1/C.
        - μ_{k|c} is a small random perturbation around μ_k.
        """
        eps = 1e-6
        for k in range(self.num_topics):
            base = self.topic_means[k]
            if np.linalg.norm(base) == 0.0:
                base = self.rng.randn(self.embedding_size)
                base /= np.linalg.norm(base) + eps
            for c in range(self.num_components):
                v = base + 0.01 * self.rng.randn(self.embedding_size)
                v /= np.linalg.norm(v) + eps
                self.component_means[k, c] = np.asarray(
                    v,
                    dtype=self.embedding_dtype,
                )
            self.mixture_weights[k] = 1.0 / self.num_components

    def _random_init_topic(self, embeddings, k: int) -> None:
        """
        Randomly initialize mixture parameters for topic k.
        Used when no data is assigned to topic k in an EM iteration.
        """
        eps = 1e-6
        self.mixture_weights[k] = 1.0 / self.num_components
        for c in range(self.num_components):
            v = self.rng.randn(self.embedding_size)
            v /= np.linalg.norm(v) + eps
            self.component_means[k, c] = np.asarray(
                v,
                dtype=self.embedding_dtype,
            )
        self.kappa_per_topic[k] = self.kappa_default
        eff = self.component_means[k].mean(axis=0)
        eff_norm = np.linalg.norm(eff)
        if eff_norm > 0:
            eff /= eff_norm
        self.topic_means[k] = np.asarray(eff, dtype=self.embedding_dtype)
        self.topic_counts[k] = 0
        self.sum_topic_vectors[k] = 0.0

    def _init_topic_from_vector(
        self, embeddings, k: int, x: np.ndarray
    ) -> None:
        arr = np.asarray(x, dtype=np.float64)
        norm = np.linalg.norm(arr)
        if not np.isfinite(norm) or norm <= 0.0:
            self._random_init_topic(k)
            return

        direction = arr / (norm + 1e-12)
        self.mixture_weights[k] = 1.0 / self.num_components
        for c in range(self.num_components):
            if c == 0:
                v = direction
            else:
                v = direction + 0.01 * self.rng.randn(self.embedding_size)
                v /= np.linalg.norm(v) + 1e-12
            self.component_means[k, c] = np.asarray(
                v,
                dtype=self.embedding_dtype,
            )

        eff = self.component_means[k].mean(axis=0)
        eff_norm = np.linalg.norm(eff)
        if eff_norm > 0.0:
            eff /= eff_norm
        self.topic_means[k] = np.asarray(eff, dtype=self.embedding_dtype)
        self.kappa_per_topic[k] = self.kappa_default

    def _recompute_doc_topic_counts(self) -> None:
        """
        Recompute document–topic counts n_{k,d} from current topic_assignments.

        This ensures that in the E-step we really use N_{dk}^{(-n)} from the
        previous state of z.
        """
        self.topic_counts_per_doc[:] = 0
        for d, doc_topics in enumerate(self.topic_assignments):
            topics = np.asarray(doc_topics, dtype=np.int64)
            if topics.size == 0:
                continue
            valid = topics[(topics >= 0) & (topics < self.num_topics)]
            if valid.size == 0:
                continue
            self.topic_counts_per_doc[:, d] = np.bincount(
                valid,
                minlength=self.num_topics,
            ).astype(np.int32, copy=False)

    def _recompute_topic_counts_from_doc_counts(self) -> None:
        self.topic_counts[:] = self.topic_counts_per_doc.sum(axis=1).astype(
            np.int32,
            copy=False,
        )

    def _get_empty_topics(self, *, min_count: int = 1) -> np.ndarray:
        counts = np.asarray(
            self.topic_counts_per_doc.sum(axis=1), dtype=np.int64
        )
        return np.where(counts < int(min_count))[0]

    def _find_repair_donor_sentence(
        self,
        *,
        protected_topics: set[int],
        min_topic_count_for_repair: int,
    ) -> tuple[int, int, int] | None:
        counts = self.topic_counts_per_doc.sum(axis=1).astype(np.int64)
        candidate_topics = np.argsort(-counts)
        for donor_topic_value in candidate_topics:
            donor_topic = int(donor_topic_value)
            if donor_topic in protected_topics:
                continue
            if counts[donor_topic] <= min_topic_count_for_repair:
                continue

            locations: list[tuple[int, int]] = []
            for d, assignments in enumerate(self.topic_assignments):
                arr = np.asarray(assignments)
                if arr.size == 0:
                    continue
                sentence_indices = np.where(arr == donor_topic)[0]
                locations.extend((d, int(i)) for i in sentence_indices)
            if locations:
                selected = locations[self.rng.randint(len(locations))]
                return selected[0], selected[1], donor_topic

        return None

    def _repair_empty_topics(
        self,
        embeddings,
        *,
        min_topic_count_for_repair: int = 1,
    ) -> dict[str, object]:
        self._recompute_doc_topic_counts()
        self._recompute_topic_counts_from_doc_counts()
        target_topics = [
            int(topic)
            for topic in self._get_empty_topics(
                min_count=min_topic_count_for_repair
            )
        ]
        repaired: list[int] = []
        failed: list[int] = []
        moves: list[dict[str, int]] = []
        protected_topics: set[int] = set()
        for target_topic in target_topics:
            donor = self._find_repair_donor_sentence(
                protected_topics=protected_topics,
                min_topic_count_for_repair=min_topic_count_for_repair,
            )
            if donor is None:
                failed.append(target_topic)
                continue
            doc_index, sentence_index, old_topic = donor
            self.topic_assignments[doc_index][sentence_index] = target_topic
            self.topic_counts_per_doc[old_topic, doc_index] -= 1
            self.topic_counts_per_doc[target_topic, doc_index] += 1
            self._init_topic_from_vector(
                embeddings,
                target_topic,
                self.encoded_corpus[doc_index][sentence_index],
            )
            repaired.append(target_topic)
            protected_topics.add(target_topic)
            moves.append(
                {
                    "target_topic": int(target_topic),
                    "old_topic": int(old_topic),
                    "doc_index": int(doc_index),
                    "sentence_index": int(sentence_index),
                }
            )
        self._recompute_topic_counts_from_doc_counts()
        if repaired:
            self._refresh_density_caches()
        return {
            "num_targets": len(target_topics),
            "num_repaired": len(repaired),
            "num_failed": len(failed),
            "target_topics": target_topics,
            "repaired_topics": repaired,
            "failed_topics": failed,
            "moves": moves,
        }

    # -------------------------------------------------------------------------
    # vMF log density (for Gibbs sampling over topics)
    # -------------------------------------------------------------------------
    def log_vmf_density_tables(self, x: np.ndarray) -> np.ndarray:
        """
        Compute log of unnormalized vMF mixture density for all topics for a single unit vector x.

        - If num_components == 1:
            log p_k ∝ log C_M(κ_k) + κ_k μ_k^T x
        - If num_components  > 1:
            log p_k ∝ log C_M(κ_k) + log ∑_c π_{k|c} exp(κ_k μ_{k|c}^T x)

        The normalization constant C_M(κ_k) penalizes large κ_k and is
        required for fair topic comparisons.
        """
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"x must be 1D, got shape {arr.shape}")
        if self.num_components == 1:
            return self._log_c_per_topic + self._scaled_topic_means @ arr
        # Mixture of vMFs per topic
        # component_means: (K, C, D)
        # x: (D,)
        # dots[k, c] = μ_{k|c}^T x
        scores = np.tensordot(
            self._scaled_component_means, arr, axes=([2], [0])
        )
        log_comp = self._log_mixture_weights + scores
        # log-sum-exp over mixture components
        m = log_comp.max(axis=1, keepdims=True)  # (K, 1)
        mix_log = m + np.log(
            np.exp(log_comp - m).sum(axis=1, keepdims=True) + 1e-12
        )  # (K, 1)
        return self._log_c_per_topic + mix_log.ravel()  # (K,)

    def log_vmf_density_matrix(self, x: np.ndarray) -> np.ndarray:
        """Compute topic log-likelihoods for a batch of unit vectors."""
        arr = np.asarray(x, dtype=np.float64)
        if arr.size == 0:
            return np.zeros((0, self.num_topics), dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.embedding_size:
            raise ValueError(
                f"Expected embedding dim {self.embedding_size}, got {arr.shape[1]}"
            )
        if self.num_components == 1:
            return (
                arr @ self._scaled_topic_means.T
                + self._log_c_per_topic[None, :]
            )
        scores = np.einsum(
            "nd,kcd->nkc",
            arr,
            self._scaled_component_means,
            optimize=True,
        )
        log_comp = scores + self._log_mixture_weights[None, :, :]
        m = log_comp.max(axis=2, keepdims=True)
        mix_log = m + np.log(
            np.exp(log_comp - m).sum(axis=2, keepdims=True) + 1e-12
        )
        return self._log_c_per_topic[None, :] + mix_log[..., 0]

    def _component_responsibilities(self, k: int, x: np.ndarray) -> np.ndarray:
        """
        Compute responsibilities weight_{ic} for a given topic k and vector x.

            weight_ic ∝ π_{k|c} * vMF(x | μ_{k|c}, κ_k)
                      ∝ π_{k|c} * exp(κ_k μ_{k|c}^T x)

        Because κ_k is shared across components of topic k, c_M(κ_k) cancels
        between numerator and denominator.
        """
        if self.num_components == 1:
            return np.array([1.0], dtype=np.float64)
        log_comp = (
            self._log_mixture_weights[k] + self._scaled_component_means[k] @ x
        )
        m = log_comp.max()
        comp = np.exp(log_comp - m)
        s = comp.sum()
        if not np.isfinite(s) or s <= 0.0:
            return np.full(
                self.num_components,
                1.0 / self.num_components,
                dtype=np.float64,
            )
        return comp / s

    def _ensure_document_assignments(
        self, doc_index: int, n_sentences: int
    ) -> None:
        if doc_index >= len(self.topic_assignments):
            topics = self._random_topic_vector(n_sentences)
            self.topic_assignments.append(topics)
            self.topic_counts_per_doc[:, doc_index] = 0
            if topics.size > 0:
                self.topic_counts_per_doc[:, doc_index] = np.bincount(
                    topics,
                    minlength=self.num_topics,
                ).astype(np.int32, copy=False)
            return

        if len(self.topic_assignments[doc_index]) != n_sentences:
            topics = self._random_topic_vector(n_sentences)
            self.topic_assignments[doc_index] = topics
            self.topic_counts_per_doc[:, doc_index] = 0
            if topics.size > 0:
                self.topic_counts_per_doc[:, doc_index] = np.bincount(
                    topics,
                    minlength=self.num_topics,
                ).astype(np.int32, copy=False)

    def _sample_topic_assignment(
        self,
        doc_index: int,
        sentence_index: int,
        log_lik: np.ndarray,
    ) -> None:
        old_k = self.topic_assignments[doc_index][sentence_index]
        if 0 <= old_k < self.num_topics:
            self.topic_counts_per_doc[old_k, doc_index] -= 1

        counts = (
            self.topic_counts_per_doc[:, doc_index].astype(float) + self.alpha
        )
        log_post = np.log(counts) + log_lik
        log_post -= log_post.max()
        post = np.exp(log_post)
        post_sum = post.sum()
        if not np.isfinite(post_sum) or post_sum <= 0.0:
            post[:] = 1.0 / self.num_topics
        else:
            post /= post_sum

        new_k = int(self.rng.choice(self.num_topics, p=post))
        self.topic_assignments[doc_index][sentence_index] = new_k
        self.topic_counts_per_doc[new_k, doc_index] += 1

    def _accumulate_assignment_statistics(
        self,
        *,
        encoded_docs: Sequence[np.ndarray],
        sample_assignments: Sequence[Sequence[int]],
        nk: np.ndarray,
        nk_comp: np.ndarray,
        r: np.ndarray,
    ) -> None:
        for d in range(self.num_documents):
            enc = encoded_docs[d]
            if enc.size == 0 or d >= len(sample_assignments):
                continue
            assignments = np.asarray(sample_assignments[d], dtype=np.int32)
            if assignments.size == 0:
                continue
            limit = min(enc.shape[0], assignments.shape[0])
            if limit <= 0:
                continue
            accumulate_doc_assignment_statistics(
                encoded_doc=enc[:limit],
                assignments=assignments[:limit],
                log_mixture_weights=self._log_mixture_weights,
                scaled_component_means=self._scaled_component_means,
                nk=nk,
                nk_comp=nk_comp,
                r=r,
            )

    def _compute_assignment_statistics(
        self,
        *,
        encoded_docs: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k_topics = self.num_topics
        c_components = self.num_components
        d_dim = self.embedding_size
        nk = np.zeros(k_topics, dtype=np.float64)
        nk_comp = np.zeros((k_topics, c_components), dtype=np.float64)
        r = np.zeros((k_topics, c_components, d_dim), dtype=np.float64)
        self._accumulate_assignment_statistics(
            encoded_docs=encoded_docs,
            sample_assignments=self.topic_assignments,
            nk=nk,
            nk_comp=nk_comp,
            r=r,
        )
        return nk, nk_comp, r

    def _run_e_step(
        self,
        *,
        encoded_docs: Sequence[np.ndarray],
        num_sweeps: int,
        num_samples: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k_topics = self.num_topics
        c_components = self.num_components
        d_dim = self.embedding_size
        nk = np.zeros(k_topics, dtype=np.float64)
        r = np.zeros((k_topics, c_components, d_dim), dtype=np.float64)
        nk_comp = np.zeros((k_topics, c_components), dtype=np.float64)
        kept_sample_count = 0
        for sweep in range(num_sweeps):
            for d in range(self.num_documents):
                enc = encoded_docs[d]
                if enc.size == 0:
                    continue
                self._ensure_document_assignments(d, enc.shape[0])
                log_lik_doc = self.log_vmf_density_matrix(enc)
                counts_doc = np.array(
                    self.topic_counts_per_doc[:, d],
                    dtype=np.int32,
                    copy=True,
                )
                uniforms = self.rng.random(enc.shape[0]).astype(
                    np.float64, copy=False
                )
                sample_doc_topic_assignments(
                    assignments=self.topic_assignments[d],
                    counts=counts_doc,
                    log_lik_doc=log_lik_doc,
                    alpha=self.alpha,
                    uniforms=uniforms,
                )
                self.topic_counts_per_doc[:, d] = counts_doc

            if sweep >= num_sweeps - num_samples:
                self._accumulate_assignment_statistics(
                    encoded_docs=encoded_docs,
                    sample_assignments=self.topic_assignments,
                    nk=nk,
                    nk_comp=nk_comp,
                    r=r,
                )
                kept_sample_count += 1

        sample_count = float(max(kept_sample_count, 1))
        if sample_count > 1.0:
            nk /= sample_count
            nk_comp /= sample_count
            r /= sample_count
        return nk, nk_comp, r

    def _apply_m_step_updates(
        self,
        embeddings,
        *,
        nk: np.ndarray,
        nk_comp: np.ndarray,
        r: np.ndarray,
    ) -> None:
        c_components = self.num_components
        d_dim = self.embedding_size
        eps = 1e-12
        self._last_estimated_kappa_max = None
        self._last_clipped_topic_ids = []

        for k in range(self.num_topics):
            n_k = nk[k]
            if n_k <= 0:
                self._random_init_topic(embeddings, k)
                continue

            pi_k = nk_comp[k] / (n_k + eps)
            if not np.isfinite(pi_k).all() or pi_k.sum() <= 0.0:
                pi_k = np.full(
                    c_components, 1.0 / c_components, dtype=np.float64
                )
            else:
                pi_k /= pi_k.sum()
            self.mixture_weights[k] = pi_k

            lengths: list[float] = []
            for c in range(c_components):
                rkc = r[k, c]
                norm_rkc = np.linalg.norm(rkc)
                if norm_rkc == 0.0:
                    v = self.rng.standard_normal(size=(d_dim,))
                    v /= np.linalg.norm(v) + eps
                    self.component_means[k, c] = np.asarray(
                        v,
                        dtype=self.embedding_dtype,
                    )
                else:
                    self.component_means[k, c] = np.asarray(
                        rkc / (norm_rkc + eps),
                        dtype=self.embedding_dtype,
                    )
                    lengths.append(float(norm_rkc))

            r_k = 0.0
            if lengths:
                r_k = float(sum(lengths) / (n_k + eps))
                r_k = float(np.clip(r_k, 1e-6, 1.0 - 1e-6))

            if n_k <= 1 or r_k <= 0.0:
                self.kappa_per_topic[k] = self.kappa_default
            else:
                numerator = r_k * d_dim - r_k**3
                denominator = 1.0 - r_k**2
                kappa_est = numerator / (denominator + eps)
                if self.kappa_solver == "newton":
                    kappa_est = solve_kappa_newton(r_k, d_dim, kappa_est)
                self.kappa_per_topic[k] = self._sanitize_kappa_estimate(
                    kappa_est,
                    topic_id=k,
                    record_diagnostics=True,
                )

            eff = (
                self.mixture_weights[k][:, None] * self.component_means[k]
            ).sum(axis=0)
            norm_eff = np.linalg.norm(eff)
            if norm_eff > 0.0:
                eff /= norm_eff
            self.topic_means[k] = np.asarray(eff, dtype=self.embedding_dtype)
            self.sum_topic_vectors[k] = np.asarray(
                r[k].sum(axis=0),
                dtype=self.embedding_dtype,
            )

        self.topic_counts[:] = self.topic_counts_per_doc.sum(axis=1).astype(
            np.int32,
            copy=False,
        )
        self._refresh_density_caches()

    # ------------------------------------------------------------------------
    # SAEM: stochastic-approximation averaging of the E-step statistics
    # ------------------------------------------------------------------------
    def saem_step_size(self, t: int) -> float:
        """gamma_t of outer iteration t (1-based): 1 during burn-in, (t - T_0)^(-a) after."""
        if self.saem_burn_in is None or t <= self.saem_burn_in:
            return 1.0
        return float(t - self.saem_burn_in) ** (-self.saem_decay)

    def _saem_update(
        self,
        gamma: float,
        nk: np.ndarray,
        nk_comp: np.ndarray,
        r: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """s_t = s_{t-1} + gamma_t (S_t - s_{t-1}); with gamma == 1 this is S_t itself."""
        if self._sa_state is None or gamma >= 1.0:
            state = (nk, nk_comp, r)
        else:
            s_nk, s_nk_comp, s_r = self._sa_state
            state = (
                s_nk + gamma * (nk - s_nk),
                s_nk_comp + gamma * (nk_comp - s_nk_comp),
                s_r + gamma * (r - s_r),
            )
        self._sa_state = state
        return state

    def _saem_objective(self, nk: np.ndarray, r: np.ndarray) -> float:
        """Q(Delta; s) = sum_k { n_k log C_M(kappa_k) + kappa_k mu_k . r_k } at the M-step input."""
        log_c = self._log_vmf_normalization_const(self.kappa_per_topic)
        r_sum = np.asarray(r, dtype=np.float64).sum(axis=1)
        dots = np.einsum(
            "km,km->k", np.asarray(self.topic_means, dtype=np.float64), r_sum
        )
        return float(
            np.sum(
                np.asarray(nk, dtype=np.float64) * log_c
                + self.kappa_per_topic * dots
            )
        )

    def _update_alpha_if_needed(
        self,
        *,
        iteration: int,
        estimate_alpha: bool,
        alpha_update_every: int,
        alpha_max_iter: int,
        alpha_tol: float,
        alpha_min_value: float,
    ) -> tuple[bool, bool | None]:
        if not estimate_alpha or ((iteration + 1) % alpha_update_every != 0):
            return False, None

        converged = self._update_alpha_fixed_point(
            max_iter=alpha_max_iter,
            tol=alpha_tol,
            min_alpha=alpha_min_value,
        )
        alpha = np.asarray(self.alpha, dtype=np.float64)
        num_at_floor = int(np.sum(alpha <= alpha_min_value + 1e-12))
        if not converged:
            warnings.warn(
                f"Alpha update did not converge within {alpha_max_iter} iterations.",
            )
        return True, converged

    def _compute_average_log_likelihood(
        self, *, encoded_docs: Sequence[np.ndarray]
    ) -> float:
        if self.num_components > 1:
            ave_ll = self._compute_average_mixture_log_likelihood(
                encoded_docs=encoded_docs
            )
        else:
            ave_ll = calculate_avg_ll_vmf_from_encoded(
                encoded_corpus=encoded_docs,
                topic_assignments=self.topic_assignments,
                topic_means=self.topic_means,
                kappa_per_topic=self.kappa_per_topic,
            )
        self.average_ll.append(ave_ll)
        return ave_ll

    def _compute_average_mixture_log_likelihood(
        self, *, encoded_docs: Sequence[np.ndarray]
    ) -> float:
        total_ll = 0.0
        total_count = 0
        for enc, doc_topics in zip(
            encoded_docs, self.topic_assignments, strict=False
        ):
            arr = np.asarray(enc, dtype=np.float64)
            if arr.size == 0:
                continue
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            topics = np.asarray(doc_topics, dtype=np.int64)
            if topics.size == 0:
                continue
            limit = min(arr.shape[0], topics.shape[0])
            if limit <= 0:
                continue
            scores = self.log_vmf_density_matrix(arr[:limit])
            for row_index, topic_index in enumerate(topics[:limit]):
                if 0 <= topic_index < self.num_topics:
                    total_ll += float(scores[row_index, topic_index])
                    total_count += 1
        if total_count == 0:
            return float("nan")
        return total_ll / total_count

    def _run_training_iteration(
        self,
        embeddings,
        *,
        iteration: int,
        num_sweeps: int,
        num_samples: int,
        estimate_alpha: bool,
        alpha_update_every: int,
        alpha_max_iter: int,
        alpha_tol: float,
        alpha_min_value: float,
        repair_empty_topics: bool,
        min_topic_count_for_repair: int,
        compute_avg_log_likelihood: bool,
    ):
        encoded_docs = embeddings
        nk, nk_comp, r = self._run_e_step(
            encoded_docs=encoded_docs,
            num_sweeps=num_sweeps,
            num_samples=num_samples,
        )
        if repair_empty_topics:
            self._repair_empty_topics(
                embeddings,
                min_topic_count_for_repair=min_topic_count_for_repair,
            )
        # SAEM step: after burn-in the M-step sees the averaged statistics
        t = iteration + 1
        gamma = self.saem_step_size(t)
        nk, nk_comp, r = self._saem_update(gamma, nk, nk_comp, r)
        self._apply_m_step_updates(embeddings, nk=nk, nk_comp=nk_comp, r=r)
        saem_objective = self._saem_objective(nk, r)
        self.saem_gamma.append(float(gamma))
        self.saem_objective.append(float(saem_objective))
        alpha_frozen = self.saem_burn_in is not None and t > self.saem_burn_in
        alpha_updated, alpha_converged = self._update_alpha_if_needed(
            iteration=iteration,
            estimate_alpha=bool(estimate_alpha) and not alpha_frozen,
            alpha_update_every=alpha_update_every,
            alpha_max_iter=alpha_max_iter,
            alpha_tol=alpha_tol,
            alpha_min_value=alpha_min_value,
        )
        if compute_avg_log_likelihood:
            avg_log_likelihood = self._compute_average_log_likelihood(
                encoded_docs=encoded_docs
            )
        else:
            avg_log_likelihood = (
                float(self.average_ll[-1]) if self.average_ll else float("nan")
            )
            avg_log_likelihood_sec = 0.0

    # -------------------------------------------------------------------------
    # EM-style learning (E-step: Gibbs for z, M-step: update mixture parameters)
    # -------------------------------------------------------------------------
    def sample(
        self,
        embeddings,
        num_iterations: 10,
        num_sweeps: int = 20,
        num_samples: int = 8,
        *,
        estimate_alpha: bool = True,
        alpha_update_every: int = 1,
        alpha_max_iter: int = 100,
        alpha_tol: float = 1e-5,
        alpha_min_value: float = 1e-3,
        repair_empty_topics: bool = True,
        min_topic_count_for_repair: int = 1,
        avg_log_likelihood_every: int = 1,
        invariant_check_every: int = 1,
    ) -> None:
        """
        Run EM-style learning for vMF-Sentence-LDA.
        Each EM iteration t:
        E-step:
            - Keep topic parameters (μ_{k|c}, κ_k, π_{k|c}) fixed.
            - Recompute n_{k,d} from current z.
            - For each document d, run one sweep of Gibbs sampling over sentences:
                  p(z_{d,i} = k | z_{d,-i}, α, Δ)
                  ∝ (N_{dk}^{(-i)} + α_k) * vMF_mixture(x_{d,i} | Δ_k)
        M-step:
            - Given new topic assignments z, recompute mixture parameters:
                R_{k|c} = Σ_i weight_{ic} x_i
                μ_{k|c} = R_{k|c} / ||R_{k|c}||
                π_{k|c} = (1 / N_k) Σ_i weight_{ic}
                r_k = (1 / N_k) Σ_c ||R_{k|c}||
                κ_k ≈ (r_k M - r_k^3) / (1 - r_k^2)
        Args:
            num_iterations: Number of EM iterations.
            num_sweeps: Number of Gibbs sweeps per EM iteration (ζ).
            num_samples: Number of samples to keep from the last sweeps (B).
        At the end of each iteration, compute the average log-likelihood.
        """
        if num_sweeps < 1:
            raise ValueError("num_sweeps must be >= 1")
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        if alpha_update_every < 1:
            raise ValueError("alpha_update_every must be >= 1")
        if not np.isfinite(alpha_min_value) or alpha_min_value <= 0.0:
            raise ValueError("alpha_min_value must be positive and finite")
        if min_topic_count_for_repair < 1:
            raise ValueError("min_topic_count_for_repair must be >= 1")
        if avg_log_likelihood_every < 1:
            raise ValueError("avg_log_likelihood_every must be >= 1")
        if invariant_check_every < 1:
            raise ValueError("invariant_check_every must be >= 1")
        if num_samples > num_sweeps:
            num_samples = num_sweeps
        for it in trange(num_iterations, desc="Running Training iterations."):
            compute_avg_log_likelihood = self._should_run_periodic_step(
                iteration=it,
                total_iterations=num_iterations,
                every=avg_log_likelihood_every,
            )
            self._run_training_iteration(
                embeddings,
                iteration=it,
                num_sweeps=num_sweeps,
                num_samples=num_samples,
                estimate_alpha=estimate_alpha,
                alpha_update_every=alpha_update_every,
                alpha_max_iter=alpha_max_iter,
                alpha_tol=alpha_tol,
                alpha_min_value=alpha_min_value,
                repair_empty_topics=repair_empty_topics,
                min_topic_count_for_repair=min_topic_count_for_repair,
                compute_avg_log_likelihood=compute_avg_log_likelihood,
            )

    def get_num_documents(self) -> int:
        """Return the number of documents used to train the model."""
        if not hasattr(self, "topic_counts_per_doc"):
            raise AttributeError(
                "topic_counts_per_doc is not available. "
                "Make sure you have run `sample(...)` before calling this method."
            )
        return int(self.topic_counts_per_doc.shape[1])

    def infer_topic_proportions(
        self,
        embeddings: list[np.ndarray],
    ) -> list[np.ndarray]:
        sentence_posteriors = []
        for d, encoded_doc in enumerate(embeddings):
            enc = np.asarray(encoded_doc, dtype=np.float64)
            if enc.size == 0:
                sentence_posteriors.append(
                    np.zeros((0, self.num_topics), dtype=np.float64)
                )
                continue
            if enc.ndim == 1:
                enc = enc.reshape(1, -1)
            log_lik_doc = self.log_vmf_density_matrix(enc)
            probs = np.copy(log_lik_doc)
            probs -= probs.max(axis=1, keepdims=True)
            probs = np.exp(probs)
            row_sums = probs.sum(axis=1, keepdims=True)
            invalid = ~np.isfinite(row_sums) | (row_sums <= 0.0)
            row_sums[invalid] = 1.0
            probs /= row_sums
            if np.any(invalid):
                probs[invalid.ravel()] = 1.0 / self.num_topics
            sentence_posteriors.append(probs)
        return sentence_posteriors
