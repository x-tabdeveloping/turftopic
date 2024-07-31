import numpy as np
from sklearn.decomposition._nmf import _beta_divergence, _initialize_nmf
from sklearn.utils.extmath import safe_sparse_dot

EPSILON = np.finfo(np.float32).eps


def weighted_nmf(
    dtm: np.ndarray,
    weight: np.ndarray,
    n_components: int,
    seed: int,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Multiplicative Update algorithm for a special case of weighted NMF, where
    only the rows are weighted, but not the individual elements in the data matrix."""
    doc_topic_matrix, components = _initialize_nmf(
        dtm, n_components, random_state=seed
    )
    U = components.T
    V = doc_topic_matrix.T
    weighted_A = dtm.T.multiply(weight)  # .T
    prev_error = np.inf
    for i in range(0, max_iter):
        # Update V
        numerator = safe_sparse_dot(U.T, weighted_A)
        denominator = np.linalg.multi_dot((U.T, U, V * weight))
        denominator[denominator <= 0] = EPSILON
        delta = numerator
        delta /= denominator
        delta[np.isinf(delta) & (V == 0)] = 0
        V *= delta
        # Update U
        numerator = safe_sparse_dot(weighted_A, V.T)
        denominator = np.linalg.multi_dot((U, V * weight, V.T))
        denominator[denominator <= 0] = EPSILON
        delta = numerator
        delta /= denominator
        delta[np.isinf(delta) & (U == 0)] = 0
        U *= delta
        if (tol > 0) and (i % 10 == 0):
            error = _beta_divergence(dtm, V.T, U.T, 2)
            if (error - prev_error) > tol:
                break
            prev_error = error
    components, doc_topic_matrix = U.T, V.T
    return components, doc_topic_matrix
