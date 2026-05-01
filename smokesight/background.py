"""Background radiance estimation.

Estimates a (H, W, N_lambda) background plate L0 plus a per-pixel
sigma_L0 and a (H, W) confidence map. Every downstream tau pixel passes
through that confidence gate -- an overconfident background is the
single biggest source of erroneous tau retrievals on real data.

Four estimators are supported. They differ mostly in how robust they
are to brief plume incursions during the calibration window:

  * temporal_median  -- per-pixel median across the first n frames.
                        Robust to outliers, what you want by default.
  * temporal_mean    -- per-pixel mean. Cheaper but biased high if a
                        plume drifts through.
  * percentile_10    -- the 10th percentile. Useful for emissive plumes
                        where the plume is brighter than the background.
  * gmm              -- 2-component Gaussian mixture per pixel; takes
                        the lower-mean component as background. Slower;
                        worth it for cluttered scenes with persistent
                        movement.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from smokesight._results import BackgroundResult, CalibrationResult
from smokesight._types import FloatArray

VALID_METHODS = ("temporal_median", "temporal_mean", "gmm", "percentile_10")


def background(
    cal: CalibrationResult,
    *,
    n_frames: int = 100,
    method: str = "temporal_median",
    mask: Optional[FloatArray] = None,
    min_confidence: float = 0.5,
) -> BackgroundResult:
    """Estimate L0 from the first n_frames of cal.L."""
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}, got {method!r}")

    total = cal.L.shape[0]
    if n_frames > total:
        raise ValueError(f"n_frames={n_frames} exceeds total frames in cal ({total})")

    window = cal.L[:n_frames]  # (n, H, W, N_lambda)
    L0, sigma_L0, confidence = _estimate(window, method)

    if mask is not None:
        if mask.shape != confidence.shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match confidence shape {confidence.shape}"
            )
        confidence = np.where(mask.astype(bool), 0.0, confidence)

    # downstream consumers can filter on the (returned) min_confidence; we
    # don't pre-mask L0 because the user may want the underlying values.
    confidence = np.clip(confidence, 0.0, 1.0).astype(np.float32)
    _ = min_confidence  # carried into BackgroundResult metadata? not yet -- doc'd as gate

    return BackgroundResult(
        L0=L0.astype(np.float32),
        sigma_L0=sigma_L0.astype(np.float32),
        confidence=confidence,
        method=method,
        n_frames_used=n_frames,
    )


def _estimate(
    window: FloatArray, method: str
) -> tuple[FloatArray, FloatArray, FloatArray]:
    if method == "temporal_median":
        return _temporal_median(window)
    if method == "temporal_mean":
        return _temporal_mean(window)
    if method == "percentile_10":
        return _percentile_10(window)
    if method == "gmm":
        return _gmm(window)
    raise AssertionError(f"unhandled method {method!r}")  # validated upstream


def _temporal_median(window: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    L0 = np.median(window, axis=0)
    # IQR / 1.35 is the asymptotically-consistent estimator of sigma for a
    # Gaussian when you used the median for centre.
    q1, q3 = np.percentile(window, [25, 75], axis=0)
    sigma_L0 = (q3 - q1) / 1.35
    iqr = (q3 - q1).mean(axis=-1) if window.shape[-1] > 1 else (q3 - q1)[..., 0]
    median_panchromatic = L0.mean(axis=-1) if L0.ndim == 3 else L0
    confidence = 1.0 - _safe_div(iqr, np.abs(median_panchromatic))
    return L0, sigma_L0, confidence


def _temporal_mean(window: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    L0 = window.mean(axis=0)
    sigma_L0 = window.std(axis=0, ddof=1)
    pan_mean = L0.mean(axis=-1) if L0.ndim == 3 else L0
    pan_std = sigma_L0.mean(axis=-1) if sigma_L0.ndim == 3 else sigma_L0
    confidence = 1.0 - _safe_div(pan_std, np.abs(pan_mean))
    return L0, sigma_L0, confidence


def _percentile_10(window: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    L0 = np.percentile(window, 10, axis=0)
    p50, p90 = np.percentile(window, [50, 90], axis=0)
    sigma_L0 = (p90 - L0) / 1.28  # one-sided gaussian distance from 50th to 90th
    spread = (p90 - L0).mean(axis=-1) if window.shape[-1] > 1 else (p90 - L0)[..., 0]
    centre = p50.mean(axis=-1) if p50.ndim == 3 else p50
    confidence = 1.0 - _safe_div(spread, np.abs(centre))
    return L0, sigma_L0, confidence


def _gmm(window: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:
        raise ImportError(
            "method='gmm' requires scikit-learn; install with "
            "`pip install scikit-learn` or pick another method"
        ) from exc

    n, h, w, n_lambda = window.shape
    L0 = np.empty((h, w, n_lambda), dtype=np.float32)
    sigma_L0 = np.empty_like(L0)
    confidence = np.empty((h, w), dtype=np.float32)

    # Per-pixel 2-component GMM. This is slow on big videos -- callers
    # usually reach for it only when the simpler methods break.
    for y in range(h):
        for x in range(w):
            samples = window[:, y, x, :].reshape(n, -1)  # (n, N_lambda)
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(samples)
            lo = int(np.argmin(gmm.means_[:, 0]))  # background = lower-mean component
            L0[y, x] = gmm.means_[lo]
            sigma_L0[y, x] = np.sqrt(np.diag(gmm.covariances_[lo]))
            confidence[y, x] = float(gmm.weights_[lo])
    return L0, sigma_L0, confidence


def _safe_div(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    """Element-wise divide, returning 0 where denominator is 0 or non-finite."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(
            (denominator > 0) & np.isfinite(denominator),
            numerator / denominator,
            0.0,
        )
    return np.asarray(out)
