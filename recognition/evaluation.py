import numpy as np
from skimage.metrics import structural_similarity


def evaluate(training, validation, predicted):
    """Validation marginal KL and aligned level-by-time depth SSIM; training fixes scales."""
    train, real, fake = (np.asarray(x, dtype=np.float64) for x in (training, validation, predicted))
    assert train.ndim == real.ndim == fake.ndim == 2
    assert real.shape == fake.shape and real.shape[1] == train.shape[1] == 40
    assert len(train) > 0 and len(real) >= 7
    assert all(np.isfinite(x).all() for x in (train, real, fake))
    divergences = []
    for reference, observed, generated in zip(train.T, real.T, fake.T):
        low, high = reference.min(), reference.max()
        edges = np.r_[-np.inf, np.linspace(low, max(high, low + 1e-6), 33), np.inf]
        p, q = (np.histogram(x, bins=edges)[0] / len(x) + 1e-8 for x in (observed, generated))
        p, q = (x / x.sum() for x in (p, q))
        divergences.append(float(np.sum(p * np.log(p / q))))
    # Arrange volume rows by relative price: BID10 ... BID1, ASK1 ... ASK10.
    indices = np.r_[np.arange(39, 20, -2), np.arange(20, 40, 2)]
    reference, observed, generated = (x[:, indices].T for x in (train, real, fake))
    log_ssim = structural_similarity(observed, generated, win_size=7,
                                    data_range=max(float(np.ptp(reference)), 1e-6))
    reference, observed, generated = (np.expm1(x) for x in (reference, observed, generated))
    assert all(np.isfinite(x).all() for x in (reference, observed, generated))
    raw_ssim = structural_similarity(observed, generated, win_size=7,
                                    data_range=max(float(np.ptp(reference)), 1e-6))
    return {"mean_kl": float(np.mean(divergences)), "max_kl": max(divergences),
            "midpoint_kl": divergences[0], "feature_kl": divergences,
            "depth_ssim": float(raw_ssim), "log_depth_ssim": float(log_ssim),
            "kl_target_met": bool(np.mean(divergences) < 0.1), "ssim_target_met": bool(raw_ssim > 0.6)}


if __name__ == "__main__":
    # TODO measure KL divergence between spread and mid-price returns
    raise NotImplemented
