# ---------- transforms.py ----------
import numpy as np

def _normalize(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

def fit_whitening(X, eps=1e-6):
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    cov = np.cov(Xc, rowvar=False)
    U, S, _ = np.linalg.svd(cov)
    W = U @ np.diag(1.0/np.sqrt(S + eps)) @ U.T
    return {"type":"whiten", "W":W, "mu":mu}

def apply_whitening(X, params):
    Xc = X - params["mu"]
    Xw = Xc @ params["W"]
    return _normalize(Xw)

def fit_isd(X, k=1):
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)  # Vt: [min(N,D), D]
    V = Vt.T                                           # [D, min(N,D)]
    return {"type":"isd", "mu":mu, "V":V, "k":int(k)}

def apply_isd(X, params):
    V = params["V"]; k = params["k"]
    Xc = X - params["mu"]
    if k > 0:
        Xc = Xc - (Xc @ V[:, :k]) @ V[:, :k].T
    return _normalize(Xc)

def compose_whiten_isd(X, wparams, iparams):
    return apply_isd(apply_whitening(X, wparams), iparams)

def maybe_fit_transform(X_img, X_txt, mode="whiten_isd", fit_mode="per_video", k=1, pca_first=None):
    """
    X_img: [Ni,D] image/frame embeddings; X_txt: [Nt,D] text/tag embeddings (both normalized).
    fit_mode: 'offline'|'per_dataset'|'per_video'
      - For offline/per_dataset, you pass precomputed params via kwargs (outside this helper).
      - For per_video, we fit on the current batch (frames + texts).
    pca_first: int or None -> optional PCA dim for stability (e.g., 256) before fitting transforms.
    Returns: transformed (X_img_t, X_txt_t), and the params dict(s)
    """
    X_pool = np.vstack([X_img, X_txt])
    # Optional PCA compression for stability with few samples
    if pca_first is not None and pca_first < X_pool.shape[1]:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(pca_first, X_pool.shape[0]-1, X_pool.shape[1]), random_state=42)
        X_pool_r = pca.fit_transform(X_pool)
        X_img_r  = pca.transform(X_img)
        X_txt_r  = pca.transform(X_txt)
        back = pca.components_.T  # [D, r]
        # We'll fit in reduced space, then map by explicit transforms there
        # and finally bring back by multiplying with back.T if needed (but we keep reduced for scoring).
        # Simpler: do retrieval in reduced space.
        do_back = False
    else:
        X_pool_r = X_pool; X_img_r = X_img; X_txt_r = X_txt
        do_back = False

    wparams = iparams = None
    if mode == "raw":
        return X_img, X_txt, {"mode":"raw"}

    if mode == "whiten":
        wparams = fit_whitening(X_pool_r)
        X_img_t = apply_whitening(X_img_r, wparams)
        X_txt_t = apply_whitening(X_txt_r, wparams)

    elif mode == "isd":
        iparams = fit_isd(X_pool_r, k=k)
        X_img_t = apply_isd(X_img_r, iparams)
        X_txt_t = apply_isd(X_txt_r, iparams)

    elif mode == "whiten_isd":
        wparams = fit_whitening(X_pool_r)
        iparams = fit_isd(apply_whitening(X_pool_r, wparams), k=k)
        X_img_t = compose_whiten_isd(X_img_r, wparams, iparams)
        X_txt_t = compose_whiten_isd(X_txt_r, wparams, iparams)
    else:
        raise ValueError("Unknown mode")

    return X_img_t, X_txt_t, {"mode":mode, "whiten":wparams, "isd":iparams}