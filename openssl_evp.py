import ctypes
import os
import time
from ctypes import POINTER, byref, c_char_p, c_int, c_size_t, c_void_p, create_string_buffer

DEFAULT_DLL = os.path.join(os.environ.get("LOCALAPPDATA", ""), "pqenv", "Library", "bin", "libcrypto-4-x64.dll")

_LIB = None
_DLL_PATH = None


def load(dll_path=None):
    global _LIB, _DLL_PATH
    if _LIB is not None:
        return _LIB
    candidates = [dll_path, os.environ.get("PQ_OPENSSL_DLL"), DEFAULT_DLL]
    for cand in candidates:
        if cand and os.path.exists(cand):
            _DLL_PATH = cand
            break
    else:
        raise FileNotFoundError(
            "OpenSSL 3.5+ DLL not found. Install via: "
            "micromamba create -y -p %LOCALAPPDATA%\\pqenv -c conda-forge openssl=4.0.2 "
            "or set PQ_OPENSSL_DLL to a libcrypto DLL with ML-KEM/ML-DSA/SLH-DSA support."
        )
    lib = ctypes.CDLL(_DLL_PATH)
    lib.EVP_PKEY_CTX_new_from_name.argtypes = [c_void_p, c_char_p, c_char_p]
    lib.EVP_PKEY_CTX_new_from_name.restype = c_void_p
    lib.EVP_PKEY_CTX_new.argtypes = [c_void_p, c_void_p]
    lib.EVP_PKEY_CTX_new.restype = c_void_p
    lib.EVP_PKEY_CTX_free.argtypes = [c_void_p]
    lib.EVP_PKEY_keygen_init.argtypes = [c_void_p]
    lib.EVP_PKEY_keygen_init.restype = c_int
    lib.EVP_PKEY_keygen.argtypes = [c_void_p, POINTER(c_void_p)]
    lib.EVP_PKEY_keygen.restype = c_int
    lib.EVP_PKEY_free.argtypes = [c_void_p]
    lib.EVP_PKEY_get_size.argtypes = [c_void_p]
    lib.EVP_PKEY_get_size.restype = c_int
    lib.EVP_PKEY_encapsulate_init.argtypes = [c_void_p, c_void_p]
    lib.EVP_PKEY_encapsulate_init.restype = c_int
    lib.EVP_PKEY_encapsulate.argtypes = [c_void_p, c_void_p, POINTER(c_size_t), c_void_p, POINTER(c_size_t)]
    lib.EVP_PKEY_encapsulate.restype = c_int
    lib.EVP_PKEY_decapsulate_init.argtypes = [c_void_p, c_void_p]
    lib.EVP_PKEY_decapsulate_init.restype = c_int
    lib.EVP_PKEY_decapsulate.argtypes = [c_void_p, c_void_p, POINTER(c_size_t), c_void_p, c_size_t]
    lib.EVP_PKEY_decapsulate.restype = c_int
    lib.EVP_MD_CTX_new.restype = c_void_p
    lib.EVP_MD_CTX_free.argtypes = [c_void_p]
    lib.EVP_DigestSignInit.argtypes = [c_void_p, POINTER(c_void_p), c_void_p, c_void_p, c_void_p]
    lib.EVP_DigestSignInit.restype = c_int
    lib.EVP_DigestSign.argtypes = [c_void_p, c_void_p, POINTER(c_size_t), c_void_p, c_size_t]
    lib.EVP_DigestSign.restype = c_int
    lib.EVP_DigestVerifyInit.argtypes = [c_void_p, POINTER(c_void_p), c_void_p, c_void_p, c_void_p]
    lib.EVP_DigestVerifyInit.restype = c_int
    lib.EVP_DigestVerify.argtypes = [c_void_p, c_void_p, c_size_t, c_void_p, c_size_t]
    lib.EVP_DigestVerify.restype = c_int
    lib.OpenSSL_version.argtypes = [c_int]
    lib.OpenSSL_version.restype = c_char_p
    _LIB = lib
    return lib


def version_string():
    lib = load()
    return lib.OpenSSL_version(0).decode()


def dll_path():
    load()
    return _DLL_PATH


def keygen(name):
    lib = load()
    ctx = lib.EVP_PKEY_CTX_new_from_name(None, name.encode() if isinstance(name, str) else name, None)
    if not ctx:
        raise RuntimeError(f"algorithm not available: {name}")
    if lib.EVP_PKEY_keygen_init(ctx) != 1:
        raise RuntimeError("keygen_init failed")
    pkey = c_void_p()
    if lib.EVP_PKEY_keygen(ctx, byref(pkey)) != 1:
        raise RuntimeError(f"keygen failed for {name}")
    lib.EVP_PKEY_CTX_free(ctx)
    return pkey


def pkey_size(pkey):
    return load().EVP_PKEY_get_size(pkey)


def encaps(pkey):
    lib = load()
    ctx = lib.EVP_PKEY_CTX_new(pkey, None)
    if lib.EVP_PKEY_encapsulate_init(ctx, None) != 1:
        raise RuntimeError("encapsulate_init failed")
    ct_len = c_size_t(0)
    ss_len = c_size_t(0)
    if lib.EVP_PKEY_encapsulate(ctx, None, byref(ct_len), None, byref(ss_len)) != 1:
        raise RuntimeError("encapsulate size query failed")
    ct = create_string_buffer(ct_len.value)
    ss = create_string_buffer(ss_len.value)
    l1 = c_size_t(ct_len.value)
    l2 = c_size_t(ss_len.value)
    if lib.EVP_PKEY_encapsulate(ctx, ct, byref(l1), ss, byref(l2)) != 1:
        raise RuntimeError("encapsulate failed")
    lib.EVP_PKEY_CTX_free(ctx)
    return ct.raw[: l1.value], ss.raw[: l2.value]


def decaps(pkey, ciphertext):
    lib = load()
    ctx = lib.EVP_PKEY_CTX_new(pkey, None)
    if lib.EVP_PKEY_decapsulate_init(ctx, None) != 1:
        raise RuntimeError("decapsulate_init failed")
    out = create_string_buffer(64)
    out_len = c_size_t(64)
    buf = create_string_buffer(bytes(ciphertext))
    if lib.EVP_PKEY_decapsulate(ctx, out, byref(out_len), buf, len(ciphertext)) != 1:
        raise RuntimeError("decapsulate failed")
    lib.EVP_PKEY_CTX_free(ctx)
    return out.raw[: out_len.value]


def sign(pkey, msg):
    lib = load()
    mctx = lib.EVP_MD_CTX_new()
    pctx = c_void_p()
    if lib.EVP_DigestSignInit(mctx, byref(pctx), None, None, pkey) != 1:
        raise RuntimeError("DigestSignInit failed")
    size = pkey_size(pkey)
    sig = create_string_buffer(size)
    sig_len = c_size_t(size)
    mbuf = create_string_buffer(bytes(msg))
    if lib.EVP_DigestSign(mctx, sig, byref(sig_len), mbuf, len(msg)) != 1:
        raise RuntimeError("DigestSign failed")
    lib.EVP_MD_CTX_free(mctx)
    return sig.raw[: sig_len.value]


def verify(pkey, msg, signature):
    lib = load()
    mctx = lib.EVP_MD_CTX_new()
    pctx = c_void_p()
    if lib.EVP_DigestVerifyInit(mctx, byref(pctx), None, None, pkey) != 1:
        raise RuntimeError("DigestVerifyInit failed")
    mbuf = create_string_buffer(bytes(msg))
    sbuf = create_string_buffer(bytes(signature))
    ok = lib.EVP_DigestVerify(mctx, sbuf, len(signature), mbuf, len(msg))
    lib.EVP_MD_CTX_free(mctx)
    if ok != 1:
        raise ValueError("signature verification failed")


def keygen_time_us(name, n_ops, n_batches):
    samples = []
    for _ in range(n_batches):
        t0 = time.perf_counter_ns()
        for _ in range(n_ops):
            p = keygen(name)
            load().EVP_PKEY_free(p)
        samples.append((time.perf_counter_ns() - t0) / n_ops / 1e3)
    return samples
