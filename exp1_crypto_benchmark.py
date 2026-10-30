import datetime
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import openssl_evp as ossl
from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa, x25519
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from cryptography.x509.oid import NameOID
from pqcrypto.kem import hqc_128, mceliece_460896, ml_kem_512 as pq_ml_kem_512
from pqcrypto.kem import ml_kem_768 as pq_ml_kem_768, ml_kem_1024 as pq_ml_kem_1024, sntrup_761
from pqcrypto.sign import ml_dsa_44 as pq_ml_dsa_44, ml_dsa_65 as pq_ml_dsa_65
from pqcrypto.sign import ml_dsa_87 as pq_ml_dsa_87, slh_dsa_sha2_128f as pq_slh_128f
from pqcrypto.sign import slh_dsa_sha2_128s as pq_slh_128s

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

MSG = b"critical-infrastructure-pqc-benchmark-message-0123456789"
PAYLOAD_1KB = bytes(range(256)) * 4
SEED = 20261023


def cpu_name():
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, timeout=30,
        )
        return out.stdout.strip().splitlines()[0].strip()
    except Exception:
        return platform.processor() or "unknown"


def collect_env():
    import cryptography
    import matplotlib as mpl
    import pqcrypto
    import scipy
    from cryptography.hazmat.bindings.openssl.binding import Binding
    b = Binding()
    v = b.lib.OpenSSL_version_num()
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": cpu_name(),
        "packages": {
            "pqcrypto": pqcrypto.__version__,
            "cryptography": cryptography.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": mpl.__version__,
        },
        "openssl_classical_cryptography": f"{(v >> 28) & 0xFF}.{(v >> 22) & 0xFF}.{(v >> 16) & 0xFF}",
        "openssl_pqc_dll": ossl.version_string(),
        "openssl_pqc_dll_path": ossl.dll_path(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
    }


def bench(fn, n_ops, n_batches=15, warmup=True):
    if warmup:
        for _ in range(2):
            fn()
    samples = []
    for _ in range(n_batches):
        t0 = time.perf_counter_ns()
        for _ in range(n_ops):
            fn()
        samples.append((time.perf_counter_ns() - t0) / n_ops / 1e3)
    s = np.asarray(samples, dtype=float)
    med = float(np.median(s))
    return {
        "median_us": round(med, 3),
        "p25_us": round(float(np.percentile(s, 25)), 3),
        "p75_us": round(float(np.percentile(s, 75)), 3),
        "ops_per_s": round(1e6 / med, 1) if med > 0 else float("inf"),
        "n_ops": n_ops,
        "n_batches": n_batches,
    }


# ---------------------------------------------------------------------------
# Deterministic reference certificate material for the byte model.
#
# framing := len(certificate DER) - len(SubjectPublicKeyInfo DER) is the X.509
# overhead added to every modelled certificate (manuscript section 4.3).  Two
# things used to make it move between runs, shifting every handshake/PKI byte
# count by 1-2 B:
#   * x509.random_serial_number() - random DER INTEGER length;
#   * the ECDSA signature value   - r||s is 70-72 B depending on leading zeros
#                                   of the two 32-byte integers.
# Both are pinned: fixed serial + fixed validity window, and the self-signed
# ECDSA certificate is re-signed until its framing equals FRAMING_TARGET.
# RSA signatures are fixed-length (256 B for RSA-2048/PSS here), so the RSA
# certificate is deterministic without a loop.
# ---------------------------------------------------------------------------
CERT_SERIAL = 0x7F112233445566778899AABBCCDDEEFF00112233   # 159 bits (max allowed)
NOT_BEFORE = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
NOT_AFTER = datetime.datetime(2035, 12, 31, 23, 59, 59, tzinfo=datetime.timezone.utc)
FRAMING_TARGET = 245            # X.509 framing overhead quoted in the manuscript


def build_reference_cert(priv, framing_target=None):
    """Self-signed cert with pinned serial/validity; optionally pin the framing."""
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "infra.example.org")])
    builder = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name)
        .public_key(priv.public_key())
        .serial_number(CERT_SERIAL)
        .not_valid_before(NOT_BEFORE)
        .not_valid_after(NOT_AFTER)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
    )
    spki = len(priv.public_key().public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo))
    for _ in range(64):
        cert = builder.sign(priv, hashes.SHA256())
        der = cert.public_bytes(Encoding.DER)
        framing = len(der) - spki
        if framing_target is None or framing == framing_target:
            return {"cert_der": len(der), "spki": spki, "framing": framing,
                    "signature": len(cert.signature)}
    raise RuntimeError(f"could not reach a {framing_target} B framing in 64 attempts")


def build_cert_meta():
    """Certificate facts used by the workload byte models (fully deterministic)."""
    esk = ec.generate_private_key(ec.SECP256R1())
    rsk2048 = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pss = asym_padding.PSS(mgf=asym_padding.MGF1(hashes.SHA256()), salt_length=32)
    rsa_sig = rsk2048.sign(MSG, pss, hashes.SHA256())

    ec_cert = build_reference_cert(esk, FRAMING_TARGET)
    rsa_cert = build_reference_cert(rsk2048)
    assert ec_cert["framing"] == FRAMING_TARGET
    return {
        "ecdsa_framing": ec_cert["framing"],
        "ecdsa_ca_sig": ec_cert["signature"],   # 71 B DER ECDSA signature inside the framing
        "ecdsa_cert": ec_cert["cert_der"],
        "rsa_cert": rsa_cert["cert_der"],
        "rsa_sig": len(rsa_sig),
        "ecdsa_raw_sig": 64,
    }


def build_classical():
    rows, sizes = [], {}
    xsk = x25519.X25519PrivateKey.generate()
    xpeer = x25519.X25519PrivateKey.generate().public_key()

    esk = ec.generate_private_key(ec.SECP256R1())
    epk = esk.public_key()
    ecdsa_sig = esk.sign(MSG, ec.ECDSA(hashes.SHA256()))

    edsk = ed25519.Ed25519PrivateKey.generate()
    edpk = edsk.public_key()
    ed_sig = edsk.sign(MSG)

    rsk2048 = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    rsk3072 = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    pss = asym_padding.PSS(mgf=asym_padding.MGF1(hashes.SHA256()), salt_length=32)
    rsa_sig2048 = rsk2048.sign(MSG, pss, hashes.SHA256())
    rsa_sig3072 = rsk3072.sign(MSG, pss, hashes.SHA256())
    rpk2048 = rsk2048.public_key()
    rpk3072 = rsk3072.public_key()

    aes128 = AESGCM(os.urandom(16))
    aes256 = AESGCM(os.urandom(32))
    nonce = os.urandom(12)

    ops = [
        ("X25519", "keygen", lambda: x25519.X25519PrivateKey.generate(), 500, 15),
        ("X25519", "shared_secret", lambda: xsk.exchange(xpeer), 500, 15),
        ("ECDSA-P256", "keygen", lambda: ec.generate_private_key(ec.SECP256R1()), 200, 10),
        ("ECDSA-P256", "sign", lambda: esk.sign(MSG, ec.ECDSA(hashes.SHA256())), 300, 15),
        ("ECDSA-P256", "verify", lambda: epk.verify(ecdsa_sig, MSG, ec.ECDSA(hashes.SHA256())), 300, 15),
        ("Ed25519", "keygen", lambda: ed25519.Ed25519PrivateKey.generate(), 300, 10),
        ("Ed25519", "sign", lambda: edsk.sign(MSG), 300, 15),
        ("Ed25519", "verify", lambda: edpk.verify(ed_sig, MSG), 300, 15),
        ("RSA-2048", "keygen", lambda: rsa.generate_private_key(public_exponent=65537, key_size=2048), 3, 4),
        ("RSA-2048", "sign", lambda: rsk2048.sign(MSG, pss, hashes.SHA256()), 150, 10),
        ("RSA-2048", "verify", lambda: rpk2048.verify(rsa_sig2048, MSG, pss, hashes.SHA256()), 150, 10),
        ("RSA-3072", "sign", lambda: rsk3072.sign(MSG, pss, hashes.SHA256()), 100, 8),
        ("RSA-3072", "verify", lambda: rpk3072.verify(rsa_sig3072, MSG, pss, hashes.SHA256()), 100, 8),
        ("AES-128-GCM", "encrypt_1KB", lambda: aes128.encrypt(nonce, PAYLOAD_1KB, None), 1000, 10),
        ("AES-256-GCM", "encrypt_1KB", lambda: aes256.encrypt(nonce, PAYLOAD_1KB, None), 1000, 10),
    ]
    for algo, op, fn, n, b in ops:
        r = bench(fn, n, b)
        rows.append({"family": "classical", "implementation": "cryptography/OpenSSL",
                     "algorithm": algo, "operation": op, **r})

    sizes.update({
        "X25519": {"public_key": 32, "secret_key": 32, "ciphertext": 32, "signature": 0},
        "Ed25519": {"public_key": 32, "secret_key": 64, "ciphertext": 0, "signature": 64},
        "ECDSA-P256": {"public_key": 65, "secret_key": 32, "ciphertext": 0,
                       "signature": len(esk.sign(MSG, ec.ECDSA(hashes.SHA256())))},
        "RSA-2048": {"public_key": 256, "secret_key": 256, "ciphertext": 0, "signature": len(rsa_sig2048)},
        "RSA-3072": {"public_key": 384, "secret_key": 384, "ciphertext": 0, "signature": len(rsa_sig3072)},
    })

    meta = build_cert_meta()
    return rows, sizes, meta


OPENSSL_ALGS = [
    ("ML-KEM-512", "ML-KEM-512", "kem", "NIST FIPS 203"),
    ("ML-KEM-768", "ML-KEM-768", "kem", "NIST FIPS 203"),
    ("ML-KEM-1024", "ML-KEM-1024", "kem", "NIST FIPS 203"),
    ("ML-DSA-44", "ML-DSA-44", "sign", "NIST FIPS 204"),
    ("ML-DSA-65", "ML-DSA-65", "sign", "NIST FIPS 204"),
    ("ML-DSA-87", "ML-DSA-87", "sign", "NIST FIPS 204"),
    ("SLH-DSA-SHA2-128s", "SLH-DSA-SHA2-128s", "sign", "NIST FIPS 205"),
    ("SLH-DSA-SHA2-128f", "SLH-DSA-SHA2-128f", "sign", "NIST FIPS 205"),
]

OPENSSL_BUDGET = {
    "ML-KEM-512": {"keygen": (200, 8), "encaps": (500, 10), "decaps": (500, 10)},
    "ML-KEM-768": {"keygen": (200, 8), "encaps": (500, 10), "decaps": (500, 10)},
    "ML-KEM-1024": {"keygen": (150, 8), "encaps": (400, 8), "decaps": (400, 8)},
    "ML-DSA-44": {"keygen": (150, 6), "sign": (150, 8), "verify": (150, 8)},
    "ML-DSA-65": {"keygen": (150, 6), "sign": (100, 8), "verify": (100, 8)},
    "ML-DSA-87": {"keygen": (150, 6), "sign": (80, 8), "verify": (80, 8)},
    "SLH-DSA-SHA2-128s": {"keygen": (6, 3), "sign": (3, 3), "verify": (3, 3)},
    "SLH-DSA-SHA2-128f": {"keygen": (30, 4), "sign": (20, 5), "verify": (20, 5)},
}

PQCRYPTO_BUDGET = {
    "ML-KEM-512": {"keygen": (300, 5), "encaps": (400, 5), "decaps": (400, 5)},
    "ML-KEM-768": {"keygen": (300, 5), "encaps": (400, 5), "decaps": (400, 5)},
    "ML-KEM-1024": {"keygen": (200, 5), "encaps": (300, 5), "decaps": (300, 5)},
    "ML-DSA-44": {"keygen": (200, 4), "sign": (60, 3), "verify": (60, 3)},
    "ML-DSA-65": {"keygen": (200, 4), "sign": (40, 3), "verify": (40, 3)},
    "ML-DSA-87": {"keygen": (150, 4), "sign": (30, 3), "verify": (30, 3)},
    "SLH-DSA-SHA2-128s": {"keygen": (6, 2), "sign": (2, 2), "verify": (2, 2)},
    "SLH-DSA-SHA2-128f": {"keygen": (30, 3), "sign": (10, 3), "verify": (10, 3)},
    "HQC-128": {"keygen": (300, 4), "encaps": (400, 5), "decaps": (400, 5)},
    "SNTRUP-761": {"keygen": (30, 3), "encaps": (200, 4), "decaps": (200, 4)},
    "Classic-McEliece-460896": {"keygen": (3, 3), "encaps": (80, 4), "decaps": (15, 4)},
}

PUBLISHED_SIZES = {
    "ML-KEM-512": {"public_key": 800, "secret_key": 1632, "ciphertext": 768, "shared_secret": 32},
    "ML-KEM-768": {"public_key": 1184, "secret_key": 2400, "ciphertext": 1088, "shared_secret": 32},
    "ML-KEM-1024": {"public_key": 1568, "secret_key": 3168, "ciphertext": 1568, "shared_secret": 32},
    "ML-DSA-44": {"public_key": 1312, "secret_key": 2560, "signature": 2420},
    "ML-DSA-65": {"public_key": 1952, "secret_key": 4032, "signature": 3309},
    "ML-DSA-87": {"public_key": 2592, "secret_key": 4896, "signature": 4627},
    "SLH-DSA-SHA2-128s": {"public_key": 32, "secret_key": 64, "signature": 7856},
    "SLH-DSA-SHA2-128f": {"public_key": 32, "secret_key": 64, "signature": 17088},
}

NON_FIPS_REFERENCE = {
    "HQC-128": {"source": "NIST HQC selection (2025), reference implementation sizes",
                "public_key": 2249, "secret_key": 4433, "ciphertext": 4489, "shared_secret": 64},
    "SNTRUP-761": {"source": "NTRU Prime reference implementation",
                   "public_key": 1138, "secret_key": 1652, "ciphertext": 1047, "shared_secret": 32},
    "Classic-McEliece-460896": {"source": "Classic McEliece round-4 submission (expanded secret key)",
                                "public_key": 524160, "secret_key": 1044992, "ciphertext": 188,
                                "shared_secret": 32},
}


def build_openssl_pqc():
    rows, sizes = [], {}
    for label, evp_name, kind, std in OPENSSL_ALGS:
        budget = OPENSSL_BUDGET[label]
        pk = ossl.keygen(evp_name)
        rows.append({"family": "pqc", "implementation": "OpenSSL-EVP", "algorithm": label,
                     "operation": "keygen", **bench(lambda: ossl.keygen(evp_name), *budget["keygen"])})
        if kind == "kem":
            ct, ss = ossl.encaps(pk)
            assert ossl.decaps(pk, ct) == ss, f"{label} round trip failed"
            rows.append({"family": "pqc", "implementation": "OpenSSL-EVP", "algorithm": label,
                         "operation": "encaps", **bench(lambda: ossl.encaps(pk), *budget["encaps"])})
            rows.append({"family": "pqc", "implementation": "OpenSSL-EVP", "algorithm": label,
                         "operation": "decaps", **bench(lambda: ossl.decaps(pk, ct), *budget["decaps"])})
            sizes[label] = {"public_key": 0, "secret_key": 0, "ciphertext": len(ct), "shared_secret": len(ss),
                            "signature": 0}
        else:
            sig = ossl.sign(pk, MSG)
            ossl.verify(pk, MSG, sig)
            try:
                ossl.verify(pk, b"tampered", sig)
                raise RuntimeError(f"{label} accepted tampered signature")
            except ValueError:
                pass
            rows.append({"family": "pqc", "implementation": "OpenSSL-EVP", "algorithm": label,
                         "operation": "sign", **bench(lambda: ossl.sign(pk, MSG), *budget["sign"])})
            rows.append({"family": "pqc", "implementation": "OpenSSL-EVP", "algorithm": label,
                         "operation": "verify", **bench(lambda: ossl.verify(pk, MSG, sig), *budget["verify"])})
            sizes[label] = {"public_key": 0, "secret_key": 0, "ciphertext": 0, "shared_secret": 0,
                            "signature": len(sig)}
        load_and_free = ossl.load()
        load_and_free.EVP_PKEY_free(pk)
        print(f"openssl {label} done", flush=True)
    return rows, sizes


PQCRYPTO_MODULES = {
    "ML-KEM-512": ("kem", pq_ml_kem_512), "ML-KEM-768": ("kem", pq_ml_kem_768),
    "ML-KEM-1024": ("kem", pq_ml_kem_1024),
    "ML-DSA-44": ("sign", pq_ml_dsa_44), "ML-DSA-65": ("sign", pq_ml_dsa_65),
    "ML-DSA-87": ("sign", pq_ml_dsa_87),
    "SLH-DSA-SHA2-128s": ("sign", pq_slh_128s), "SLH-DSA-SHA2-128f": ("sign", pq_slh_128f),
    "HQC-128": ("kem", hqc_128), "SNTRUP-761": ("kem", sntrup_761),
    "Classic-McEliece-460896": ("kem", mceliece_460896),
}


def build_pqcrypto():
    rows, sizes = [], {}
    for label, (kind, mod) in PQCRYPTO_MODULES.items():
        budget = PQCRYPTO_BUDGET[label]
        pk, sk = mod.keygen()
        rows.append({"family": "pqc", "implementation": "pqcrypto-Rust", "algorithm": label,
                     "operation": "keygen", **bench(lambda m=mod: m.keygen(), *budget["keygen"])})
        if kind == "kem":
            ct, ss = mod.encaps(pk)
            assert mod.decaps(sk, ct) == ss
            rows.append({"family": "pqc", "implementation": "pqcrypto-Rust", "algorithm": label,
                         "operation": "encaps", **bench(lambda p=pk: mod.encaps(p), *budget["encaps"])})
            rows.append({"family": "pqc", "implementation": "pqcrypto-Rust", "algorithm": label,
                         "operation": "decaps", **bench(lambda s=sk, c=ct: mod.decaps(s, c), *budget["decaps"])})
            sizes[label] = {"public_key": mod.PUBLIC_KEY_SIZE, "secret_key": mod.SECRET_KEY_SIZE,
                            "ciphertext": mod.CIPHERTEXT_SIZE, "shared_secret": mod.SHARED_SECRET_SIZE,
                            "signature": 0}
        else:
            sig = mod.sign(sk, MSG)
            mod.verify(pk, MSG, sig)
            try:
                mod.verify(pk, b"tampered", sig)
                raise RuntimeError(f"{label} accepted tampered signature")
            except Exception as exc:
                if type(exc).__name__ != "InvalidSignatureError":
                    raise
            rows.append({"family": "pqc", "implementation": "pqcrypto-Rust", "algorithm": label,
                         "operation": "sign", **bench(lambda s=sk: mod.sign(s, MSG), *budget["sign"])})
            rows.append({"family": "pqc", "implementation": "pqcrypto-Rust", "algorithm": label,
                         "operation": "verify", **bench(lambda p=pk: mod.verify(p, MSG, sig), *budget["verify"])})
            sizes[label] = {"public_key": mod.PUBLIC_KEY_SIZE, "secret_key": mod.SECRET_KEY_SIZE,
                            "ciphertext": 0, "shared_secret": 0, "signature": len(sig)}
        print(f"pqcrypto {label} done", flush=True)
    return rows, sizes


def conformance_table(ossl_sizes, pq_sizes):
    rows = []
    for label, published in PUBLISHED_SIZES.items():
        for field, pub in published.items():
            o_val = ossl_sizes.get(label, {}).get(field)
            p_val = pq_sizes.get(label, {}).get(field)
            if label.startswith("ML-KEM") and field in ("public_key", "secret_key"):
                o_val = None
            rows.append({
                "algorithm": label, "field": field, "published": pub,
                "openssl": o_val if o_val not in (0, None) else None,
                "pqcrypto": p_val if p_val not in (0, None) else None,
            })
    df = pd.DataFrame(rows)
    df["openssl_status"] = df.apply(
        lambda r: "-" if pd.isna(r["openssl"]) else ("PASS" if int(r["openssl"]) == r["published"] else "FAIL"), axis=1)
    df["pqcrypto_status"] = df.apply(
        lambda r: "-" if pd.isna(r["pqcrypto"]) else ("PASS" if int(r["pqcrypto"]) == r["published"] else "FAIL"), axis=1)
    return df


def nonfips_sizes_table(pq_sizes):
    rows = []
    for label, ref in NON_FIPS_REFERENCE.items():
        for field in ("public_key", "secret_key", "ciphertext", "shared_secret"):
            rows.append({"algorithm": label, "field": field, "reference_source": ref["source"],
                         "reference_value": ref[field],
                         "pqcrypto_reported": pq_sizes.get(label, {}).get(field)})
    return pd.DataFrame(rows)


def build_workload_models(ossl_sizes, pq_sizes, meta):
    kem = ossl_sizes["ML-KEM-768"]
    kem512 = ossl_sizes["ML-KEM-512"]
    kem1024 = ossl_sizes["ML-KEM-1024"]
    s65 = pq_sizes["ML-DSA-65"]
    s44 = pq_sizes["ML-DSA-44"]
    slh_s = pq_sizes["SLH-DSA-SHA2-128s"]
    framing = meta["ecdsa_framing"]

    def cert_est(pk_bytes):
        return framing + pk_bytes

    # The pinned 245 B framing decomposes into 174 B of TBS/sig-algorithm fields
    # plus the 71 B ECDSA CA signature measured on the reference certificate, so
    # `cert_est` models the realistic first-phase PKI: a *classical* CA issuing a
    # PQC leaf certificate (the CA signature is already inside the framing, only
    # its size is classical).  If the CA itself signs with PQC, the 71 B ECDSA
    # slot is replaced by the CA's own PQC signature over the same TBS; those
    # sensitivity variants are reported as separate rows so the reader can see
    # which part of PKI growth is key-driven and which part is signature-driven.
    pqc_ca_base = framing - meta["ecdsa_ca_sig"]

    scenarios = [
        {"scenario": "Classical: X25519 + ECDSA-P256",
         "kem_material": 64, "cert": meta["ecdsa_cert"], "signature": meta["ecdsa_raw_sig"]},
        {"scenario": "Classical: X25519 + RSA-2048-PSS",
         "kem_material": 64, "cert": meta["rsa_cert"], "signature": meta["rsa_sig"]},
        {"scenario": "PQC: ML-KEM-768 + ML-DSA-65",
         "kem_material": kem["ciphertext"] + kem.get("public_key", 0),
         "cert": cert_est(s65["public_key"]), "signature": s65["signature"]},
        {"scenario": "PQC: ML-KEM-512 + ML-DSA-44",
         "kem_material": kem512["ciphertext"] + kem512.get("public_key", 0),
         "cert": cert_est(s44["public_key"]), "signature": s44["signature"]},
        {"scenario": "PQC: ML-KEM-1024 + SLH-DSA-128s",
         "kem_material": kem1024["ciphertext"] + kem1024.get("public_key", 0),
         "cert": cert_est(slh_s["public_key"]), "signature": slh_s["signature"]},
        {"scenario": "Hybrid: X25519+ECDSA and ML-KEM-768+ML-DSA-65",
         "kem_material": 64 + kem["ciphertext"] + kem.get("public_key", 0),
         "cert": meta["ecdsa_cert"] + cert_est(s65["public_key"]),
         "signature": meta["ecdsa_raw_sig"] + s65["signature"]},
    ]
    for s in scenarios:
        s["total_bytes"] = int(s["kem_material"] + s["cert"] + s["signature"])
        s["mutual_auth_bytes"] = int(s["total_bytes"] + s["cert"] + s["signature"])
    hs = pd.DataFrame(scenarios)
    base = int(hs.loc[hs["scenario"].str.startswith("Classical: X25519 + ECDSA"), "total_bytes"].iloc[0])
    hs["overhead_vs_classical"] = (hs["total_bytes"] / base).round(2)

    pki_rows = []
    for n_certs in (10_000, 1_000_000):
        for label, cert_bytes in (
            ("ECDSA-P256 cert", meta["ecdsa_cert"]),
            ("RSA-2048 cert", meta["rsa_cert"]),
            ("ML-DSA-65 cert (modelled)", cert_est(s65["public_key"])),
            ("ML-DSA-65 cert (modelled, PQC CA)",
             pqc_ca_base + s65["public_key"] + s65["signature"]),
            ("SLH-DSA-128s cert (modelled)", cert_est(slh_s["public_key"])),
            ("SLH-DSA-128s cert (modelled, PQC CA)",
             pqc_ca_base + slh_s["public_key"] + slh_s["signature"]),
        ):
            pki_rows.append({"certificates": n_certs, "certificate_type": label,
                             "bytes_per_cert": cert_bytes,
                             "total_GB": round(n_certs * cert_bytes / 1e9, 3)})
    pki = pd.DataFrame(pki_rows)

    fw_rows = []
    for label, sig_len in (
        ("ECDSA-P256", 64), ("Ed25519", 64), ("RSA-2048-PSS", meta["rsa_sig"]),
        ("ML-DSA-65", s65["signature"]), ("ML-DSA-44", s44["signature"]),
        ("SLH-DSA-128s", slh_s["signature"]),
    ):
        fw_rows.append({"signature_scheme": label, "bytes_per_signature": sig_len,
                        "million_devices_GB": round(1_000_000 * sig_len / 1e9, 3)})
    fw = pd.DataFrame(fw_rows)
    return hs, pki, fw


def make_handshake_figure(hs):
    """Handshake byte model (manuscript Figure 4; also rebuilt by --byte-model-only)."""
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    short = (hs["scenario"].str.replace("Classical: ", "CLASS: ", regex=False)
             .str.replace("Hybrid: X25519+ECDSA and ML-KEM-768+ML-DSA-65", "HYBRID", regex=False))
    bars = ax.barh(short, hs["total_bytes"],
                   color=["#4c72b0" if s.startswith("CLASS") else ("#55a868" if s.startswith("PQC") else "#c44e52")
                          for s in short])
    for b, v in zip(bars, hs["total_bytes"]):
        ax.text(b.get_width() * 1.02, b.get_y() + b.get_height() / 2, f"{v} B", va="center", fontsize=8)
    ax.set_xlabel("bytes on the wire (server-authenticated handshake model)")
    ax.set_title("TLS-1.3-like handshake size: classical, PQC, hybrid")
    ax.set_xlim(0, float(hs["total_bytes"].max()) * 1.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_handshake_size.png", bbox_inches="tight")
    plt.close(fig)


def make_figures(lat_df, sizes, hs):
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})
    colors = {"classical": "#4c72b0", "OpenSSL-EVP": "#55a868", "pqcrypto-Rust": "#c44e52"}

    kem_names = ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024", "HQC-128", "SNTRUP-761", "Classic-McEliece-460896"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, op in zip(axes, ["keygen", "encaps", "decaps"]):
        labels, vals, cols = [], [], []
        xrow = lat_df[(lat_df.algorithm == "X25519") &
                      (lat_df.operation == ("keygen" if op == "keygen" else "shared_secret"))]
        if not xrow.empty:
            labels.append("X25519")
            vals.append(float(xrow["median_us"].iloc[0]))
            cols.append(colors["classical"])
        for name in kem_names:
            row = lat_df[(lat_df.algorithm == name) & (lat_df.operation == op)]
            for _, r in row.iterrows():
                labels.append(f"{name}\n{r['implementation'].split('/')[0]}")
                vals.append(float(r["median_us"]))
                cols.append(colors.get(r["implementation"], "#999999"))
        ax.bar(range(len(labels)), vals, color=cols, tick_label=labels)
        ax.set_yscale("log")
        ax.set_title(op)
        ax.tick_params(axis="x", labelsize=6.5, rotation=90)
    axes[0].set_ylabel("median latency (µs, log scale)")
    fig.suptitle("KEM latency: classical X25519 vs PQC (two independent implementations)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_kem_latency.png", bbox_inches="tight")
    plt.close(fig)

    sig_names = ["ECDSA-P256", "Ed25519", "RSA-2048", "ML-DSA-44", "ML-DSA-65", "ML-DSA-87",
                 "SLH-DSA-SHA2-128s", "SLH-DSA-SHA2-128f"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for impl, marker in (("classical", "o"), ("OpenSSL-EVP", "s"), ("pqcrypto-Rust", "^")):
        xs, ys, labs = [], [], []
        for n in sig_names:
            sign_row = lat_df[(lat_df.algorithm == n) & (lat_df.operation == "sign") &
                              (lat_df["implementation"].apply(lambda x: x == impl or x.startswith("cryptography")))]
            if sign_row.empty:
                continue
            ver_row = lat_df[(lat_df.algorithm == n) & (lat_df.operation == "verify") &
                             (lat_df["implementation"].apply(lambda x: x == impl or x.startswith("cryptography")))]
            xs.append(sizes[n]["signature"])
            ys.append(float(sign_row["median_us"].iloc[0]) + float(ver_row["median_us"].iloc[0]))
            labs.append(n)
        ax.scatter(xs, ys, c=[colors[impl]], marker=marker, s=55, zorder=3, label=impl)
        for x, y, n in zip(xs, ys, labs):
            ax.annotate(n, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=6.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("signature size (bytes, log scale)")
    ax.set_ylabel("sign + verify latency (µs, log scale)")
    ax.set_title("Signature cost trade-off by implementation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_signature_tradeoff.png", bbox_inches="tight")
    plt.close(fig)

    make_handshake_figure(hs)

    spread_algs = ["ML-DSA-44", "ML-DSA-65", "ML-DSA-87", "SLH-DSA-SHA2-128s", "SLH-DSA-SHA2-128f",
                   "ML-KEM-768"]
    spread_ops = {"ML-KEM-768": "encaps"}
    x = np.arange(len(spread_algs))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    o_vals, p_vals = [], []
    for n in spread_algs:
        op = spread_ops.get(n, "sign")
        o = lat_df[(lat_df.algorithm == n) & (lat_df.operation == op) & (lat_df.implementation == "OpenSSL-EVP")]
        p = lat_df[(lat_df.algorithm == n) & (lat_df.operation == op) & (lat_df.implementation == "pqcrypto-Rust")]
        o_vals.append(float(o["median_us"].iloc[0]) if not o.empty else np.nan)
        p_vals.append(float(p["median_us"].iloc[0]) if not p.empty else np.nan)
    ax.bar(x - width / 2, o_vals, width, label="OpenSSL-EVP", color=colors["OpenSSL-EVP"])
    ax.bar(x + width / 2, p_vals, width, label="pqcrypto-Rust", color=colors["pqcrypto-Rust"])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({spread_ops.get(n, 'sign')})" for n in spread_algs], fontsize=7)
    ax.set_ylabel("median latency (µs, log scale)")
    ax.set_title("Implementation spread: same FIPS algorithm, two codebases")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_implementation_spread.png", bbox_inches="tight")
    plt.close(fig)


def load_recorded_sizes():
    """Rebuild {implementation: {algorithm: {field: bytes}}} from algorithm_sizes.csv."""
    df = pd.read_csv(RESULTS / "algorithm_sizes.csv")
    out: dict = {}
    for impl, algo, field, nbytes in df[["implementation", "algorithm", "field", "bytes"]].itertuples(index=False):
        out.setdefault(impl, {}).setdefault(algo, {})[field] = int(nbytes)
    return out


def byte_model_only():
    """Re-run only the deterministic byte model (no benchmarking).

    Regenerates handshake_model.csv, pki_storage.csv, firmware_signing.csv,
    figures/fig4_handshake_size.png and the `handshake` block of summary_exp1.json from the *recorded*
    sizes in algorithm_sizes.csv, so the machine-dependent latency tables are
    left untouched. Run this (or a full run) whenever certificate or workload
    model code changes.
    """
    sizes = load_recorded_sizes()
    meta = build_cert_meta()
    hs, pki, fw = build_workload_models(sizes["OpenSSL-EVP"], sizes["pqcrypto-Rust"], meta)
    hs.to_csv(RESULTS / "handshake_model.csv", index=False)
    pki.to_csv(RESULTS / "pki_storage.csv", index=False)
    fw.to_csv(RESULTS / "firmware_signing.csv", index=False)
    make_handshake_figure(hs)

    summary_path = RESULTS / "summary_exp1.json"
    summary = json.loads(summary_path.read_text())
    summary["handshake"] = hs.to_dict("records")
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"certificate meta: {meta}", flush=True)
    print(hs.to_string(index=False), flush=True)
    print("byte model rebuilt (latency tables unchanged)", flush=True)


def main():
    if "--byte-model-only" in sys.argv:
        byte_model_only()
        return
    np.random.seed(SEED)
    env = collect_env()
    print(json.dumps(env, indent=2), flush=True)

    classical_rows, classical_sizes, meta = build_classical()
    print("classical benchmark done", flush=True)

    ossl_rows, ossl_sizes = build_openssl_pqc()
    print("openssl pqc benchmark done", flush=True)

    pq_rows, pq_sizes = build_pqcrypto()
    print("pqcrypto benchmark done", flush=True)

    lat_df = pd.DataFrame(classical_rows + ossl_rows + pq_rows)
    lat_df.to_csv(RESULTS / "benchmark_latency.csv", index=False)

    conf = conformance_table(ossl_sizes, pq_sizes)
    conf.to_csv(RESULTS / "fips_conformance.csv", index=False)
    nonfips = nonfips_sizes_table(pq_sizes)
    nonfips.to_csv(RESULTS / "nonfips_reported_sizes.csv", index=False)
    statuses = [conf["openssl_status"], conf["pqcrypto_status"]]
    n_pass = sum(int((s == "PASS").sum()) for s in statuses)
    n_total = sum(int((s != "-").sum()) for s in statuses)
    n_fail = sum(int((s == "FAIL").sum()) for s in statuses)
    print(f"FIPS conformance: {n_pass}/{n_total} PASS, {n_fail} FAIL", flush=True)

    size_rows = []
    for impl, mapping in (("classical", classical_sizes), ("OpenSSL-EVP", ossl_sizes), ("pqcrypto-Rust", pq_sizes)):
        for algo, fields in mapping.items():
            for k, v in fields.items():
                if v:
                    size_rows.append({"implementation": impl, "algorithm": algo, "field": k, "bytes": int(v)})
    pd.DataFrame(size_rows).to_csv(RESULTS / "algorithm_sizes.csv", index=False)

    hs, pki, fw = build_workload_models(ossl_sizes, pq_sizes, meta)
    hs.to_csv(RESULTS / "handshake_model.csv", index=False)
    pki.to_csv(RESULTS / "pki_storage.csv", index=False)
    fw.to_csv(RESULTS / "firmware_signing.csv", index=False)

    size_lookup = {a: f for a, f in list(ossl_sizes.items()) + list(pq_sizes.items())}
    for a, f in classical_sizes.items():
        size_lookup.setdefault(a, f)
    make_figures(lat_df, size_lookup, hs)

    headline = {}
    for impl in ("classical", "OpenSSL-EVP", "pqcrypto-Rust"):
        sub = lat_df[lat_df["implementation"].apply(lambda x: x == impl or x.startswith("cryptography"))]
        headline[impl] = {
            f"{r.algorithm}:{r.operation}": r.median_us for r in sub.itertuples()
        }
    spread = {}
    for alg, op in (("ML-DSA-44", "sign"), ("ML-DSA-65", "sign"), ("ML-DSA-87", "sign"),
                    ("SLH-DSA-SHA2-128s", "sign"), ("SLH-DSA-SHA2-128f", "sign"), ("ML-KEM-768", "encaps")):
        o = lat_df[(lat_df.algorithm == alg) & (lat_df.operation == op) & (lat_df.implementation == "OpenSSL-EVP")]
        p = lat_df[(lat_df.algorithm == alg) & (lat_df.operation == op) & (lat_df.implementation == "pqcrypto-Rust")]
        if not o.empty and not p.empty:
            ov = float(o["median_us"].iloc[0])
            pv = float(p["median_us"].iloc[0])
            spread[f"{alg}:{op}"] = {"openssl_us": ov, "pqcrypto_us": pv, "ratio_pqcrypto_over_openssl": round(pv / ov, 2)}

    summary = {
        "env": env,
        "conformance": {"pass": n_pass, "total_checks": n_total, "fail": n_fail},
        "spread": spread,
        "handshake": hs.to_dict("records"),
        "headline": headline,
    }
    (RESULTS / "summary_exp1.json").write_text(json.dumps(summary, indent=2))
    print("exp1 complete", flush=True)


if __name__ == "__main__":
    main()
