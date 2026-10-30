- Package purpose: Reproducible experiment on post-quantum cryptography migration for critical infrastructure.
- Two experiments:
  1. `exp1_crypto_benchmark.py` — benchmarks NIST PQC standards and classical crypto, with FIPS size checks and critical-infrastructure workload models.
  2. `exp2_migration_model.py` — analytical HNDL migration decision model, validated by quadrature/Monte-Carlo, plus sector portfolio and policy-value analysis.
- Main files: `exp1_crypto_benchmark.py`, `exp2_migration_model.py`, `openssl_evp.py`, `run_all.py`, `requirements.txt`, `results/`.
- Setup: Python ≥3.10; install `requirements.txt`; OpenSSL ≥3.5 needed for PQC harness.
- Run: `python run_all.py` or run each experiment script individually.
- Verify only: `python run_all.py --verify` → expected `27 entries checked, 0 mismatches`.
- Expected integrity checks:
  - Experiment 1: `FIPS conformance: 38/38 PASS, 0 FAIL`
  - Experiment 2: all seven validation rows `PASS`
- Determinism: seed `20261023`; Experiment 2 deterministic given seed/package versions; Experiment 1 latency is machine-dependent, but sizes/conformance/byte models are deterministic.
- Recorded environment: Python 3.12.10, Windows 10, Intel i5-8265U, OpenSSL 4.0.2 for PQC, OpenSSL 4.0.0 for classical, pqcrypto 1.0.0.
- Runtime: about 5–8 minutes on a laptop-class CPU.
