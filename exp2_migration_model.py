import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate, stats

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

SEED = 20261023
rng = np.random.default_rng(SEED)

SCENARIOS = {
    "early": {"dist": "lognorm", "median": 3.0, "sigma": 0.25, "note": "median CRQC arrival 2029"},
    "central": {"dist": "lognorm", "median": 9.0, "sigma": 0.35, "note": "median CRQC arrival 2035"},
    "late": {"dist": "lognorm", "median": 19.0, "sigma": 0.45, "note": "median CRQC arrival 2045"},
    "exp-matched": {"dist": "exp", "median": 9.0, "sigma": None, "note": "exponential, median 2035"},
}


def make_dist(name):
    s = SCENARIOS[name]
    if s["dist"] == "lognorm":
        return stats.lognorm(s=s["sigma"], scale=s["median"])
    lam = np.log(2.0) / s["median"]
    return stats.expon(scale=1.0 / lam)


def make_pdf(name):
    d = make_dist(name)
    return d.pdf


def exposure_closed_form_exp(m, L, lam):
    return m - np.exp(-lam * L) * (1.0 - np.exp(-lam * m)) / lam


def exposure_quad(m, L, dist):
    f = lambda t: dist.cdf(t + L)
    val, _ = integrate.quad(f, 0.0, m, limit=200)
    return val


def exposure_mc(T, m, L):
    return np.mean(m - np.clip(T - L, 0.0, m))


def exposure_vectorized(m, L, dist, nodes=48):
    x, w = np.polynomial.legendre.leggauss(nodes)
    m = np.atleast_1d(np.asarray(m, dtype=float))
    L = np.atleast_1d(np.asarray(L, dtype=float))
    t = 0.5 * m[..., None] * (x + 1.0)
    ww = 0.5 * m[..., None] * w
    return np.sum(ww * dist.cdf(t + L[..., None]), axis=-1)


def cost_vectorized(m, v, L, k, dist):
    return v * exposure_vectorized(m, L, dist) + k / m


def solve_optimal_deadline(v, L, k, dist):
    v = np.atleast_1d(np.asarray(v, dtype=float))
    L = np.atleast_1d(np.asarray(L, dtype=float))
    k = np.atleast_1d(np.asarray(k, dtype=float))

    def g(m):
        return v * dist.cdf(m + L) - k / (m ** 2)

    lo = np.full_like(v, 1e-8)
    hi = np.full_like(v, 1.0)
    for _ in range(60):
        need = g(hi) <= 0
        if not np.any(need):
            break
        hi = np.where(need, hi * 2.0, hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        pos = g(mid) > 0
        hi = np.where(pos, mid, hi)
        lo = np.where(pos, lo, mid)
    return 0.5 * (lo + hi)


def theorem1_validation():
    lam = np.log(2.0) / SCENARIOS["exp-matched"]["median"]
    dist = make_dist("exp-matched")
    rows = []
    for L in (1.0, 5.0, 10.0, 20.0):
        for m in (1.0, 3.0, 5.0, 10.0, 19.0, 30.0):
            closed = exposure_closed_form_exp(m, L, lam)
            quad = exposure_quad(m, L, dist)
            T = dist.rvs(size=200_000, random_state=rng)
            mc = exposure_mc(T, m, L)
            se = np.std(m - np.clip(T - L, 0, m), ddof=1) / np.sqrt(len(T))
            rows.append({
                "L": L, "m": m, "closed_form": closed, "quadrature": quad,
                "quadrature_rel_err": abs(quad - closed) / closed,
                "mc": mc, "mc_z": abs(mc - closed) / se if se > 0 else 0.0,
            })
    df = pd.DataFrame(rows)
    return df


def theorem2_validation(dist, n_assets=400):
    v = rng.lognormal(mean=0.0, sigma=1.0, size=n_assets)
    L = rng.uniform(1.0, 30.0, size=n_assets)
    k = rng.lognormal(mean=0.0, sigma=0.75, size=n_assets)
    m_star = solve_optimal_deadline(v, L, k, dist)
    grid = np.exp(np.linspace(np.log(0.05), np.log(40.0), 1200))
    dev = []
    for i in range(n_assets):
        vals = cost_vectorized(grid, v[i], L[i], k[i], dist)
        m_grid = grid[np.argmin(vals)]
        c_star = float(cost_vectorized(m_star[i:i + 1], v[i], L[i], k[i], dist)[0])
        c_grid = float(np.min(vals))
        dev.append((abs(m_grid - m_star[i]) / m_star[i], abs(c_grid - c_star) / c_star))
    dev = np.array(dev)
    g0 = v * dist.cdf(1e-6 + L) - k / (1e-6 ** 2)
    g1 = v * dist.cdf(1e5 + L) - k / (1e5 ** 2)
    return pd.DataFrame({
        "asset": np.arange(n_assets), "v": v, "L": L, "k": k, "m_star": m_star,
        "grid_rel_dev": dev[:, 0], "cost_rel_dev": dev[:, 1],
        "g_at_lo": g0, "g_at_hi": g1,
    })


def theorem3_validation(dist):
    rows = []
    v_grid = np.logspace(-1, 2, 12)
    k_grid = np.logspace(-1, 2, 12)
    L_grid = np.array([1.0, 5.0, 10.0, 20.0, 30.0])
    for L in L_grid:
        base = solve_optimal_deadline(v_grid, L, np.ones_like(v_grid), dist)
        rows.append({"direction": "v", "L": L,
                     "monotone_decreasing": bool(np.all(np.diff(base) < 0))})
    for L in L_grid:
        base = solve_optimal_deadline(np.ones_like(k_grid), L, k_grid, dist)
        rows.append({"direction": "k", "L": L,
                     "monotone_increasing": bool(np.all(np.diff(base) > 0))})
    for v0 in (0.3, 1.0, 10.0, 100.0):
        base = solve_optimal_deadline(v0, L_grid, np.ones_like(L_grid), dist)
        rows.append({"direction": "L", "v": v0,
                     "monotone_decreasing": bool(np.all(np.diff(base) < 0))})
    return pd.DataFrame(rows)


def theorem4_validation(n_draws=200_000):
    rows = []
    for scen in SCENARIOS:
        dist = make_dist(scen)
        for eps in (0.05, 0.10, 0.25):
            q = dist.ppf(eps)
            for tau in (1.0, 2.0, 5.0, 10.0):
                feasible = tau <= q
                s_star = q - tau if feasible else np.nan
                c = tau if not feasible else s_star + tau
                T = dist.rvs(size=n_draws, random_state=rng)
                p_arrive_before = float(np.mean(T <= c))
                se = np.sqrt(max(p_arrive_before * (1 - p_arrive_before), 1e-12) / n_draws)
                rows.append({
                    "scenario": scen, "epsilon": eps, "tau": tau,
                    "F_inv_eps": q, "feasible": feasible, "latest_start": s_star,
                    "completion": c, "empirical_P_arrival_before_completion": p_arrive_before,
                    "mc_ok": bool(p_arrive_before <= eps + 4 * se) if feasible else bool(p_arrive_before > eps),
                })
    return pd.DataFrame(rows)


SECTORS = {
    "Electricity grid": {"L": (10, 30), "tau": (1, 4, 10), "criticality": 5.0, "legacy": 1.3},
    "Healthcare": {"L": (10, 30), "tau": (1, 5, 12), "criticality": 4.0, "legacy": 1.2},
    "Government & defence": {"L": (10, 25), "tau": (1, 4, 9), "criticality": 5.0, "legacy": 1.1},
    "Financial services": {"L": (3, 15), "tau": (0.5, 2, 6), "criticality": 3.0, "legacy": 0.9},
    "Telecommunications": {"L": (1, 5), "tau": (0.5, 2, 5), "criticality": 2.0, "legacy": 0.8},
    "Rail & transport": {"L": (5, 20), "tau": (1, 3, 8), "criticality": 3.0, "legacy": 1.4},
}


def draw_sector_portfolio(n_per_sector=3000):
    """Draw the asset inventory once: six sectors x n_per_sector assets.

    The *same* assets are re-used for every CRQC arrival scenario (common
    random numbers), so differences between scenarios isolate the arrival
    distribution F instead of mixing in portfolio-composition noise.  Draw
    order per sector is L, tau, v, k -- keep it stable, the seed is 20261023.
    """
    frames = []
    for name, p in SECTORS.items():
        L = rng.uniform(p["L"][0], p["L"][1], n_per_sector)
        tau = rng.triangular(p["tau"][0], p["tau"][1], p["tau"][2], n_per_sector)
        v = p["criticality"] * rng.lognormal(0.0, 0.7, n_per_sector)
        k = p["legacy"] * rng.lognormal(0.0, 0.5, n_per_sector)
        frames.append(pd.DataFrame({
            "sector": name, "v": v, "L": L, "tau": tau, "k": k,
        }))
    return pd.concat(frames, ignore_index=True)


def evaluate_sector_portfolio(portfolio, dist, eps=0.10):
    """Attach model outputs (m*, MUI, deadline deficit, urgency classes)."""
    out = portfolio.copy()
    m_star = solve_optimal_deadline(out["v"].to_numpy(), out["L"].to_numpy(),
                                    out["k"].to_numpy(), dist)
    out["m_star"] = m_star
    out["MUI"] = out["tau"].to_numpy() / m_star
    delta = dist.ppf(eps) - out["tau"].to_numpy()
    out["deadline_deficit"] = delta
    out["urgent"] = out["MUI"] >= 1.0
    out["high"] = (out["MUI"] >= 0.5) & (out["MUI"] < 1.0)
    out["infeasible_epsilon"] = delta < 0
    return out


def policy_value(portfolio, dist):
    policies = {
        "immediate (m = tau)": None,
        "model-optimal (m = m*)": None,
        "fixed 2035 (m = 9)": 9.0,
        "fixed 2040 (m = 14)": 14.0,
    }
    m_opt = portfolio["m_star"].to_numpy()
    m_imm = np.maximum(portfolio["tau"].to_numpy(), 0.25)
    costs = {}
    costs["immediate (m = tau)"] = cost_vectorized(m_imm, portfolio["v"].to_numpy(),
                                                   portfolio["L"].to_numpy(), portfolio["k"].to_numpy(), dist)
    costs["model-optimal (m = m*)"] = cost_vectorized(m_opt, portfolio["v"].to_numpy(),
                                                      portfolio["L"].to_numpy(), portfolio["k"].to_numpy(), dist)
    costs["fixed 2035 (m = 9)"] = cost_vectorized(np.full_like(m_opt, 9.0), portfolio["v"].to_numpy(),
                                                  portfolio["L"].to_numpy(), portfolio["k"].to_numpy(), dist)
    costs["fixed 2040 (m = 14)"] = cost_vectorized(np.full_like(m_opt, 14.0), portfolio["v"].to_numpy(),
                                                   portfolio["L"].to_numpy(), portfolio["k"].to_numpy(), dist)
    rows = []
    for name, c in costs.items():
        rows.append({"policy": name, "mean_cost": float(np.mean(c)), "median_cost": float(np.median(c)),
                     "p90_cost": float(np.percentile(c, 90))})
    opt = costs["model-optimal (m = m*)"]
    for name, c in costs.items():
        if name.startswith("model-optimal"):
            continue
        diff = c - opt
        se = np.std(diff, ddof=1) / np.sqrt(len(diff))
        rows.append({"policy": name, "mean_cost": float(np.mean(c)), "median_cost": float(np.median(c)),
                     "p90_cost": float(np.percentile(c, 90)),
                     "mean_regret_vs_optimal": float(np.mean(diff)),
                     "regret_ci_low": float(np.mean(diff) - 1.96 * se),
                     "regret_ci_high": float(np.mean(diff) + 1.96 * se)})
    return pd.DataFrame(rows), costs


def grover_table():
    rows = []
    for name, classical_bits, mode in (
        ("AES-128", 128, "key search"), ("AES-256", 256, "key search"),
        ("ChaCha20-256", 256, "key search"), ("SHA-256 preimage", 256, "preimage search"),
        ("SHA-256 collision", 256, "quantum collision"),
    ):
        factor = 1 / 3.0 if mode == "quantum collision" else 0.5
        q_bits = classical_bits * factor
        rows.append({"primitive": name, "mode": mode, "classical_bits": classical_bits,
                     "quantum_bits": q_bits, "key_length_restore_factor": 2.0 if mode != "quantum collision" else 3.0})
    return pd.DataFrame(rows)


def make_figures(exposure_df, t2, sectors_df, policies_df, costs, dist_central):
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})
    dist_c = make_dist("central")
    m_grid = np.linspace(0.01, 20, 200)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for L in (1, 5, 10, 20, 30):
        e = exposure_vectorized(m_grid, np.full_like(m_grid, float(L)), dist_c)
        ax.plot(m_grid, e, label=f"L = {L} yr")
    ax.set_xlabel("migration completion deadline m (years from now)")
    ax.set_ylabel("expected exposed record-years")
    ax.set_title("Harvest-now-decrypt-later exposure (central CRQC scenario)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig9_hndl_exposure.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4))
    for i, (v0, L0, k0) in enumerate(((5.0, 20.0, 1.0), (1.0, 10.0, 1.0), (10.0, 5.0, 2.0))):
        m_star = float(solve_optimal_deadline(v0, L0, k0, dist_c)[0])
        c = cost_vectorized(m_grid, v0, L0, k0, dist_c)
        ax.plot(m_grid, c, label=f"v={v0}, L={L0}, k={k0}  (m*={m_star:.2f})")
        ax.axvline(m_star, linestyle="--", alpha=0.4)
    ax.set_xlabel("deadline m (years)")
    ax.set_ylabel("total expected cost C(m)")
    ax.set_yscale("log")
    ax.set_title("Strict convexity of the migration cost functional")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig7_cost_convexity.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4))
    v_grid = np.logspace(-1, 2, 60)
    for L in (1.0, 5.0, 10.0, 20.0):
        m_star = solve_optimal_deadline(v_grid, L, np.ones_like(v_grid), dist_c)
        ax.plot(v_grid, m_star, label=f"L = {L:.0f} yr")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("asset criticality value v (log scale)")
    ax.set_ylabel("optimal deadline m* (years, log scale)")
    ax.set_title("Comparative statics: m* decreases in v and in L (k = 1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig10_comparative_statics.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4))
    t = np.linspace(0.01, 35, 300)
    for scen, style in (("early", "-"), ("central", "-"), ("late", "-")):
        d = make_dist(scen)
        ax.plot(t, d.cdf(t), linestyle=style, label=f"{scen}: {SCENARIOS[scen]['note']}")
        q10 = d.ppf(0.10)
        ax.axvline(q10, linestyle=":", alpha=0.5)
        ax.plot([q10], [0.10], "ko", markersize=4)
    ax.set_xlabel("years from 2026")
    ax.set_ylabel("CDF of CRQC arrival time")
    ax.set_title("CRQC arrival scenarios; dots mark the 10% deadline quantile")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_arrival_scenarios.png", bbox_inches="tight")
    plt.close(fig)

    summary = sectors_df.groupby("sector").agg(
        urgent_share=("urgent", "mean"), high_share=("high", "mean"),
        infeasible_share=("infeasible_epsilon", "mean"), median_MUI=("MUI", "median"),
        median_m_star=("m_star", "median"), median_tau=("tau", "median"),
    ).sort_values("median_MUI", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(summary))
    ax.bar(x - 0.18, 100 * summary["urgent_share"], 0.36, label="urgent (MUI >= 1)", color="#c44e52")
    ax.bar(x + 0.18, 100 * summary["infeasible_share"], 0.36, label="epsilon-infeasible (tau > F^-1(eps))",
           color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(summary.index, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("share of assets (%)")
    ax.set_title("Sector migration urgency under the central CRQC scenario (eps = 0.10)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig6_sector_urgency.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    pol = policies_df[~policies_df.policy.str.contains("optimal")]
    names = list(pol.policy)
    means = list(pol.mean_cost)
    ax.bar(names, means, color=["#4c72b0", "#55a868", "#8172b3"])
    opt_mean = float(policies_df[policies_df.policy.str.contains("optimal")]["mean_cost"].iloc[0])
    ax.axhline(opt_mean, color="#c44e52", linestyle="--", label=f"model-optimal mean = {opt_mean:.2f}")
    ax.set_yscale("log")
    ax.set_ylabel("mean portfolio cost (log scale)")
    ax.set_title("Policy value: model-guided deadline vs naive schedules")
    ax.tick_params(axis="x", labelsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig8_policy_value.png", bbox_inches="tight")
    plt.close(fig)
    return summary.reset_index()


def main():
    np.random.seed(SEED)
    validation_rows = []

    exp_df = theorem1_validation()
    exp_df.to_csv(RESULTS / "theorem1_exposure.csv", index=False)
    validation_rows.append({
        "check": "T1 closed-form vs quadrature",
        "value": float(exp_df["quadrature_rel_err"].max()),
        "tolerance": 1e-8, "status": "PASS" if exp_df["quadrature_rel_err"].max() < 1e-8 else "FAIL",
    })
    validation_rows.append({
        "check": "T1 closed-form vs Monte Carlo (N=200k)",
        "value": float(exp_df["mc_z"].max()),
        "tolerance": 4.0, "status": "PASS" if exp_df["mc_z"].max() < 4.0 else "FAIL",
    })

    dist_c = make_dist("central")
    t2 = theorem2_validation(dist_c)
    t2.to_csv(RESULTS / "theorem2_optimality.csv", index=False)
    validation_rows.append({
        "check": "T2 bracket signs (g(0+)<0<g(inf))",
        "value": float(np.mean((t2["g_at_lo"] < 0) & (t2["g_at_hi"] > 0))),
        "tolerance": 1.0, "status": "PASS" if np.all((t2["g_at_lo"] < 0) & (t2["g_at_hi"] > 0)) else "FAIL",
    })
    validation_rows.append({
        "check": "T2 grid-argmin relative deviation (max)",
        "value": float(t2["grid_rel_dev"].max()),
        "tolerance": 0.05, "status": "PASS" if t2["grid_rel_dev"].max() < 0.05 else "FAIL",
    })
    validation_rows.append({
        "check": "T2 cost gap at m* vs grid min (max rel)",
        "value": float(t2["cost_rel_dev"].max()),
        "tolerance": 0.01, "status": "PASS" if t2["cost_rel_dev"].max() < 0.01 else "FAIL",
    })

    t3 = theorem3_validation(dist_c)
    t3.to_csv(RESULTS / "theorem3_comparative_statics.csv", index=False)
    sign_cols = [c for c in t3.columns if c.startswith("monotone")]
    all_true = all(bool(t3[c].all()) for c in sign_cols)
    validation_rows.append({
        "check": "T3 monotonicity sign agreement (fraction)",
        "value": float(np.mean([t3[c].mean() for c in sign_cols])),
        "tolerance": 1.0, "status": "PASS" if all_true else "FAIL",
    })

    t4 = theorem4_validation()
    t4.to_csv(RESULTS / "theorem4_feasibility.csv", index=False)
    validation_rows.append({
        "check": "T4 Monte Carlo feasibility coverage (all rows)",
        "value": float(t4["mc_ok"].mean()),
        "tolerance": 1.0, "status": "PASS" if t4["mc_ok"].all() else "FAIL",
    })

    portfolio = draw_sector_portfolio()
    portfolio_c = evaluate_sector_portfolio(portfolio, dist_c)
    portfolio_c.to_csv(RESULTS / "sector_portfolio.csv", index=False)
    policies_df, costs = policy_value(portfolio_c, dist_c)
    policies_df.to_csv(RESULTS / "policy_value.csv", index=False)

    sector_summary = make_figures(exp_df, t2, portfolio_c, policies_df, costs, dist_c)
    sector_summary.to_csv(RESULTS / "sector_priorities.csv", index=False)

    grover = grover_table()
    grover.to_csv(RESULTS / "symmetric_quantum_bits.csv", index=False)

    scenario_sectors, scenario_policies = [], []
    for scen in ("early", "central", "late"):
        d_scen = make_dist(scen)
        # common random numbers: identical assets, only F changes
        port_scen = evaluate_sector_portfolio(portfolio, d_scen)
        agg = port_scen.groupby("sector").agg(
            urgent_share=("urgent", "mean"), infeasible_share=("infeasible_epsilon", "mean"),
            median_MUI=("MUI", "median"), median_m_star=("m_star", "median"),
        ).reset_index()
        agg["scenario"] = scen
        scenario_sectors.append(agg)
        pv_scen, _ = policy_value(port_scen, d_scen)
        pv_scen["scenario"] = scen
        scenario_policies.append(pv_scen)
    pd.concat(scenario_sectors, ignore_index=True).to_csv(
        RESULTS / "sector_priorities_all_scenarios.csv", index=False)
    pd.concat(scenario_policies, ignore_index=True).to_csv(
        RESULTS / "policy_value_all_scenarios.csv", index=False)

    val_df = pd.DataFrame(validation_rows)
    val_df.to_csv(RESULTS / "model_validation.csv", index=False)

    latest = t4[(t4.epsilon == 0.10) & (t4.tau.isin([2.0, 5.0]))].copy()
    latest.to_csv(RESULTS / "latest_start.csv", index=False)

    summary = {
        "seed": SEED,
        "scenarios": SCENARIOS,
        "validations": validation_rows,
        "sector_summary": sector_summary.to_dict("records"),
        "sector_summary_by_scenario": pd.concat(scenario_sectors, ignore_index=True).to_dict("records"),
        "policy_value": policies_df.to_dict("records"),
        "policy_value_by_scenario": pd.concat(scenario_policies, ignore_index=True).to_dict("records"),
        "latest_start_eps010": latest.to_dict("records"),
        "grover": grover.to_dict("records"),
        "theorem2_headline": {
            "n_assets": int(len(t2)),
            "m_star_median_years": float(t2["m_star"].median()),
            "m_star_range": [float(t2["m_star"].min()), float(t2["m_star"].max())],
        },
    }
    (RESULTS / "summary_exp2.json").write_text(json.dumps(summary, indent=2, default=float))
    print(val_df.to_string(index=False), flush=True)
    print("exp2 complete", flush=True)


if __name__ == "__main__":
    main()
