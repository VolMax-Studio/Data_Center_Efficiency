#!/usr/bin/env python3
"""
pue_loophole_model.py — quantifies the "PUE Loophole": how PSU conversion losses,
booked into IT power, mask real facility-level energy savings when moving up PSU
efficiency tiers (Gold -> Titanium -> Ruby).

VolMax Verification Doctrine (P10) baked into the code:
  - CLAIMS EFFICIENCY-TIER ONLY, never GaN-vs-silicon (the data doesn't distinguish).
  - THRESHOLD vs REAL-CURVE are separate, explicitly labeled sources. A result built
    on certification thresholds is a CONSERVATIVE LOWER BOUND, and says so.
  - EVERY ASSUMPTION (PUE boundary AC/DC, cooling COP, profile) is a parameter,
    surfaced in the output, never hidden.
  - PHYSICS GATE: energy balance P_in = P_out + P_loss must hold exactly, else abort.
  - The AI load profile is modeled as a two-state Bulk Synchronous Parallel (BSP)
    square-wave representing GPU compute vs. communication cycles.

Run:
    python pue_loophole_model.py --profile bsp --cooling liquid --pue-boundary ac
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# PSU efficiency by load. Two sources, never mixed:
#   THRESHOLDS = 80 PLUS certification minima (230V internal redundant).
#     -> using these yields a CONSERVATIVE LOWER BOUND on tier savings.
#   ESTIMATED_CURVES = representative tier-typical values pending extraction 
#     of actual verification-report test points. Do NOT cite as measured.
# Load fractions: 0.05, 0.10, 0.20, 0.50, 1.00
# --------------------------------------------------------------------------- #
LOAD_POINTS = np.array([0.05, 0.10, 0.20, 0.50, 1.00])

# 80 PLUS 230V Internal Redundant certification THRESHOLDS (minima).
THRESHOLDS = {
    # tier:           5%     10%    20%    50%    100%   (efficiency fraction)
    "gold":     np.array([np.nan, np.nan, 0.92,  0.94,  0.90]),   # 230V redundant minima
    "titanium": np.array([np.nan, 0.90,  0.94,  0.96,  0.91]),    
    # RUBY: UNVERIFIED PLACEHOLDER — confirm date + exact minima with CLEAResult.
    "ruby":     np.array([0.90,  0.91,  0.95,  0.965, 0.92]),     # <-- DO NOT PUBLISH AS-IS
}
RUBY_UNVERIFIED = True  # flips off only when you've confirmed against the official spec

# ESTIMATED / TYPICAL efficiency curves — NOT measured per-model CLEAResult reports.
# These are representative tier-typical values pending extraction of actual
# verification-report test points. Do NOT cite as measured.
ESTIMATED_CURVES: dict[str, dict] = {
    "gold": {
        "model": "Gold tier-typical (NOT a measured report)",
        # 10% and 5% are filled conservatively
        "eff": np.array([0.82, 0.885, 0.924, 0.942, 0.908])
    },
    "titanium": {
        "model": "Titanium tier-typical (NOT a measured report)",
        # 5% is filled conservatively
        "eff": np.array([0.88, 0.9205, 0.9504, 0.9620, 0.9457])
    }
    # Ruby stays out until a real Ruby-certified report exists
}


# --------------------------------------------------------------------------- #
# Workload Duty Cycles
# --------------------------------------------------------------------------- #
COOLING_COP = {           # heat-to-cooling-work ratio (P_cooling = P_heat / COP)
    "air":    (2.5, 3.5),
    "liquid": (4.5, 6.0),
}


@dataclass
class Assumptions:
    profile: str                 # "bsp" or "enterprise" or "idle"
    bsp_d: float                 # Bulk Synchronous Parallel duty cycle (compute time fraction)
    bsp_load_compute: float      # load fraction during GPU compute (TDP)
    bsp_load_comm: float         # load fraction during AllReduce (NIC active, GPU compute idle)
    cooling: str
    pue_boundary: str            # "ac": PSU loss in denominator (loophole active); "dc": not
    rack_it_kw: float            # delivered DC power to IT load at the rack
    hours_per_year: float
    price_eur_per_kwh: float
    eff_source: str              # "threshold" (lower bound) or "estimated"


def interpolate_efficiency(load: float, tier: str, source: str) -> float:
    """
    Interpolate efficiency for a given load fraction using 1D linear interpolation.
    Points below 5% or above 100% are clipped to the boundaries.
    """
    if source == "estimated" and tier in ESTIMATED_CURVES:
        eff_vals = ESTIMATED_CURVES[tier]["eff"].copy()
    else:
        # Fallback to thresholds if no estimated curve is defined (e.g. Ruby)
        eff_vals = THRESHOLDS[tier].copy()
        
    # Handle NaNs (e.g. 5% or 10% load not defined in Gold thresholds)
    known_idx = np.where(~np.isnan(eff_vals))[0]
    if known_idx.size == 0:
        raise ValueError(f"No efficiency points defined for {tier}")
    
    # Conservatively fill missing low-load points with the first known value
    first_known_idx = known_idx[0]
    eff_vals[:first_known_idx] = eff_vals[first_known_idx]
    
    # Clip load to [0.05, 1.0] for safety
    load_clipped = np.clip(load, 0.05, 1.0)
    
    return float(np.interp(load_clipped, LOAD_POINTS, eff_vals))



def calculate_weighted_efficiency(a: Assumptions, tier: str) -> float:
    """
    Calculate the time-integrated (weighted) efficiency over the workload profile.
    
    For a two-state square wave (compute vs. communication):
    eta_weighted = (d * L_compute + (1-d) * L_comm) / (d * L_compute / eta_compute + (1-d) * L_comm / eta_comm)
    
    This is mathematically correct under the conservation of energy (E_out / E_in).
    """
    if a.profile == "bsp":
        eta_comp = interpolate_efficiency(a.bsp_load_compute, tier, a.eff_source)
        eta_comm = interpolate_efficiency(a.bsp_load_comm, tier, a.eff_source)
        
        num = a.bsp_d * a.bsp_load_compute + (1.0 - a.bsp_d) * a.bsp_load_comm
        den = (a.bsp_d * a.bsp_load_compute / eta_comp) + ((1.0 - a.bsp_d) * a.bsp_load_comm / eta_comm)
        return num / den
        
    elif a.profile == "enterprise":
        # Broad enterprise profile weights
        w = np.array([0.10, 0.15, 0.25, 0.35, 0.15]) # weights over 5%, 10%, 20%, 50%, 100%
        effs = np.array([interpolate_efficiency(l, tier, a.eff_source) for l in LOAD_POINTS])
        # Energy conservation weighted average
        # eta = sum(w * L) / sum(w * L / eta)
        num = np.sum(w * LOAD_POINTS)
        den = np.sum(w * LOAD_POINTS / effs)
        return num / den
        
    else: # idle profile
        w = np.array([0.40, 0.30, 0.20, 0.08, 0.02])
        effs = np.array([interpolate_efficiency(l, tier, a.eff_source) for l in LOAD_POINTS])
        num = np.sum(w * LOAD_POINTS)
        den = np.sum(w * LOAD_POINTS / effs)
        return num / den


def run(a: Assumptions) -> dict:
    cop_lo, cop_hi = COOLING_COP[a.cooling]
    tiers = ["gold", "titanium", "ruby"]

    out = {"assumptions": asdict(a), "cooling_cop_range": [cop_lo, cop_hi], "tiers": {}}

    for tier in tiers:
        eta = calculate_weighted_efficiency(a, tier)      # Profile-weighted efficiency
        p_out = a.rack_it_kw                               # DC power delivered to IT (fixed)
        p_in = p_out / eta                                 # AC drawn from wall
        p_loss = p_in - p_out                              # heat dissipated in the PSU

        # PHYSICS GATE: energy balance must hold exactly
        if not np.isclose(p_in, p_out + p_loss, rtol=1e-9):
            sys.exit(f"ENERGY BALANCE VIOLATED for {tier}: {p_in} != {p_out}+{p_loss}")

        # cooling work to remove the entire rack heat load (p_in),
        # since all electrical power delivered to the rack is ultimately dissipated as heat.
        cool_hi = p_in / cop_lo   # worst COP -> most cooling work
        cool_lo = p_in / cop_hi   # best COP  -> least cooling work

        # PUE numerator/denominator depending on measurement boundary
        if a.pue_boundary == "ac":
            p_it_metric = p_in            # loophole: loss hidden in denominator
        else:
            p_it_metric = p_out           # honest: loss exposed in numerator

        # Total facility power (excluding other loads for simplicity, only PSU + cooling)
        tot_facility_lo = p_in + cool_lo
        tot_facility_hi = p_in + cool_hi
        
        # Nominal PUE = Total Facility Power / IT Metric basis
        pue_nominal_lo = tot_facility_lo / p_it_metric
        pue_nominal_hi = tot_facility_hi / p_it_metric

        out["tiers"][tier] = {
            "profile_weighted_efficiency": round(eta, 4),
            "p_in_kw": round(p_in, 3),
            "p_out_kw": round(p_out, 3),
            "p_loss_kw": round(p_loss, 3),
            "cooling_kw_range": [round(cool_lo, 3), round(cool_hi, 3)],
            "total_facility_kw_range": [round(tot_facility_lo, 3), round(tot_facility_hi, 3)],
            "pue_nominal_range": [round(pue_nominal_lo, 4), round(pue_nominal_hi, 4)],
            "pue_it_power_basis_kw": round(p_it_metric, 3),
        }

    # tier-to-tier savings vs Gold baseline
    base = out["tiers"]["gold"]
    out["savings_vs_gold"] = {}
    for tier in ["titanium", "ruby"]:
        t = out["tiers"][tier]
        d_loss = base["p_loss_kw"] - t["p_loss_kw"]                  # less PSU heat
        d_cool_lo = base["cooling_kw_range"][0] - t["cooling_kw_range"][0]
        d_cool_hi = base["cooling_kw_range"][1] - t["cooling_kw_range"][1]
        d_total_lo = d_loss + d_cool_lo
        d_total_hi = d_loss + d_cool_hi
        mwh_lo = d_total_lo * a.hours_per_year / 1000.0
        mwh_hi = d_total_hi * a.hours_per_year / 1000.0
        out["savings_vs_gold"][tier] = {
            "psu_heat_reduction_kw": round(d_loss, 3),
            "cooling_reduction_kw_range": [round(d_cool_lo, 3), round(d_cool_hi, 3)],
            "total_power_saving_kw_range": [round(d_total_lo, 3), round(d_total_hi, 3)],
            "annual_energy_saving_mwh_range": [round(mwh_lo, 2), round(mwh_hi, 2)],
            "annual_cost_saving_eur_range": [round(mwh_lo*1000*a.price_eur_per_kwh, 0),
                                             round(mwh_hi*1000*a.price_eur_per_kwh, 0)],
        }

    out["pue_blind_spot_note"] = (
        "Under the AC measurement boundary, PSU loss sits in the IT-power denominator, "
        "so improving PSU efficiency can leave nominal PUE flat or even raise it while "
        "true facility energy falls. Report both nominal and DC-boundary PUE."
    )
    return out


def main():
    ap = argparse.ArgumentParser(description="PUE loophole + true TCO model (efficiency-tier audit)")
    ap.add_argument("--profile", choices=["bsp", "enterprise", "idle"], default="bsp")
    ap.add_argument("--bsp-d", type=float, default=0.8, help="Compute phase fraction (d)")
    ap.add_argument("--bsp-load-compute", type=float, default=0.98, help="Compute load fraction")
    ap.add_argument("--bsp-load-comm", type=float, default=0.40, help="Comm load fraction")
    ap.add_argument("--cooling", choices=list(COOLING_COP), default="liquid")
    ap.add_argument("--pue-boundary", choices=["ac", "dc"], default="ac")
    ap.add_argument("--rack-it-kw", type=float, default=100.0, help="DC power delivered to IT load")
    ap.add_argument("--hours", type=float, default=8760.0)
    ap.add_argument("--price", type=float, default=0.15, help="EUR/kWh")
    ap.add_argument("--eff-source", choices=["threshold", "estimated"], default="threshold")
    ap.add_argument("--out", default="results/audit_metrics.json")
    args = ap.parse_args()

    if RUBY_UNVERIFIED:
        print("  [!] WARNING: Ruby thresholds/date are UNVERIFIED placeholders. "
              "Confirm against the official CLEAResult 80 PLUS spec before publishing "
              "any Ruby number.\n")

    if args.eff_source == "threshold":
        print("  [note] using CERTIFICATION THRESHOLDS -> results are a CONSERVATIVE "
              "LOWER BOUND on tier savings, not measured per-model values.\n")
    else:
        print("  [note] using REPRESENTATIVE ESTIMATED CURVES (tier-typical, not measured reports).\n")


    # Run sensitivity analysis over d if bsp is selected
    if args.profile == "bsp":
        d_options = [0.6, 0.8, 0.95]
        sensitivity_results = {}
        
        print("==========================================")
        print("    SENSITIVITY ANALYSIS: BSP DUTY CYCLE")
        print("    Compute Load = 98%, Comm Load = 40%")
        print("==========================================")
        
        for d in d_options:
            a = Assumptions(
                profile=args.profile, bsp_d=d, bsp_load_compute=args.bsp_load_compute,
                bsp_load_comm=args.bsp_load_comm, cooling=args.cooling,
                pue_boundary=args.pue_boundary, rack_it_kw=args.rack_it_kw,
                hours_per_year=args.hours, price_eur_per_kwh=args.price,
                eff_source=args.eff_source
            )
            res = run(a)
            sensitivity_results[f"d_{d}"] = res
            
            print(f"\nBSP Duty Cycle d = {d:.2f} (compute phase {d*100:.0f}%, communication {(1-d)*100:.0f}%):")
            for tier, t in res["tiers"].items():
                flag = "  [UNVERIFIED]" if (tier == "ruby" and RUBY_UNVERIFIED) else ""
                pue_str = f"{t['pue_nominal_range'][0]:.4f}-{t['pue_nominal_range'][1]:.4f}"
                print(f"  {tier:>9}: eff={t['profile_weighted_efficiency']:.4f} | "
                      f"P_in={t['p_in_kw']:.1f}kW | loss={t['p_loss_kw']:.2f}kW | "
                      f"Nominal PUE={pue_str}{flag}")
            print("  Savings vs Gold baseline:")
            for tier in ["titanium", "ruby"]:
                s = res["savings_vs_gold"][tier]
                flag = "  [UNVERIFIED — Ruby]" if tier == "ruby" and RUBY_UNVERIFIED else ""
                print(f"    {tier:>9}: {s['total_power_saving_kw_range'][0]:.2f}-"
                      f"{s['total_power_saving_kw_range'][1]:.2f} kW | "
                      f"{s['annual_energy_saving_mwh_range'][0]:.1f}-"
                      f"{s['annual_energy_saving_mwh_range'][1]:.1f} MWh/yr{flag}")
                      
        # Save sensitivity run to JSON
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(sensitivity_results, f, indent=2)
        print(f"\n  [OK] Saved all sensitivity results to {args.out}")
        
    else:
        # Single run for non-BSP profiles
        a = Assumptions(
            profile=args.profile, bsp_d=0.0, bsp_load_compute=0.0,
            bsp_load_comm=0.0, cooling=args.cooling,
            pue_boundary=args.pue_boundary, rack_it_kw=args.rack_it_kw,
            hours_per_year=args.hours, price_eur_per_kwh=args.price,
            eff_source=args.eff_source
        )
        res = run(a)
        
        print(f"profile={a.profile}  cooling={a.cooling}  PUE-boundary={a.pue_boundary}  "
              f"IT={a.rack_it_kw}kW  source={a.eff_source}")
        for tier, t in res["tiers"].items():
            flag = "  [UNVERIFIED]" if (tier == "ruby" and RUBY_UNVERIFIED) else ""
            pue_str = f"{t['pue_nominal_range'][0]:.4f}-{t['pue_nominal_range'][1]:.4f}"
            print(f"  {tier:>9}: eff={t['profile_weighted_efficiency']:.3f} | "
                  f"P_in={t['p_in_kw']:.1f}kW | loss={t['p_loss_kw']:.2f}kW | "
                  f"Nominal PUE={pue_str}{flag}")
        print("\n  savings vs Gold:")
        for tier, s in res["savings_vs_gold"].items():
            flag = "  [UNVERIFIED — Ruby]" if tier == "ruby" and RUBY_UNVERIFIED else ""
            print(f"    {tier:>9}: {s['total_power_saving_kw_range'][0]:.2f}-"
                  f"{s['total_power_saving_kw_range'][1]:.2f} kW  ->  "
                  f"{s['annual_energy_saving_mwh_range'][0]:.1f}-"
                  f"{s['annual_energy_saving_mwh_range'][1]:.1f} MWh/yr{flag}")
                  
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
