#!/usr/bin/env python3
"""
make_figs.py — generates high-fidelity, professional visualizations for the
Data Center Efficiency PUE Loophole Audit.

Visualizations:
  1. PSU Efficiency Curves (Gold vs. Titanium vs. Ruby) - highlighting verification flags.
  2. PUE Blind Spot (Nominal PUE flatline vs. True Facility Power reduction).
  3. Annual TCO Savings Waterfall/Breakdown (PSU Loss Reduction + Cooling Reduction).
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
RESULTS_FILE = Path("results/audit_metrics.json")
if not RESULTS_FILE.exists():
    print(f"[!] Error: {RESULTS_FILE} not found. Run pue_loophole_model.py first.")
    exit(1)

with open(RESULTS_FILE, "r") as f:
    data = json.load(f)

# If sensitivity run is detected, focus on d = 0.80 for dashboard presentation
if "d_0.8" in data:
    bsp_data = data["d_0.8"]
    all_runs = data
else:
    bsp_data = data
    all_runs = {"d_0.8": data}

assumptions = bsp_data["assumptions"]
cooling_cop_range = bsp_data["cooling_cop_range"]
cop_mean = np.mean(cooling_cop_range)

# Setup modern style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['text.color'] = '#2d3748'
plt.rcParams['axes.labelcolor'] = '#2d3748'
plt.rcParams['xtick.color'] = '#4a5568'
plt.rcParams['ytick.color'] = '#4a5568'
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

# Vibrant and professional HSL-based palette
COLORS = {
    "gold": "#dd6b20",      # Amber/Orange
    "titanium": "#3182ce",  # Professional Blue
    "ruby": "#e53e3e",      # Ruby Red
    "cooling": "#319795",   # Teal
    "facility": "#4a5568"   # Slate Gray
}

# Create a 3-panel dashboard
fig = plt.figure(figsize=(15, 10), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

from pue_loophole_model import THRESHOLDS

# --------------------------------------------------------------------------- #
# Panel 1: PSU Efficiency Curves (Certification Thresholds)
# --------------------------------------------------------------------------- #
ax1 = fig.add_subplot(gs[0, 0])
load_grid = np.linspace(0.05, 1.0, 200)

points = np.array([0.05, 0.10, 0.20, 0.50, 1.00])
curves = {}
for tier, vals in THRESHOLDS.items():
    copied = vals.copy()
    known = np.where(~np.isnan(copied))[0]
    first = known[0]
    copied[:first] = copied[first]
    curves[tier] = copied

for tier, eff_points in curves.items():
    lbl = f"{tier.upper()} ({'UNVERIFIED' if tier == 'ruby' else '80 PLUS Spec'})"
    style = '--' if tier == 'ruby' else '-'
    ax1.plot(load_grid * 100, np.interp(load_grid, points, eff_points) * 100, 
             style, label=lbl, color=COLORS[tier], linewidth=2.5)
    # Scatter verified test points (filter out NaNs for visualization)
    raw_vals = THRESHOLDS[tier]
    valid = ~np.isnan(raw_vals)
    ax1.scatter(points[valid] * 100, raw_vals[valid] * 100, color=COLORS[tier], s=50, zorder=5)

ax1.set_title("A. 80 PLUS Certification Thresholds (230V)", weight='bold', pad=10)
ax1.set_xlabel("PSU Load Fraction (%)")
ax1.set_ylabel("Efficiency (%)")
ax1.set_xlim(0, 100)
ax1.set_ylim(80, 100)
ax1.legend(frameon=True, facecolor='white', edgecolor='#e2e8f0', loc='lower right')


# Highlight crucial low-load points
ax1.axvline(5, color='#cbd5e0', linestyle=':', zorder=1)
ax1.axvline(10, color='#cbd5e0', linestyle=':', zorder=1)
ax1.text(5.5, 82, "5% Load\n(Ruby Spec)", fontsize=9, color='#718096')
ax1.text(10.5, 82, "10% Load\n(Titanium Spec)", fontsize=9, color='#718096')

# --------------------------------------------------------------------------- #
# Panel 2: PUE Blind Spot (Nominal PUE flatline vs. Facility Power)
# --------------------------------------------------------------------------- #
ax2 = fig.add_subplot(gs[0, 1])

tiers = ["gold", "titanium", "ruby"]
p_in = [bsp_data["tiers"][t]["p_in_kw"] for t in tiers]
p_loss = [bsp_data["tiers"][t]["p_loss_kw"] for t in tiers]
cool_mean = [np.mean(bsp_data["tiers"][t]["cooling_kw_range"]) for t in tiers]
pue_mean = [np.mean(bsp_data["tiers"][t]["pue_nominal_range"]) for t in tiers]

x = np.arange(len(tiers))
width = 0.35

# Plot Facility Power breakdown on primary y-axis
rects1 = ax2.bar(x - width/2, p_in, width, label="IT input Power (P_in)", color='#cbd5e0')
rects2 = ax2.bar(x - width/2, cool_mean, width, bottom=p_in, label="PSU + IT Cooling", color=COLORS["cooling"])

ax2.set_ylabel("Power Consumption (kW)", color='#2d3748')
ax2.set_title("B. The PUE Blind Spot (100 kW IT Load, d = 0.80)", weight='bold', pad=10)
ax2.set_xticks(x)
ax2.set_xticklabels([t.upper() for t in tiers])
ax2.set_ylim(0, 150)
ax2.tick_params(axis='y', labelcolor='#2d3748')
ax2.legend(loc='upper left', frameon=True, facecolor='white')

# Plot Nominal PUE on secondary y-axis
ax2_sec = ax2.twinx()
ax2_sec.plot(x, pue_mean, color=COLORS["ruby"], marker='o', linewidth=3, markersize=8, label="Nominal PUE (AC)")
ax2_sec.set_ylabel("Nominal PUE Value", color=COLORS["ruby"])
ax2_sec.tick_params(axis='y', labelcolor=COLORS["ruby"])
ax2_sec.set_ylim(1.0, 1.3)
ax2_sec.grid(False) # avoid overlapping gridlines
ax2_sec.legend(loc='upper right', frameon=True, facecolor='white')

# Add annotations to highlight the flat PUE value
for i, val in enumerate(pue_mean):
    ax2_sec.annotate(f"PUE = {val:.4f}", (x[i], val), textcoords="offset points", 
                     xytext=(0,10), ha='center', weight='bold', color=COLORS["ruby"])

# --------------------------------------------------------------------------- #
# Panel 3: Annual Energy Savings Sensitivity (MWh/year vs. d)
# --------------------------------------------------------------------------- #
ax3 = fig.add_subplot(gs[1, :])

d_vals = []
titanium_savings = []
ruby_savings = []

# Sort runs by duty cycle
for k in sorted(all_runs.keys(), key=lambda x: float(x.split("_")[1])):
    d = float(k.split("_")[1])
    d_vals.append(d * 100) # percentage
    
    run_metrics = all_runs[k]["savings_vs_gold"]
    # Mean savings
    titanium_savings.append(np.mean(run_metrics["titanium"]["annual_energy_saving_mwh_range"]))
    ruby_savings.append(np.mean(run_metrics["ruby"]["annual_energy_saving_mwh_range"]))

ax3.plot(d_vals, titanium_savings, marker='s', color=COLORS["titanium"], linewidth=3, markersize=8,
         label="Gold -> Titanium Savings")
ax3.plot(d_vals, ruby_savings, '--', marker='^', color=COLORS["ruby"], linewidth=3, markersize=8,
         label="Gold -> Ruby Savings (UNVERIFIED)")

ax3.set_title("C. Annual Energy Savings Sensitivity vs. GPU Compute Duty Cycle (d)", weight='bold', pad=10)
ax3.set_xlabel("GPU Compute Phase Duty Cycle (d, %)")
ax3.set_ylabel("Annual Facility-Level Energy Savings (MWh/year)")
ax3.set_xticks(d_vals)
ax3.set_xlim(50, 100)
ax3.legend(frameon=True, facecolor='white', edgecolor='#e2e8f0', loc='upper left')

# Add ranges as shaded areas if multiple COPs exist
for k in sorted(all_runs.keys(), key=lambda x: float(x.split("_")[1])):
    d = float(k.split("_")[1]) * 100
    t_range = all_runs[k]["savings_vs_gold"]["titanium"]["annual_energy_saving_mwh_range"]
    r_range = all_runs[k]["savings_vs_gold"]["ruby"]["annual_energy_saving_mwh_range"]
    ax3.fill_between([d], [t_range[0]], [t_range[1]], color=COLORS["titanium"], alpha=0.2)
    ax3.fill_between([d], [r_range[0]], [r_range[1]], color=COLORS["ruby"], alpha=0.1)

# Set figure title
fig.suptitle("VolMax Audit Dashboard: Data Center PSU Efficiency & PUE Loophole Analysis", 
             weight='bold', y=0.98)

# Save figure
Path("results").mkdir(exist_ok=True)
fig_path = Path("results/pue_audit_dashboard.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"[OK] Generated dashboard and saved to {fig_path}")
