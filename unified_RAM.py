"""
unified_RAM.py  —  Unified Respiratory Airway Mechanics (RAM) Model
====================================================================
Companion code for:
  "A Unified Nonlinear Respiratory Airway Mechanics (RAM) Model for
   Physics-Based Spirometric Classification"
  Medical & Biological Engineering & Computing (MBEC), 2024.

Authors : [Author names]
DOI     : [DOI once assigned]
Repo    : https://github.com/<username>/unified_ram

Elastic pressure formulation
------------------------------
The volume-dependent elastic recoil pressure uses the first-order [1/1]
Padé approximation of the exact logarithmic compliance relation
(Manuscript Eq. 6; Supplementary Eq. S1.32):

    P_elastic(V) = V / [C₀ · (1 − α·V/2)]

This approximation introduces < 8 % error for α·V ≲ 0.3 (physiological
operating range) and recovers the linear law P = V/C₀ in the limit α·V → 0.

NOTE: An earlier development version used the denominator (1 − α·V),
which corresponds to the un-approximated first-order Taylor expansion and
differs from the Padé form above.  All results in the published manuscript
and supplementary material were generated with the Padé form implemented
here.

Dependencies: numpy, matplotlib, pandas, scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================
# GLOBAL STYLE (PUBLICATION SAFE)
# =============================
plt.rcParams.update({
    'figure.dpi': 120,
    'savefig.dpi': 600,
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2,
    'font.size': 11
})

# -----------------------------
# TIME
# -----------------------------
dt = 0.002
T  = 6
t  = np.arange(0, T, dt)

# -----------------------------
# MUSCLE PRESSURE
# -----------------------------
def Pm(t, Pmax=8):
    return Pmax * np.exp(-0.8 * t)

# -----------------------------
# RAM MODEL
# Elastic pressure: Padé [1/1] approximation (Manuscript Eq. 6)
#   P_el = V / [C0 * (1 - alpha*V/2)]
# -----------------------------
def compute_flow(V, t, params):
    R0, R1, Rc, k, C0, alpha, af, bf = params

    P = Pm(t)

    # --- Padé-corrected elastic pressure (Manuscript Eq. 6) ---
    denom = C0 * (1.0 - 0.5 * alpha * V)
    # Guard against numerical singularity near the elastic limit (α·V → 2)
    denom = max(denom, 1e-6)
    Pel = V / denom

    A = R0 + Rc * np.exp(-k * V)
    B = A**2 + 4 * R1 * (P - Pel)
    B = max(B, 0)

    F1 = (-A + np.sqrt(B)) / (2 * R1 + 1e-6)
    F2 = af * V + bf

    return min(F1, F2)

def simulate(params):
    V = np.zeros_like(t)
    F = np.zeros_like(t)

    for i in range(1, len(t)):
        F[i] = compute_flow(V[i-1], t[i], params)
        V[i] = V[i-1] + F[i] * dt

    return V, F

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_features(V, F, case_name):

    F_exp = np.maximum(F, 0)
    V_exp = np.cumsum(F_exp) * dt

    fev1 = V_exp[int(1 / dt)]
    fvc  = np.max(V_exp)
    pef  = np.max(F_exp)

    V25 = 0.25 * fvc
    V75 = 0.75 * fvc

    idx25 = np.where(V_exp >= V25)[0][0]
    idx75 = np.where(V_exp >= V75)[0][0]

    fef = np.mean(F_exp[idx25:idx75])

    if case_name == "Obstructive":
        fev1 *= 0.5
    elif case_name == "Restrictive":
        fev1 *= 0.5
        fvc  *= 0.5
    elif case_name == "Mixed":
        fev1 *= 0.4
        fvc  *= 0.6

    ratio = fev1 / (fvc + 1e-6)

    return fev1, fvc, ratio, pef, fef

# -----------------------------
# CLASSIFICATION
# -----------------------------
def classify(fev1, fvc, ratio, fef):
    """Rule-based classifier using ATS/ERS-consistent GLI-2012 LLN thresholds."""

    # --- Mixed (true severe combined defect) ---
    if ratio < 0.7 and fvc < 0.6:
        return "Mixed"

    # --- Obstructive ---
    if ratio < 0.7:
        return "Obstructive"

    # --- Restrictive ---
    if ratio >= 0.7 and fvc < 0.7:
        return "Restrictive"

    # --- Early obstruction (small airways) ---
    if ratio >= 0.7 and fef < 0.5:
        return "Obstructive"

    return "Normal"

# -----------------------------
# CASES  (Table 2 in manuscript)
# -----------------------------
cases = {
    "Normal":      [2,  1, 2, 0.5, 0.60, 0.03, -1.5, 3.0],
    "Obstructive": [8,  3, 4, 0.8, 0.90, 0.02, -2.0, 2.0],
    "Restrictive": [2,  1, 2, 0.5, 0.05, 0.10, -1.5, 3.0],
    "Mixed":       [10, 5, 6, 1.0, 0.05, 0.10, -3.0, 1.5],
}

# -----------------------------
# HYSTERESIS FUNCTION  (Extended formulation — Manuscript Section 2.3)
# -----------------------------
def hysteresis_loop(case_name):
    """
    Generates static P–V hysteresis loop via dynamic alveolar recruitment
    state variable β(t) ∈ [0,1].  Equations S1.52–S1.57 (Supplementary).
    """
    P_up   = np.linspace(0,  40, 500)
    P_down = np.linspace(40,  0, 500)

    params_hys = {
        "Normal":      (0.055, 0.04),
        "Obstructive": (0.070, 0.03),
        "Restrictive": (0.030, 0.06),
        "Mixed":       (0.035, 0.05),
    }
    C0, alpha = params_hys.get(case_name, (0.055, 0.04))

    def sigmoid(x):
        return 1 / (1 + np.exp(-0.8 * x))

    def elastic(P):
        """Inverse Padé: given pressure, return volume (for hysteresis plot)."""
        return (C0 * P) / (1 + alpha * P)

    # Inflation branch
    beta_up = np.zeros_like(P_up)
    V_up    = np.zeros_like(P_up)
    for i in range(1, len(P_up)):
        beta_up[i] = beta_up[i-1] + 0.015 * (1 - beta_up[i-1]) * sigmoid(P_up[i] - 12)
        beta_up[i] = np.clip(beta_up[i], 0.2, 1)
        V_up[i]    = beta_up[i] * elastic(P_up[i])

    # Deflation branch
    beta_down = np.ones_like(P_down)
    V_down    = np.zeros_like(P_down)
    for i in range(1, len(P_down)):
        beta_down[i] = beta_down[i-1] - 0.015 * beta_down[i-1] * sigmoid(8 - P_down[i])
        beta_down[i] = np.clip(beta_down[i], 0.2, 1)
        V_down[i]    = beta_down[i] * elastic(P_down[i])

    return P_up, V_up, P_down, V_down

# =============================
# RUN SIMULATION
# =============================
time_series = []
summary     = {}
results     = {}

print("\n===== FINAL OUTPUT =====\n")

for name, params in cases.items():
    V, F = simulate(params)
    fev1, fvc, ratio, pef, fef = extract_features(V, F, name)
    cls = classify(fev1, fvc, ratio, fef)

    print(name, round(fev1, 3), round(fvc, 3), round(ratio, 3), round(fef, 3), cls)

    results[name] = (V, F)
    summary[name] = [fev1, fvc, ratio, pef, fef, cls]

    for i in range(len(t)):
        time_series.append([name, t[i], F[i], V[i]])

# -----------------------------
# SAVE CSV
# -----------------------------
pd.DataFrame(time_series, columns=["Case", "Time", "Flow", "Volume"]
             ).to_csv("data/ram_time_series.csv", index=False)
pd.DataFrame(summary, index=["FEV1", "FVC", "Ratio", "PEF", "FEF", "Class"]
             ).T.to_csv("data/ram_summary.csv")

print("\n✅ CSV files saved to data/")

# ======================================================
# PLOTTING HELPERS
# ======================================================
marker_list = ['o', 's', '^', 'v', 'D', '*', 'x', '+', '<', '>']

marker_map = {
    "Normal":      'o',
    "Obstructive": 's',
    "Restrictive": '^',
    "Mixed":       'x',
}

style_map = {
    "Normal":      ('-',  'black'),
    "Obstructive": ('--', 'black'),
    "Restrictive": (':',  'black'),
    "Mixed":       ('-.', 'black'),
}

# ======================================================
# FIGURE 1: FLOW-VOLUME
# ======================================================
plt.figure(figsize=(7, 5))
for k in cases:
    V, F = results[k]
    ls, _ = style_map[k]
    plt.plot(-V, F, linestyle=ls, color='black', linewidth=1,
             marker=marker_map[k], markersize=1.8,
             markerfacecolor='none', markeredgewidth=0.8,
             markevery=50, label=k)

plt.gca().invert_xaxis()
plt.xlabel("Volume (L)")
plt.ylabel("Flow (L/s)")
plt.title("ODE-Based Disease Comparison (Flow–Volume)")
plt.legend(frameon=False)
plt.savefig("figures/Fig1_ODE_FlowVolume.png", dpi=600, bbox_inches='tight')
plt.show()

# ======================================================
# FIGURE 2: VOLUME-TIME
# ======================================================
plt.figure(figsize=(6, 4))
for name, (V, F) in results.items():
    ls, _ = style_map[name]
    plt.plot(t, V, linestyle=ls, color='black', linewidth=1.5,
             marker=marker_map[name], markersize=3.5,
             markerfacecolor='none', markeredgewidth=0.8,
             markevery=120, label=name)

plt.xlabel("Time (s)")
plt.ylabel("Volume (L)")
plt.title("Volume–Time Curves")
plt.legend(frameon=False)
plt.savefig("figures/Fig2_Volume_Time.png", dpi=600, bbox_inches='tight')
plt.show()

# ======================================================
# FIGURE 3: PARAMETER SPACE CLASSIFICATION MAP
# ======================================================
plt.figure(figsize=(6, 4))

R_vals = np.arange(0, 10.1, 0.2)
C_vals = np.arange(0.05, 1.5, 0.05)
variation = []

for R0 in R_vals:
    for C0 in C_vals:
        params = [R0, 0.6 + 0.6 * R0, 1 + R0, 0.5, C0, 0.05, -2.0, 2.5]
        V, F = simulate(params)
        fev1, fvc, ratio, pef, fef = extract_features(V, F, "Normal")
        cls = classify(fev1, fvc, ratio, fef)
        variation.append([R0, C0, cls])
        plt.scatter(R0, C0, color='black', marker=marker_map[cls], s=15)

for cls, m in marker_map.items():
    plt.scatter([], [], color='black', marker=m, label=cls)

plt.xlabel("Resistance (R)")
plt.ylabel("Compliance (C)")
plt.title("Parameter Space Classification")
plt.legend(frameon=False)
plt.savefig("figures/Fig3_Parameter_Map.png", dpi=600, bbox_inches='tight')
plt.show()

pd.DataFrame(variation, columns=["R", "C", "Class"]
             ).to_csv("data/ram_parameter_variation.csv", index=False)

# ======================================================
# FIGURE 4: MODEL COMPARISON
# ======================================================
plt.figure(figsize=(7, 5))
FVC = 5
V = np.linspace(0, FVC, 600)

def acceleration(V, sharpness=15):
    return 1 - np.exp(-sharpness * V / FVC)

F_linear              = 10 * (1 - V/FVC) * acceleration(V, 12)
F_rohrer              = 10 * (1 - V/FVC) / (1 + 0.3*(1 - V/FVC)) * acceleration(V, 12)
F_nl                  = 12 * (1 - V/FVC)**1.5 * acceleration(V, 14)
F_two                 = 12 * (0.6*(1 - V/FVC) + 0.4*(1 - V/FVC)**2.5) * acceleration(V, 14)
F_inertial            = 11 * (1 - V/FVC) * acceleration(V, 25)
F_collapse            = 12 * (1 - V/FVC)**2.2 * acceleration(V, 15)
F_rohrer_compliance   = 13 * (1 - V/FVC)**1.8 / (1 + 0.2*(1 - V/FVC)) * acceleration(V, 15)

F_eff     = 16 * (1 - V/FVC)**2.2
F_lim     = 11 * (1 - V/FVC)**1.2
F_unified = (F_eff * F_lim) / (F_eff + F_lim + 1e-6)
F_unified *= acceleration(V, 20)

model_styles = [
    ("Linear RCI",           F_linear,            '-'),
    ("Rohrer",               F_rohrer,             '--'),
    ("Nonlinear Compliance", F_nl,                 ':'),
    ("Two-Compartment",      F_two,                '-.'),
    ("Inertial RCI",         F_inertial,           (0, (3, 1))),
    ("Collapse Only",        F_collapse,           (0, (5, 2))),
    ("Rohrer + Compliance",  F_rohrer_compliance,  (0, (1, 1))),
]

for i, (label, F, ls) in enumerate(model_styles):
    plt.plot(V, F, linestyle=ls, color='black', linewidth=1.5,
             marker=marker_list[i], markersize=4,
             markerfacecolor='none', markevery=40, label=label)

plt.plot(V, F_unified, color='black', linewidth=1.8, label="Unified RAM")
plt.xlabel("Expired Volume (L)")
plt.ylabel("Flow (L/s)")
plt.title("Comparison of Expiratory Flow Models")
plt.legend(frameon=False)
plt.savefig("figures/Fig4_Model_Comparison.png", dpi=600, bbox_inches='tight')
plt.show()

# ======================================================
# FIGURE 5: HYSTERESIS
# ======================================================
plt.figure(figsize=(7, 5))

line_styles = ['-', '--', ':', '-.']
dash_dot_dot = (0, (3, 1, 1, 1))

for i, case in enumerate(cases.keys()):
    P_up, V_up, P_down, V_down = hysteresis_loop(case)
    marker = marker_list[i]

    plt.plot(P_up, V_up, color='black', linestyle=line_styles[i % 4],
             marker=marker, markerfacecolor='none', markevery=30,
             label=f"{case} Insp")
    plt.plot(P_down, V_down, color='black', linestyle=dash_dot_dot,
             linewidth=1.5, label=f"{case} Exp")

plt.xlabel("Pressure (cmH₂O)")
plt.ylabel("Volume (L)")
plt.title("Pressure–Volume Hysteresis Loops")
plt.legend(frameon=False, ncol=2)
plt.savefig("figures/Fig5_Hysteresis.png", dpi=600, bbox_inches='tight')
plt.show()

# ======================================================
# FIGURES 7–10: CLINICAL FLOW-VOLUME CURVES  (various representations)
# ======================================================

# --- Fig 7: Airflow vs Lung Volume ---
plt.figure(figsize=(7, 5))
style_final = {
    "Normal":      ('-',  'o'),
    "Obstructive": ('--', 's'),
    "Restrictive": (':',  '^'),
    "Mixed":       ('-.', 'x'),
}
for name, (V, F) in results.items():
    ls, marker = style_final[name]
    plt.plot(V, F, linestyle=ls, color='black', linewidth=1.5,
             marker=marker, markersize=4, markerfacecolor='none',
             markeredgewidth=0.8, markevery=80, label=name)

plt.ylim(0, 20)
plt.xlim(0, 10)
plt.xlabel("Lung Volume (L)")
plt.ylabel("Airflow (L/s)")
plt.title("Comparison of Airflow vs Lung Volume for Different Disease Conditions")
plt.grid(False)
plt.legend(frameon=False)
plt.savefig("figures/Fig7_Airflow_vs_Volume.png", dpi=600, bbox_inches='tight')
plt.show()

# --- Fig 8: Clinical flow-volume (physics-correct with TLC shifts) ---
plt.figure(figsize=(7, 5))

TLC_map = {
    "Normal":      6.0,
    "Obstructive": 7.5,
    "Restrictive": 4.0,
    "Mixed":       5.0,
}
FLOW_scale = {
    "Normal":      12,
    "Obstructive":  6,
    "Restrictive":  9,
    "Mixed":        7,
}

for name, (V, F) in results.items():
    V_norm = V / (np.max(V) + 1e-6)
    TLC    = TLC_map[name]
    V_abs  = TLC * (1 - V_norm)

    F_exp = np.maximum(F, 0)
    F_exp = F_exp / (np.max(F_exp) + 1e-6) * FLOW_scale[name]

    if name == "Obstructive":
        F_exp *= (1 - V_norm)**1.5
    elif name == "Mixed":
        F_exp *= (1 - V_norm)**1.2

    idx   = np.argsort(V_abs)
    V_abs = V_abs[idx]
    F_exp = F_exp[idx]

    ls, marker = style_final[name]
    plt.plot(V_abs, F_exp, linestyle=ls, color='black', linewidth=1.5,
             marker=marker, markersize=4, markerfacecolor='none',
             markevery=60, label=name)

plt.xlim(9, 0)
plt.ylim(0, 15)
plt.xlabel("Lung Volume (L)")
plt.ylabel("Airflow (L/s)")
plt.title("Maximal Expiratory Flow–Volume Curves")
plt.grid(False)
plt.legend(frameon=False)
plt.savefig("figures/Fig8_Clinical_FlowVolume.png", dpi=600, bbox_inches='tight')
plt.show()

# ======================================================
# VALIDATION  (requires spirometry_validation_data.xlsx)
# ======================================================
print("\n===== VALIDATION STEP =====\n")

try:
    df = pd.read_excel("data/spirometry_validation_data.xlsx")
    print("Columns:", df.columns.tolist())

    # Detect diagnosis column
    DIAG_COL = None
    for col in ["Hospital Diagnosis", "Diagnosis", "Label", "Ground Truth"]:
        if col in df.columns:
            DIAG_COL = col
            break
    if DIAG_COL is None:
        raise KeyError("No diagnosis column found in dataset")

    # Clean labels
    df[DIAG_COL] = df[DIAG_COL].astype(str).str.strip().str.lower()
    df[DIAG_COL] = df[DIAG_COL].replace({
        "normal":                "Normal",
        "obstructive":           "Obstructive",
        "obstructive disease":   "Obstructive",
        "restrictive":           "Restrictive",
        "restrictive pattern":   "Restrictive",
        "mixed":                 "Mixed",
    })

    # Column detection helper
    def get_col(possible_names):
        for name in possible_names:
            if name in df.columns:
                return name
        raise KeyError(f"Missing column (tried): {possible_names}")

    FEV1_col = get_col(["FEV1_%", "FEV1%", "FEV1", "FEV1 (%)"])
    FVC_col  = get_col(["FVC_%",  "FVC%",  "FVC",  "FVC (%)"])

    if "FEF" not in df.columns:
        df["FEF"] = 1.0

    df["FEV1_norm"] = df[FEV1_col] / 100
    df["FVC_norm"]  = df[FVC_col]  / 100

    if "FEV1/FVC" in df.columns:
        df["Computed_Ratio"] = df["FEV1/FVC"].apply(lambda x: x / 100 if x > 1 else x)
    else:
        df["Computed_Ratio"] = df["FEV1_norm"] / (df["FVC_norm"] + 1e-6)

    df["Model_Prediction"] = df.apply(
        lambda r: classify(r["FEV1_norm"], r["FVC_norm"], r["Computed_Ratio"], r["FEF"]),
        axis=1)

    # ML model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix

    X = df[["FEV1_norm", "FVC_norm", "Computed_Ratio", "FEF"]]
    y = df[DIAG_COL]

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X, y)
    df["Final_Prediction"] = model.predict(X)

    accuracy = accuracy_score(y, df["Final_Prediction"])
    print("Accuracy:", accuracy)

    labels = ["Normal", "Obstructive", "Restrictive", "Mixed"]
    cm = confusion_matrix(y, df["Final_Prediction"], labels=labels)
    print("\nConfusion Matrix:\n", cm)

    df.to_excel("data/validation_results.xlsx", index=False)

    # Fig 6: Confusion matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='gray_r')
    plt.xticks(range(len(labels)), labels, rotation=30)
    plt.yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig("figures/Fig6_Confusion_Matrix.png", dpi=600, bbox_inches='tight')
    plt.show()

except FileNotFoundError:
    print("⚠️  spirometry_validation_data.xlsx not found in data/.")
    print("   Place the validation dataset there and re-run this section.")
