# Unified Respiratory Airway Mechanics (RAM) Model

[![DOI](https://zenodo.org/badge/DOI/PLACEHOLDER.svg)](https://doi.org/PLACEHOLDER)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://www.python.org/)

Companion code for:

> **"A Unified Nonlinear Respiratory Airway Mechanics (RAM) Model for Physics-Based Spirometric Classification"**  
> *Medical & Biological Engineering & Computing (MBEC)*, 2024.  
> DOI: [to be added after acceptance]

---

## Overview

The RAM model is a single second-order nonlinear ODE that integrates:

- Nonlinear Rohrer airway resistance  
- Volume-dependent lung compliance (Padé approximation, Manuscript Eq. 6)  
- Dynamic airway collapse (Starling mechanism)  
- Inertial effects  

Forward simulation across normal, obstructive, restrictive, and mixed parameter regimes reproduces clinically recognised spirometric morphologies, achieving 90 % classification agreement against a 30-record hybrid validation dataset.

### Governing equation (Manuscript Eq. 7)

```
I·V̈ + (R₀ + Rᶜ·e^{-kV})·|V̇|·V̇ + V / [C₀·(1 − α·V/2)] = P_muscle(t)
```

Under quasi-static conditions (I → 0), this reduces to the operational flow equation solved by RK4 integration (Manuscript Eq. 10).

---

## Repository structure

```
unified_ram/
├── unified_RAM.py                  ← Main simulation, figures, validation
├── data/
│   ├── ram_time_series.csv         ← Generated: flow & volume for all cases
│   ├── ram_summary.csv             ← Generated: FEV1, FVC, ratio, PEF, FEF
│   ├── ram_parameter_variation.csv ← Generated: R–C parameter space scan
│   └── spirometry_validation_data.xlsx  ← Place validation dataset here
├── figures/                        ← All output figures (600 dpi PNG)
│   ├── Fig1_ODE_FlowVolume.png
│   ├── Fig2_Volume_Time.png
│   ├── Fig3_Parameter_Map.png
│   ├── Fig4_Model_Comparison.png
│   ├── Fig5_Hysteresis.png
│   ├── Fig6_Confusion_Matrix.png
│   ├── Fig7_Airflow_vs_Volume.png
│   └── Fig8_Clinical_FlowVolume.png
├── docs/
│   └── equation_notes.md           ← Derivation notes and Padé error bounds
├── requirements.txt
└── LICENSE
```

---

## Elastic pressure formulation

All results in the manuscript and supplementary material use the **Padé-corrected** form:

```
P_elastic(V) = V / [C₀ · (1 − α·V/2)]          ← Manuscript Eq. 6 / Supp. Eq. S1.32
```

This is the first-order [1/1] Padé approximation of the exact logarithmic law  
`P_elastic = −(1/αC₀)·ln(1 − αV)` (Supplementary Eq. S1.29).  
It introduces < 8 % error for α·V ≲ 0.3 (the physiological operating range).

> **Important:** An earlier draft version used `1 − α·V` (Taylor, not Padé) in the  
> denominator. That version was **not** used to generate any figures or results in  
> the published manuscript. The corrected `1 − α·V/2` form is implemented in  
> `unified_RAM.py` from the first release.

---

## Installation

```bash
git clone https://github.com/<username>/unified_ram.git
cd unified_ram
pip install -r requirements.txt
```

**Python 3.9 or later required.**

---

## Quick start

```bash
python unified_RAM.py
```

Outputs saved to `data/` (CSV) and `figures/` (PNG, 600 dpi).

To run the validation section, place your `spirometry_validation_data.xlsx`  
in the `data/` folder before running.

---

## Parameter reference (Table 2 in manuscript)

| Parameter | Symbol | Normal range | Unit |
|-----------|--------|-------------|------|
| Baseline resistance | R₀ | 1–2 | cmH₂O·s·L⁻¹ |
| Turbulent resistance | R₁ | 0.5–1 | cmH₂O·s²·L⁻² |
| Collapse resistance | Rᶜ | 1–2 | cmH₂O·s·L⁻¹ |
| Collapse rate constant | k | 0.3–0.7 | L⁻¹ |
| Baseline compliance | C₀ | 0.05–0.07 | L·cmH₂O⁻¹ |
| Compliance nonlinearity | α | 0.02–0.05 | L⁻¹ |

---

## Citation

If you use this code, please cite:

```bibtex
@article{ram2024,
  author  = {[Author names]},
  title   = {A Unified Nonlinear Respiratory Airway Mechanics (RAM) Model
             for Physics-Based Spirometric Classification},
  journal = {Medical \& Biological Engineering \& Computing},
  year    = {2024},
  doi     = {[DOI]}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).
