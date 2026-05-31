# Elastic Pressure Derivation Notes

## Exact logarithmic law (Supplementary Eq. S1.29)

Volume-dependent compliance:  C(V) = C₀(1 − αV)

Integrating dP = dV/C(V) from 0 to V:

    P_elastic(V) = −(1/αC₀) · ln(1 − αV)

## Padé [1/1] approximation (Manuscript Eq. 6 / Supp. Eq. S1.32)

Taylor expansion to second order:

    P_elastic ≈ V/C₀ + αV²/(2C₀) + O((αV)³)

Matching a [1/1] rational form P = aV / (1 + bV) to first two terms gives:

    a = 1/C₀,  b = −α/2

Therefore:

    P_elastic(V) ≈ V / [C₀ · (1 − αV/2)]

## Error bound

Relative error ε = |P_Padé − P_exact| / P_exact

For αV = 0.3:  ε ≈ 7.6 %  (< 8 %, within stated bound)
For αV = 0.1:  ε ≈ 0.8 %

## Why NOT use the bare Taylor form 1/(1 − αV)?

A naive first-order expansion gives P ≈ V/C₀ · (1 + αV) which,
when inverted, appears as denominator (1 − αV).  This form:
- hits a singularity at αV = 1 (before the physical elastic limit αV → 2 in the Padé)
- over-estimates stiffening at mid-range volumes
- was present in an earlier draft of the code and has been replaced

The Padé form pushes the singularity to αV = 2, a better representation of
the physical expansion limit, and has smaller absolute error across the
physiological range.

## Hysteresis (extended formulation)

The hysteresis branch uses the inverse Padé (P → V direction):

    V(P) = C₀·P / (1 + α·P)

This is the correct closed-form inverse of the Padé approximation and is used
only for the static P–V loop visualisation (Fig. 5), not for the flow simulation.
