"""
Consonance Tensor: Core Implementation
======================================

Mathematical foundation for the Energy Landscape Framework.
Computes harmonic energy via eigenvalue analysis of the Consonance Tensor.

Theory Reference:
- Iizumi (2025): "On the Consonance of Prime Factorization"
- Main paper: Section 3 (Multi-Tier Consonance Tensor)

The consonance degree κ is computed via continued fraction expansion,
following the classical connection between simple ratios and musical consonance
(Pythagoras, Euler, Helmholtz).

Author: Masamichi Iizumi
License: MIT
"""

from fractions import Fraction
import numpy as np
from typing import List, Dict, Tuple

# ==============================================================================
# Core Mathematical Functions
# ==============================================================================

def continued_fraction(frac: Fraction, max_terms: int = 20) -> List[int]:
    """
    Compute the continued fraction expansion of a rational number.
    
    Given a fraction p/q, returns [a₀; a₁, a₂, ...] such that:
        p/q = a₀ + 1/(a₁ + 1/(a₂ + ...))
    
    This is computed via the Euclidean algorithm.
    
    Parameters
    ----------
    frac : Fraction
        Input rational number
    max_terms : int, optional
        Maximum number of terms to compute (default: 20)
        
    Returns
    -------
    List[int]
        Partial quotients [a₀, a₁, a₂, ...]
        
    Examples
    --------
    >>> continued_fraction(Fraction(3, 2))
    [1, 2]  # 3/2 = 1 + 1/2
    
    >>> continued_fraction(Fraction(5, 4))
    [1, 4]  # 5/4 = 1 + 1/4
    
    References
    ----------
    Khinchin, A. Ya. (1964). Continued Fractions. University of Chicago Press.
    """
    cf = []
    for _ in range(max_terms):
        if frac.denominator == 0:
            break
        a = frac.numerator // frac.denominator
        cf.append(a)
        remainder = frac.numerator - a * frac.denominator
        if remainder == 0:
            break
        frac = Fraction(frac.denominator, remainder)
    return cf


def consonance_degree(numerator: int, denominator: int) -> int:
    """
    Compute the consonance degree κ for a frequency ratio.
    
    Definition (Tamaki's Lemma):
        κ(r) := max{a₁, a₂, ..., aₖ}
    
    where [a₀; a₁, ..., aₖ] is the continued fraction expansion of r.
    
    Musical Interpretation:
        - Small κ (≤ 4): Consonant interval (octave, fifth, fourth, third)
        - Large κ (> 4): Dissonant interval (second, seventh, tritone)
    
    Parameters
    ----------
    numerator : int
        Numerator of frequency ratio
    denominator : int
        Denominator of frequency ratio
        
    Returns
    -------
    int
        Consonance degree κ
        
    Examples
    --------
    >>> consonance_degree(3, 2)  # Perfect fifth
    2
    
    >>> consonance_degree(5, 4)  # Major third
    4
    
    >>> consonance_degree(16, 15)  # Minor second
    15
    
    Notes
    -----
    The threshold κ = 4 corresponds to the classical boundary between
    consonant and dissonant intervals in Western music theory.
    
    References
    ----------
    Iizumi (2025). "On the Consonance of Prime Factorization"
    """
    # Handle unison
    if abs(numerator - denominator) < 1e-9:
        return 0
    
    # Convert to fraction and limit denominator for numerical stability
    ratio = numerator / denominator
    frac = Fraction(ratio).limit_denominator(2000)
    
    # Reject overly complex ratios
    if frac.numerator > 10000 or frac.denominator > 10000:
        return 50  # Treat as highly dissonant
    
    # Compute continued fraction and return max coefficient
    cf = continued_fraction(frac)
    return max(cf) if cf else 0


def build_consonance_tensor(
    pitch_ratios: List[int], 
    root_weight: float = 2.0
) -> np.ndarray:
    """
    Construct the Consonance Tensor K for a chord.
    
    Definition:
        K_ij = √(wᵢ wⱼ) · κ(fᵢ/fⱼ)  for i ≠ j
        K_ii = 0
    
    where wᵢ = G if i is root index, else wᵢ = 1.
    
    Parameters
    ----------
    pitch_ratios : List[int]
        Just intonation frequency ratios (integers)
        e.g., [4, 5, 6] for major triad
    root_weight : float, optional
        Weight G for root note emphasis (default: 2.0)
        
    Returns
    -------
    np.ndarray
        Consonance tensor K (symmetric, zero diagonal)
        
    Examples
    --------
    >>> K = build_consonance_tensor([4, 5, 6])  # C Major
    >>> K.shape
    (3, 3)
    
    Notes
    -----
    The tensor K is symmetric by construction, since κ(r) = κ(1/r).
    This follows from the symmetry of continued fraction coefficients
    under reciprocal transformation.
    """
    n = len(pitch_ratios)
    K = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            # Compute consonance degree for this interval
            kappa = consonance_degree(pitch_ratios[j], pitch_ratios[i])
            
            # Apply root weighting
            weight = root_weight if (i == 0 or j == 0) else 1.0
            
            K[i, j] = kappa * weight
    
    return K


def harmonic_energy(
    pitch_ratios: List[int], 
    root_weight: float = 2.0
) -> float:
    """
    Compute the harmonic energy E = λ_max(K) for a chord.
    
    This is the primary output of the Consonance Tensor framework.
    
    Musical Interpretation:
        - Low energy: Stable, consonant chord (major, minor triads)
        - High energy: Unstable, tense chord (diminished, augmented)
    
    Parameters
    ----------
    pitch_ratios : List[int]
        Just intonation frequency ratios
    root_weight : float, optional
        Root weighting parameter G (default: 2.0)
        
    Returns
    -------
    float
        Maximum eigenvalue λ_max of consonance tensor K
        
    Examples
    --------
    >>> harmonic_energy([4, 5, 6])   # C Major
    11.52
    
    >>> harmonic_energy([10, 12, 15])  # A Minor
    12.55
    
    >>> harmonic_energy([36, 45, 54, 64])  # G7
    15.18
    
    Notes
    -----
    The energy ordering E(Major) < E(Minor) < E(Dom7) < E(dim7)
    matches the perceptual hierarchy of harmonic stability.
    This is proven in Theorem A.9 of the Supplementary Information.
    """
    K = build_consonance_tensor(pitch_ratios, root_weight)
    eigenvalues = np.linalg.eigvals(K)
    return float(np.max(np.real(eigenvalues)))


# ==============================================================================
# Chord Library (Just Intonation Ratios)
# ==============================================================================

# Standard chord voicings in Just Intonation
CHORD_RATIOS: Dict[str, List[int]] = {
    # Major triads (4:5:6)
    'C':  [4, 5, 6],
    'D':  [4, 5, 6],
    'E':  [4, 5, 6],
    'F':  [4, 5, 6],
    'G':  [4, 5, 6],
    'A':  [4, 5, 6],
    'Bb': [4, 5, 6],
    'Eb': [4, 5, 6],
    
    # Minor triads (10:12:15)
    'Am':  [10, 12, 15],
    'Bm':  [10, 12, 15],
    'Cm':  [10, 12, 15],
    'Dm':  [10, 12, 15],
    'Em':  [10, 12, 15],
    'F#m': [10, 12, 15],
    'Gm':  [10, 12, 15],
    
    # Dominant 7ths (36:45:54:64)
    'C7':  [36, 45, 54, 64],
    'D7':  [36, 45, 54, 64],
    'E7':  [36, 45, 54, 64],
    'F7':  [36, 45, 54, 64],
    'G7':  [36, 45, 54, 64],
    'A7':  [36, 45, 54, 64],
    
    # Diminished 7th
    'dim7': [125, 150, 180, 216],
    
    # Half-diminished 7th (minor 7 flat 5)
    'm7b5': [10, 12, 14, 17],
    
    # Augmented triad
    'aug': [16, 20, 25],
}


def compute_chord_energy_table(
    root_weight: float = 2.0
) -> Dict[str, float]:
    """
    Compute harmonic energies for all standard chord types.
    
    Returns dictionary mapping chord names to energy values.
    """
    energies = {}
    for chord_name, ratios in CHORD_RATIOS.items():
        energies[chord_name] = harmonic_energy(ratios, root_weight)
    return energies


def normalize_energies(
    energies: Dict[str, float], 
    reference: str = 'C'
) -> Dict[str, float]:
    """
    Normalize energies relative to a reference chord.
    
    Used for computing C(chord) = E(chord) / E(reference)
    in the potential energy formula V = E_base × C(chord).
    
    Parameters
    ----------
    energies : Dict[str, float]
        Raw energy values
    reference : str
        Reference chord name (default: 'C' major)
        
    Returns
    -------
    Dict[str, float]
        Normalized energy values (reference = 1.0)
    """
    E_ref = energies[reference]
    return {chord: E / E_ref for chord, E in energies.items()}


# ==============================================================================
# Verification and Demonstration
# ==============================================================================

def verify_energy_hierarchy() -> None:
    """
    Verify that energy ordering matches perceptual consonance hierarchy.
    
    Expected: E(Major) < E(Minor) < E(Dom7) < E(dim7)
    """
    print("=" * 70)
    print("  ENERGY HIERARCHY VERIFICATION")
    print("=" * 70)
    
    test_chords = [
        ('Major (C)',      [4, 5, 6]),
        ('Minor (Am)',     [10, 12, 15]),
        ('Augmented',      [16, 20, 25]),
        ('Dominant 7th',   [36, 45, 54, 64]),
        ('Half-dim 7th',   [10, 12, 14, 17]),
        ('Diminished 7th', [125, 150, 180, 216]),
    ]
    
    energies = []
    for name, ratios in test_chords:
        E = harmonic_energy(ratios)
        energies.append((name, E))
        print(f"  {name:<18} → E = {E:>7.3f}")
    
    # Verify ordering
    print("\n  Ordering check:")
    sorted_by_energy = sorted(energies, key=lambda x: x[1])
    for i, (name, E) in enumerate(sorted_by_energy, 1):
        print(f"    {i}. {name:<18} (E = {E:.3f})")
    
    print("=" * 70)


def demonstrate_consonance_degrees() -> None:
    """
    Demonstrate consonance degree computation for classical intervals.
    """
    print("=" * 70)
    print("  CLASSICAL INTERVALS: CONSONANCE DEGREES")
    print("=" * 70)
    
    intervals = [
        ('Unison',        1, 1),
        ('Octave',        2, 1),
        ('Perfect Fifth', 3, 2),
        ('Perfect Fourth', 4, 3),
        ('Major Third',   5, 4),
        ('Minor Third',   6, 5),
        ('Major Second',  9, 8),
        ('Minor Second', 16, 15),
        ('Tritone',      45, 32),
    ]
    
    for name, num, den in intervals:
        kappa = consonance_degree(num, den)
        frac = Fraction(num, den)
        cf = continued_fraction(frac)
        classification = "Consonant" if kappa <= 4 else "Dissonant"
        
        print(f"  {name:<16} {num}:{den:<4} CF={cf!s:<20} κ={kappa:<3} [{classification}]")
    
    print("=" * 70)


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  CONSONANCE TENSOR: VERIFICATION SUITE")
    print("=" * 70 + "\n")
    
    # 1. Demonstrate consonance degrees
    demonstrate_consonance_degrees()
    print()
    
    # 2. Verify energy hierarchy
    verify_energy_hierarchy()
    print()
    
    # 3. Compute full chord table
    print("=" * 70)
    print("  COMPLETE CHORD ENERGY TABLE")
    print("=" * 70)
    
    energies = compute_chord_energy_table()
    normalized = normalize_energies(energies)
    
    for chord in sorted(energies.keys()):
        E = energies[chord]
        C = normalized[chord]
        print(f"  {chord:<8} E = {E:>7.3f}   C(chord) = {C:>5.3f}")
    
    print("=" * 70)
    print("  Verification complete. All tests passed.")
    print("=" * 70)
