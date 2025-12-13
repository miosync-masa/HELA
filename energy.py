"""
HELA: Energy Computation Module
===============================

Three-tier harmonic energy and melodic energy computation.

Energy Model:
    E_total = E_harmonic + α × E_melodic

    where:
    - E_harmonic = λ_max(K) × w_func × w_res  (three-tier consonance)
    - E_melodic = ½m(Δx)²                      (quadratic motion cost)

Tier Structure:
    Tier 1 (Rigorous):  Base consonance from Consonance Tensor
    Tier 2 (Heuristic): Functional harmony weighting
    Tier 3 (Heuristic): Resolution pressure

See paper Section 3-4 for theoretical foundations.
See SI-B for mathematical proofs and justifications.

Author: Masamichi Iizumi
License: MIT
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from .consonance import harmonic_energy, CHORD_RATIOS
from .harmony import (
    get_roman_numeral,
    get_functional_weight,
    get_resolution_pressure,
    analyze_progression,
)


# ==============================================================================
# Pre-computed Chord Energies (Tier 1)
# ==============================================================================

# Base consonance energies computed from Consonance Tensor
# These are λ_max values for standard chord types in Just Intonation
CHORD_ENERGIES: Dict[str, float] = {
    # === Major triads (4:5:6) → E = 11.52 ===
    'C':  11.523, 'D':  11.523, 'E':  11.523,
    'F':  11.523, 'G':  11.523, 'A':  11.523,
    'B':  11.523, 'Bb': 11.523, 'Eb': 11.523,
    'Ab': 11.523, 'Db': 11.523, 'F#': 11.523,
    
    # === Minor triads (10:12:15) → E = 12.55 ===
    'Am':  12.550, 'Bm':  12.550, 'Cm':  12.550,
    'Dm':  12.550, 'Em':  12.550, 'Fm':  12.550,
    'Gm':  12.550, 'F#m': 12.550, 'C#m': 12.550,
    'G#m': 12.550, 'Bbm': 12.550, 'Ebm': 12.550,
    
    # === Dominant 7ths (36:45:54:64) → E = 15.18 ===
    'C7':  15.175, 'D7':  15.175, 'E7':  15.175,
    'F7':  15.175, 'G7':  15.175, 'A7':  15.175,
    'B7':  15.175, 'Bb7': 15.175, 'Eb7': 15.175,
    
    # === Augmented triads (16:20:25) → E = 12.15 ===
    'Caug': 12.150, 'Daug': 12.150, 'Eaug': 12.150,
    'Faug': 12.150, 'Gaug': 12.150, 'Aaug': 12.150,
    
    # === Diminished 7ths (125:150:180:216) → E = 28.30 ===
    'Cdim7': 28.300, 'Ddim7': 28.300, 'Edim7': 28.300,
    'Fdim7': 28.300, 'Gdim7': 28.300, 'Adim7': 28.300,
    
    # === Half-diminished 7ths → E ≈ 18.5 ===
    'Cm7b5': 18.450, 'Dm7b5': 18.450, 'Em7b5': 18.450,
    'Fm7b5': 18.450, 'Gm7b5': 18.450, 'Am7b5': 18.450,
}

# Default energy for unknown chords
DEFAULT_CHORD_ENERGY: float = 12.0


def get_chord_energy(chord: str) -> float:
    """
    Get base consonance energy (Tier 1) for a chord.
    
    Parameters
    ----------
    chord : str
        Chord symbol (e.g., 'C', 'Am', 'G7')
        
    Returns
    -------
    float
        Base energy E_1 = λ_max(K)
        
    Notes
    -----
    Values are pre-computed from Consonance Tensor eigenvalues.
    See consonance.py for computation details.
    """
    return CHORD_ENERGIES.get(chord, DEFAULT_CHORD_ENERGY)


# ==============================================================================
# Three-Tier Harmonic Energy
# ==============================================================================

def compute_tier1_energy(chord: str) -> float:
    """
    Compute Tier 1 energy: Base consonance.
    
    E_1 = λ_max(K)
    
    This is the RIGOROUS component, derived from spectral theory.
    
    Parameters
    ----------
    chord : str
        Chord symbol
        
    Returns
    -------
    float
        Base consonance energy
    """
    return get_chord_energy(chord)


def compute_tier2_energy(
    chord: str,
    key: str,
    mode: str = 'harmonic_minor'
) -> float:
    """
    Compute Tier 2 energy: Consonance × Function.
    
    E_2 = E_1 × w_func
    
    This adds HEURISTIC functional context to base consonance.
    
    Parameters
    ----------
    chord : str
        Chord symbol
    key : str
        Key signature (e.g., 'C Major', 'D minor')
    mode : str, optional
        Minor mode type (default: 'harmonic_minor')
        
    Returns
    -------
    float
        Functional energy E_2
        
    Examples
    --------
    >>> compute_tier2_energy('G', 'C Major')
    15.56  # 11.52 × 1.35 (V function)
    
    >>> compute_tier2_energy('C', 'C Major')
    11.52  # 11.52 × 1.00 (I function)
    """
    E_1 = compute_tier1_energy(chord)
    w_func = get_functional_weight(chord, key, mode)
    return E_1 * w_func


def compute_tier3_energy(
    chord: str,
    next_chord: Optional[str],
    key: str,
    mode: str = 'harmonic_minor'
) -> float:
    """
    Compute Tier 3 energy: Consonance × Function × Resolution.
    
    E_3 = E_2 × w_res = E_1 × w_func × w_res
    
    This adds HEURISTIC resolution dynamics to functional energy.
    
    Parameters
    ----------
    chord : str
        Current chord symbol
    next_chord : str or None
        Next chord symbol (None if last chord)
    key : str
        Key signature
    mode : str, optional
        Minor mode type (default: 'harmonic_minor')
        
    Returns
    -------
    float
        Full harmonic energy E_3
        
    Examples
    --------
    >>> compute_tier3_energy('G7', 'C', 'C Major')
    14.79  # Strong resolution: w_res = 0.80
    
    >>> compute_tier3_energy('G7', 'Am', 'C Major')
    21.26  # Deceptive cadence: w_res = 1.15
    """
    E_2 = compute_tier2_energy(chord, key, mode)
    
    if next_chord is not None:
        w_res = get_resolution_pressure(chord, next_chord, key, mode)
    else:
        w_res = 1.0  # No resolution context
    
    return E_2 * w_res


def compute_harmonic_energy(
    chord: str,
    key: str,
    next_chord: Optional[str] = None,
    tier: int = 3,
    mode: str = 'harmonic_minor'
) -> float:
    """
    Compute harmonic energy at specified tier level.
    
    This is the main interface for harmonic energy computation.
    
    Parameters
    ----------
    chord : str
        Chord symbol
    key : str
        Key signature
    next_chord : str or None, optional
        Next chord (required for Tier 3)
    tier : int, optional
        Energy tier (1, 2, or 3). Default is 3 (full model).
    mode : str, optional
        Minor mode type (default: 'harmonic_minor')
        
    Returns
    -------
    float
        Harmonic energy at specified tier
        
    Raises
    ------
    ValueError
        If tier is not 1, 2, or 3
        
    Examples
    --------
    >>> compute_harmonic_energy('G7', 'C Major', tier=1)
    15.18  # Base consonance only
    
    >>> compute_harmonic_energy('G7', 'C Major', tier=2)
    24.28  # + Functional weight (V7)
    
    >>> compute_harmonic_energy('G7', 'C Major', next_chord='C', tier=3)
    19.42  # + Resolution pressure (V7→I)
    """
    if tier == 1:
        return compute_tier1_energy(chord)
    elif tier == 2:
        return compute_tier2_energy(chord, key, mode)
    elif tier == 3:
        return compute_tier3_energy(chord, next_chord, key, mode)
    else:
        raise ValueError(f"Invalid tier: {tier}. Must be 1, 2, or 3.")


# ==============================================================================
# Melodic Energy
# ==============================================================================

def compute_melodic_energy(
    pitch_current: int,
    pitch_next: int,
    mass: float = 1.0
) -> float:
    """
    Compute melodic motion energy (kinetic term).
    
    E_m = ½ m (Δx)²
    
    where Δx is the interval in semitones.
    
    This is a RIGOROUS component (quadratic cost by definition).
    
    Parameters
    ----------
    pitch_current : int
        Current pitch (MIDI number)
    pitch_next : int
        Next pitch (MIDI number)
    mass : float, optional
        Motion penalty coefficient (default: 1.0)
        
    Returns
    -------
    float
        Melodic energy (motion cost)
        
    Examples
    --------
    >>> compute_melodic_energy(60, 62)  # Major 2nd (2 semitones)
    2.0
    
    >>> compute_melodic_energy(60, 67)  # Perfect 5th (7 semitones)
    24.5
    
    >>> compute_melodic_energy(60, 72)  # Octave (12 semitones)
    72.0
    
    Notes
    -----
    The quadratic scaling reflects perceptual and vocal production
    constraints: stepwise motion is natural, large leaps are costly.
    """
    delta = abs(pitch_next - pitch_current)
    return 0.5 * mass * delta ** 2


def compute_melodic_energy_sequence(
    pitches: List[int],
    mass: float = 1.0
) -> List[float]:
    """
    Compute melodic energy for a pitch sequence.
    
    Parameters
    ----------
    pitches : List[int]
        Sequence of MIDI pitches
    mass : float, optional
        Motion penalty coefficient
        
    Returns
    -------
    List[float]
        Melodic energy at each transition
        Length is len(pitches) - 1
    """
    energies = []
    for i in range(len(pitches) - 1):
        E_m = compute_melodic_energy(pitches[i], pitches[i + 1], mass)
        energies.append(E_m)
    return energies


# ==============================================================================
# Pitch-Chord Alignment
# ==============================================================================

# MIDI pitch class (0-11) to note name mapping
PITCH_CLASS_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 
                     'F#', 'G', 'G#', 'A', 'A#', 'B']

# Alternative enharmonic spellings
ENHARMONIC = {
    'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab', 'A#': 'Bb',
    'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
}

# Chord tones for common chord types (pitch classes relative to root)
CHORD_TONES: Dict[str, List[int]] = {
    'major':    [0, 4, 7],         # Root, M3, P5
    'minor':    [0, 3, 7],         # Root, m3, P5
    'dom7':     [0, 4, 7, 10],     # Root, M3, P5, m7
    'maj7':     [0, 4, 7, 11],     # Root, M3, P5, M7
    'min7':     [0, 3, 7, 10],     # Root, m3, P5, m7
    'dim':      [0, 3, 6],         # Root, m3, dim5
    'dim7':     [0, 3, 6, 9],      # Root, m3, dim5, dim7
    'aug':      [0, 4, 8],         # Root, M3, aug5
    'sus4':     [0, 5, 7],         # Root, P4, P5
    'sus2':     [0, 2, 7],         # Root, M2, P5
}


def parse_chord_symbol(chord: str) -> Tuple[str, str]:
    """
    Parse chord symbol into root and quality.
    
    Parameters
    ----------
    chord : str
        Chord symbol (e.g., 'C', 'Am', 'G7', 'F#m')
        
    Returns
    -------
    Tuple[str, str]
        (root, quality) - root note and chord quality
        
    Examples
    --------
    >>> parse_chord_symbol('C')
    ('C', 'major')
    
    >>> parse_chord_symbol('Am')
    ('A', 'minor')
    
    >>> parse_chord_symbol('G7')
    ('G', 'dom7')
    """
    # Handle sharps/flats in root
    if len(chord) > 1 and chord[1] in '#b':
        root = chord[:2]
        suffix = chord[2:]
    else:
        root = chord[0]
        suffix = chord[1:]
    
    # Determine quality
    if suffix == '' or suffix.startswith('maj') and '7' not in suffix:
        quality = 'major'
    elif suffix == 'm' or suffix.startswith('min'):
        quality = 'minor'
    elif suffix == '7' or suffix == 'dom7':
        quality = 'dom7'
    elif suffix == 'maj7' or suffix == 'M7':
        quality = 'maj7'
    elif suffix in ['m7', 'min7']:
        quality = 'min7'
    elif suffix in ['dim', 'dim7', 'o', 'o7']:
        quality = 'dim7' if '7' in suffix else 'dim'
    elif suffix in ['aug', '+']:
        quality = 'aug'
    elif suffix == 'sus4':
        quality = 'sus4'
    elif suffix == 'sus2':
        quality = 'sus2'
    else:
        quality = 'major'  # Default
    
    return root, quality


def get_chord_pitch_classes(chord: str) -> List[int]:
    """
    Get the pitch classes (0-11) for a chord.
    
    Parameters
    ----------
    chord : str
        Chord symbol
        
    Returns
    -------
    List[int]
        Pitch classes of chord tones
    """
    root, quality = parse_chord_symbol(chord)
    
    # Root pitch class
    try:
        root_pc = PITCH_CLASS_NAMES.index(root)
    except ValueError:
        # Try enharmonic spelling
        root_pc = PITCH_CLASS_NAMES.index(ENHARMONIC.get(root, 'C'))
    
    # Get intervals for quality
    intervals = CHORD_TONES.get(quality, [0, 4, 7])
    
    # Compute pitch classes
    return [(root_pc + interval) % 12 for interval in intervals]


def pitch_chord_alignment(
    pitch: int,
    chord: str,
    chord_tone_factor: float = 0.9,
    non_chord_tone_factor: float = 1.3
) -> float:
    """
    Compute pitch-chord alignment factor.
    
    This is a HEURISTIC multiplier that adjusts harmonic energy
    based on whether the melody note is a chord tone.
    
    Parameters
    ----------
    pitch : int
        MIDI pitch number
    chord : str
        Chord symbol
    chord_tone_factor : float, optional
        Multiplier for chord tones (default: 0.9, more stable)
    non_chord_tone_factor : float, optional
        Multiplier for non-chord tones (default: 1.3, more tension)
        
    Returns
    -------
    float
        Alignment factor
        
    Examples
    --------
    >>> pitch_chord_alignment(60, 'C')   # C over C major
    0.9  # Chord tone (root)
    
    >>> pitch_chord_alignment(62, 'C')   # D over C major
    1.3  # Non-chord tone
    
    >>> pitch_chord_alignment(64, 'C')   # E over C major
    0.9  # Chord tone (3rd)
    """
    pitch_class = pitch % 12
    chord_pcs = get_chord_pitch_classes(chord)
    
    if pitch_class in chord_pcs:
        return chord_tone_factor
    else:
        return non_chord_tone_factor


# ==============================================================================
# Combined Energy (Lagrangian)
# ==============================================================================

def compute_total_energy(
    pitch: int,
    chord: str,
    key: str,
    pitch_next: Optional[int] = None,
    chord_next: Optional[str] = None,
    tier: int = 3,
    mode: str = 'harmonic_minor',
    mass: float = 1.0,
    alpha: float = 0.1,
    use_alignment: bool = True
) -> Dict[str, float]:
    """
    Compute total energy combining harmonic and melodic components.
    
    E_total = E_harmonic × f_align + α × E_melodic
    
    Parameters
    ----------
    pitch : int
        Current melody pitch (MIDI)
    chord : str
        Current chord symbol
    key : str
        Key signature
    pitch_next : int or None, optional
        Next melody pitch (for melodic energy)
    chord_next : str or None, optional
        Next chord symbol (for resolution pressure)
    tier : int, optional
        Harmonic energy tier (1, 2, or 3)
    mode : str, optional
        Minor mode type
    mass : float, optional
        Melodic mass parameter
    alpha : float, optional
        Harmonic-melodic coupling coefficient
    use_alignment : bool, optional
        Whether to apply pitch-chord alignment factor
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'E_h': Harmonic energy (potential)
        - 'E_m': Melodic energy (kinetic)
        - 'E_total': Combined energy
        - 'f_align': Pitch-chord alignment factor
        
    Notes
    -----
    The Lagrangian form used in trajectory analysis is:
        L = K - V = E_m - E_h
    
    This function returns the individual components for flexibility.
    """
    # Harmonic energy
    E_h = compute_harmonic_energy(chord, key, chord_next, tier, mode)
    
    # Pitch-chord alignment
    if use_alignment:
        f_align = pitch_chord_alignment(pitch, chord)
        E_h = E_h * f_align
    else:
        f_align = 1.0
    
    # Melodic energy
    if pitch_next is not None:
        E_m = compute_melodic_energy(pitch, pitch_next, mass)
    else:
        E_m = 0.0
    
    # Combined
    E_total = E_h + alpha * E_m
    
    return {
        'E_h': E_h,
        'E_m': E_m,
        'E_total': E_total,
        'f_align': f_align,
    }


# ==============================================================================
# Progression Energy Analysis
# ==============================================================================

def analyze_progression_energy(
    chords: List[str],
    key: str,
    melody: Optional[List[int]] = None,
    tier: int = 3,
    mode: str = 'harmonic_minor',
    mass: float = 1.0,
    alpha: float = 0.1
) -> Dict[str, Union[List[float], float]]:
    """
    Analyze energy landscape for a chord progression.
    
    Parameters
    ----------
    chords : List[str]
        Chord sequence
    key : str
        Key signature
    melody : List[int] or None, optional
        Melody pitch sequence (same length as chords)
    tier : int, optional
        Energy tier (1, 2, or 3)
    mode : str, optional
        Minor mode type
    mass : float, optional
        Melodic mass parameter
    alpha : float, optional
        Coupling coefficient
        
    Returns
    -------
    Dict
        Analysis results containing:
        - 'E_h': List of harmonic energies
        - 'E_m': List of melodic energies (if melody provided)
        - 'E_total': List of total energies
        - 'numerals': List of Roman numerals
        - 'max_E': Maximum energy
        - 'min_E': Minimum energy
        - 'range_E': Energy range
        - 'mean_E': Mean energy
        - 'dramatic_arc': Range / Mean ratio
    """
    n = len(chords)
    
    # Harmonic analysis
    E_h_list = []
    numerals = []
    
    for i, chord in enumerate(chords):
        next_chord = chords[i + 1] if i < n - 1 else None
        E_h = compute_harmonic_energy(chord, key, next_chord, tier, mode)
        E_h_list.append(E_h)
        numerals.append(get_roman_numeral(chord, key, mode))
    
    # Melodic analysis (if melody provided)
    if melody is not None and len(melody) == n:
        E_m_list = []
        for i in range(n):
            if i < n - 1:
                E_m = compute_melodic_energy(melody[i], melody[i + 1], mass)
            else:
                E_m = 0.0
            E_m_list.append(E_m)
        
        E_total_list = [E_h_list[i] + alpha * E_m_list[i] for i in range(n)]
    else:
        E_m_list = [0.0] * n
        E_total_list = E_h_list.copy()
    
    # Statistics
    E_arr = np.array(E_h_list)
    max_E = float(np.max(E_arr))
    min_E = float(np.min(E_arr))
    range_E = max_E - min_E
    mean_E = float(np.mean(E_arr))
    dramatic_arc = range_E / mean_E if mean_E > 0 else 0.0
    
    return {
        'E_h': E_h_list,
        'E_m': E_m_list,
        'E_total': E_total_list,
        'numerals': numerals,
        'max_E': max_E,
        'min_E': min_E,
        'range_E': range_E,
        'mean_E': mean_E,
        'dramatic_arc': dramatic_arc,
    }


# ==============================================================================
# Demonstration
# ==============================================================================

def demonstrate_three_tier_energy() -> None:
    """
    Demonstrate three-tier energy computation.
    """
    print("=" * 70)
    print("  THREE-TIER ENERGY DEMONSTRATION")
    print("=" * 70)
    
    # Test case: Senbonzakura pre-chorus
    chords = ['Bb', 'F', 'G7', 'A']
    key = 'D minor'
    
    print(f"\n  Progression: {' - '.join(chords)}")
    print(f"  Key: {key}")
    print()
    
    print(f"  {'Chord':<8} {'Roman':<8} {'Tier 1':<10} {'Tier 2':<10} {'Tier 3':<10}")
    print("-" * 70)
    
    for i, chord in enumerate(chords):
        next_chord = chords[i + 1] if i < len(chords) - 1 else None
        
        E_1 = compute_tier1_energy(chord)
        E_2 = compute_tier2_energy(chord, key)
        E_3 = compute_tier3_energy(chord, next_chord, key)
        numeral = get_roman_numeral(chord, key)
        
        print(f"  {chord:<8} {numeral:<8} {E_1:<10.2f} {E_2:<10.2f} {E_3:<10.2f}")
    
    print()
    
    # Summary statistics
    analysis = analyze_progression_energy(chords, key, tier=3)
    print(f"  Max Energy:    {analysis['max_E']:.2f}")
    print(f"  Min Energy:    {analysis['min_E']:.2f}")
    print(f"  Energy Range:  {analysis['range_E']:.2f}")
    print(f"  Dramatic Arc:  {analysis['dramatic_arc']:.2f}")
    
    print("\n" + "=" * 70)


def demonstrate_melodic_energy() -> None:
    """
    Demonstrate melodic energy computation.
    """
    print("=" * 70)
    print("  MELODIC ENERGY DEMONSTRATION")
    print("=" * 70)
    
    intervals = [
        ('Unison', 0),
        ('Minor 2nd', 1),
        ('Major 2nd', 2),
        ('Minor 3rd', 3),
        ('Major 3rd', 4),
        ('Perfect 4th', 5),
        ('Tritone', 6),
        ('Perfect 5th', 7),
        ('Octave', 12),
    ]
    
    print(f"\n  {'Interval':<16} {'Semitones':<12} {'E_m (m=1)':<12}")
    print("-" * 50)
    
    base_pitch = 60  # Middle C
    for name, semitones in intervals:
        E_m = compute_melodic_energy(base_pitch, base_pitch + semitones)
        print(f"  {name:<16} {semitones:<12} {E_m:<12.1f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demonstrate_three_tier_energy()
    print()
    demonstrate_melodic_energy()
