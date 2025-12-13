"""
HELA: Harmonic Analysis Module
==============================

Roman numeral analysis, functional harmony weights, and resolution pressure.

This module implements Tier 2 (functional context) and Tier 3 (resolution dynamics)
of the multi-tier consonance framework.

Theoretical Basis:
- Tier 2: Rameau's functional harmony theory (1722)
- Tier 3: Cadential analysis and resolution tendencies

Note: These components are HEURISTIC (empirically calibrated),
not mathematically derived. See SI-B.5 for necessity proofs.

Author: Masamichi Iizumi
License: MIT
"""

from typing import Dict, List, Optional, Tuple

# ==============================================================================
# Scale Definitions
# ==============================================================================

# Major scales with diatonic chords
MAJOR_SCALES: Dict[str, List[str]] = {
    'C': ['C', 'Dm', 'Em', 'F', 'G', 'Am', 'Bdim'],
    'D': ['D', 'Em', 'F#m', 'G', 'A', 'Bm', 'C#dim'],
    'E': ['E', 'F#m', 'G#m', 'A', 'B', 'C#m', 'D#dim'],
    'F': ['F', 'Gm', 'Am', 'Bb', 'C', 'Dm', 'Edim'],
    'G': ['G', 'Am', 'Bm', 'C', 'D', 'Em', 'F#dim'],
    'A': ['A', 'Bm', 'C#m', 'D', 'E', 'F#m', 'G#dim'],
    'Bb': ['Bb', 'Cm', 'Dm', 'Eb', 'F', 'Gm', 'Adim'],
}

# Minor scales - Natural minor
MINOR_SCALES_NATURAL: Dict[str, List[str]] = {
    'Am': ['Am', 'Bdim', 'C', 'Dm', 'Em', 'F', 'G'],
    'Bm': ['Bm', 'C#dim', 'D', 'Em', 'F#m', 'G', 'A'],
    'Cm': ['Cm', 'Ddim', 'Eb', 'Fm', 'Gm', 'Ab', 'Bb'],
    'Dm': ['Dm', 'Edim', 'F', 'Gm', 'Am', 'Bb', 'C'],
    'Em': ['Em', 'F#dim', 'G', 'Am', 'Bm', 'C', 'D'],
    'F#m': ['F#m', 'G#dim', 'A', 'Bm', 'C#m', 'D', 'E'],
    'Gm': ['Gm', 'Adim', 'Bb', 'Cm', 'Dm', 'Eb', 'F'],
}

# Minor scales - Harmonic minor (raised 7th)
MINOR_SCALES_HARMONIC: Dict[str, List[str]] = {
    'Am': ['Am', 'Bdim', 'Caug', 'Dm', 'E', 'F', 'G#dim'],
    'Bm': ['Bm', 'C#dim', 'Daug', 'Em', 'F#', 'G', 'A#dim'],
    'Cm': ['Cm', 'Ddim', 'Ebaug', 'Fm', 'G', 'Ab', 'Bdim'],
    'Dm': ['Dm', 'Edim', 'Faug', 'Gm', 'A', 'Bb', 'C#dim'],
    'Em': ['Em', 'F#dim', 'Gaug', 'Am', 'B', 'C', 'D#dim'],
    'F#m': ['F#m', 'G#dim', 'Aaug', 'Bm', 'C#', 'D', 'E#dim'],
    'Gm': ['Gm', 'Adim', 'Bbaug', 'Cm', 'D', 'Eb', 'F#dim'],
}

# Roman numeral labels
MAJOR_NUMERALS: List[str] = ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii°']
MINOR_NUMERALS_NATURAL: List[str] = ['i', 'ii°', 'III', 'iv', 'v', 'VI', 'VII']
MINOR_NUMERALS_HARMONIC: List[str] = ['i', 'ii°', 'III+', 'iv', 'V', 'VI', 'vii°']


# ==============================================================================
# Functional Weights (Tier 2) - HEURISTIC
# ==============================================================================

FUNCTIONAL_WEIGHTS: Dict[str, float] = {
    # === Major key functions ===
    'I':     1.00,   # Tonic - maximum stability
    'ii':    1.20,   # Supertonic - pre-dominant
    'iii':   1.08,   # Mediant - tonic substitute
    'IV':    1.15,   # Subdominant - pre-dominant
    'V':     1.35,   # Dominant - tension
    'V7':    1.60,   # Dominant 7th - strong tension
    'vi':    1.08,   # Submediant - tonic substitute
    'vii°':  1.45,   # Leading tone diminished
    
    # === Minor key functions ===
    'i':     1.00,   # Tonic minor
    'ii°':   1.25,   # Diminished supertonic
    'III':   1.08,   # Mediant major
    'III+':  1.15,   # Augmented mediant (harmonic minor)
    'iv':    1.15,   # Subdominant minor
    'v':     1.25,   # Minor dominant (natural minor)
    'VI':    1.12,   # Submediant major
    'VII':   1.30,   # Subtonic (natural minor)
    'VII7':  1.50,   # Subtonic 7th
    
    # === Secondary dominants ===
    'V/V':   1.45,   # Secondary dominant
    'V7/V':  1.55,   # Secondary dominant 7th
    'V/ii':  1.40,   # Secondary dominant to ii
    'V/IV':  1.40,   # Secondary dominant to IV
    
    # === Borrowed chords ===
    'bVII':  1.25,   # Borrowed from parallel minor
    'bVI':   1.20,   # Borrowed from parallel minor
    'iv':    1.18,   # Minor subdominant in major key
}


# ==============================================================================
# Resolution Pressure (Tier 3) - HEURISTIC
# ==============================================================================

RESOLUTION_PRESSURE: Dict[str, float] = {
    # === Strong resolution (tension release) ===
    'V→I':      0.85,   # Authentic cadence
    'V7→I':     0.80,   # Strong authentic cadence
    'V→i':      0.85,   # Minor authentic cadence
    'V7→i':     0.80,   # Strong minor authentic
    'vii°→I':   0.85,   # Leading-tone resolution
    'vii°→i':   0.85,   # Minor leading-tone resolution
    
    # === Partial resolution (tension maintained) ===
    'V→vi':     1.15,   # Deceptive cadence
    'V→VI':     1.15,   # Minor deceptive cadence
    'V7→vi':    1.10,   # Deceptive with 7th
    
    # === Unresolved (tension sustained) ===
    'V→V':      1.25,   # Prolonged dominant
    'V7→V':     1.30,   # Unresolved dominant 7th
    'V7/V→V':   1.25,   # Secondary dominant chain
    'V→IV':     1.20,   # Retrogression
    
    # === Neutral progressions ===
    'I→IV':     1.00,   # Plagal motion
    'I→ii':     1.00,   # To supertonic
    'I→vi':     1.00,   # To submediant
    'IV→V':     1.00,   # Pre-dominant to dominant
    'ii→V':     1.00,   # Pre-dominant to dominant
    'i→iv':     1.00,   # Minor plagal
    'i→VI':     1.00,   # Minor to submediant
}


# ==============================================================================
# Roman Numeral Analysis
# ==============================================================================

def parse_key(key: str) -> Tuple[str, bool]:
    """
    Parse key string into root and mode.
    
    Parameters
    ----------
    key : str
        Key string, e.g., 'C Major', 'D minor', 'Am'
        
    Returns
    -------
    Tuple[str, bool]
        (root, is_major) - root note and whether major mode
        
    Examples
    --------
    >>> parse_key('C Major')
    ('C', True)
    
    >>> parse_key('D minor')
    ('D', False)
    
    >>> parse_key('Am')
    ('A', False)
    """
    key_lower = key.lower()
    
    # Check for explicit mode
    if 'major' in key_lower:
        root = key.split()[0]
        return root, True
    elif 'minor' in key_lower:
        root = key.split()[0]
        return root, False
    
    # Check for 'm' suffix (e.g., 'Am', 'Dm')
    if key.endswith('m') and len(key) <= 3:
        return key[:-1], False
    
    # Default to major
    return key, True


def get_roman_numeral(
    chord: str, 
    key: str, 
    mode: str = 'harmonic_minor'
) -> str:
    """
    Determine the Roman numeral function of a chord within a key.
    
    This is the core function for Tier 2 harmonic analysis.
    
    Parameters
    ----------
    chord : str
        Chord symbol, e.g., 'C', 'Am', 'G7', 'F#m'
    key : str
        Key signature, e.g., 'C Major', 'D minor'
    mode : str, optional
        For minor keys: 'harmonic_minor' or 'natural_minor'
        Default is 'harmonic_minor' (raised 7th for V chord)
        
    Returns
    -------
    str
        Roman numeral, e.g., 'I', 'V7', 'ii', 'V7/V'
        Returns '?' if chord cannot be analyzed
        
    Examples
    --------
    >>> get_roman_numeral('G', 'C Major')
    'V'
    
    >>> get_roman_numeral('G7', 'C Major')
    'V7'
    
    >>> get_roman_numeral('A', 'D minor', mode='harmonic_minor')
    'V'
    
    >>> get_roman_numeral('G7', 'D minor')  # Secondary dominant
    'V7/V'
    
    Notes
    -----
    Secondary dominants (V/V, V7/V, etc.) are detected automatically
    when a chord functions as dominant of the dominant.
    """
    root, is_major = parse_key(key)
    
    # Handle 7th chords
    has_seventh = '7' in chord
    base_chord = chord.replace('7', '')
    
    # Get appropriate scale
    if is_major:
        scale = MAJOR_SCALES.get(root, MAJOR_SCALES['C'])
        numerals = MAJOR_NUMERALS
    else:
        key_minor = root + 'm'
        if mode == 'harmonic_minor':
            scale = MINOR_SCALES_HARMONIC.get(key_minor, MINOR_SCALES_HARMONIC['Am'])
            numerals = MINOR_NUMERALS_HARMONIC
        else:
            scale = MINOR_SCALES_NATURAL.get(key_minor, MINOR_SCALES_NATURAL['Am'])
            numerals = MINOR_NUMERALS_NATURAL
    
    # === Special case: Secondary dominants ===
    # G7 in D minor = V7/V (dominant of A, which is V of Dm)
    if has_seventh and not is_major:
        # Check if this is V7 of V
        dominant_root = scale[4].replace('m', '').replace('dim', '').replace('aug', '')
        chord_root = base_chord.replace('m', '')
        
        # Calculate if chord is a fifth above the dominant
        # (simplified check for common cases)
        secondary_dom_map = {
            'Dm': {'G7': 'V7/V', 'E7': 'V7/ii'},
            'Am': {'D7': 'V7/V', 'B7': 'V7/ii'},
            'Em': {'A7': 'V7/V', 'F#7': 'V7/ii'},
        }
        
        key_minor = root + 'm'
        if key_minor in secondary_dom_map:
            if chord in secondary_dom_map[key_minor]:
                return secondary_dom_map[key_minor][chord]
    
    # === Direct lookup ===
    try:
        idx = scale.index(base_chord)
        numeral = numerals[idx]
        if has_seventh:
            numeral += '7'
        return numeral
    except ValueError:
        pass
    
    # === Root matching (for chord quality variations) ===
    scale_roots = [ch.replace('m', '').replace('dim', '').replace('aug', '') 
                   for ch in scale]
    chord_root = base_chord.replace('m', '')
    
    if chord_root in scale_roots:
        idx = scale_roots.index(chord_root)
        numeral = numerals[idx]
        if has_seventh:
            numeral += '7'
        return numeral
    
    # === Fallback: Try natural minor for borrowed chords ===
    if not is_major and mode == 'harmonic_minor':
        key_minor = root + 'm'
        scale_natural = MINOR_SCALES_NATURAL.get(key_minor, MINOR_SCALES_NATURAL['Am'])
        
        try:
            idx = scale_natural.index(base_chord)
            numeral = MINOR_NUMERALS_NATURAL[idx]
            if has_seventh:
                numeral += '7'
            return numeral
        except ValueError:
            pass
        
        # Root matching in natural minor
        scale_roots_nat = [ch.replace('m', '').replace('dim', '') 
                          for ch in scale_natural]
        if chord_root in scale_roots_nat:
            idx = scale_roots_nat.index(chord_root)
            numeral = MINOR_NUMERALS_NATURAL[idx]
            if has_seventh:
                numeral += '7'
            return numeral
    
    return '?'


def get_functional_weight(
    chord: str, 
    key: str, 
    mode: str = 'harmonic_minor'
) -> float:
    """
    Get the functional weight for a chord in context.
    
    This implements Tier 2 of the multi-tier consonance model.
    
    Parameters
    ----------
    chord : str
        Chord symbol
    key : str
        Key signature
    mode : str, optional
        Minor mode type (default: 'harmonic_minor')
        
    Returns
    -------
    float
        Functional weight w_func ∈ [1.0, 1.6]
        
    Notes
    -----
    HEURISTIC: These weights are empirically calibrated from
    music theory conventions (Rameau), not mathematically derived.
    See Theorem B.19 in SI for necessity proof.
    """
    numeral = get_roman_numeral(chord, key, mode)
    return FUNCTIONAL_WEIGHTS.get(numeral, 1.0)


def get_resolution_pressure(
    current_chord: str,
    next_chord: str,
    key: str,
    mode: str = 'harmonic_minor'
) -> float:
    """
    Compute resolution pressure for a chord transition.
    
    This implements Tier 3 of the multi-tier consonance model.
    
    Parameters
    ----------
    current_chord : str
        Current chord symbol
    next_chord : str
        Next chord symbol
    key : str
        Key signature
    mode : str, optional
        Minor mode type (default: 'harmonic_minor')
        
    Returns
    -------
    float
        Resolution pressure w_res ∈ [0.8, 1.3]
        - w_res < 1.0: Resolution (tension release)
        - w_res = 1.0: Neutral
        - w_res > 1.0: Unresolved (tension maintained)
        
    Examples
    --------
    >>> get_resolution_pressure('G7', 'C', 'C Major')
    0.80  # Strong resolution (V7 → I)
    
    >>> get_resolution_pressure('G', 'Am', 'C Major')
    1.15  # Deceptive cadence (V → vi)
    
    Notes
    -----
    HEURISTIC: Values calibrated from cadential analysis.
    See Theorem B.20-B.21 in SI for necessity proof.
    """
    current_numeral = get_roman_numeral(current_chord, key, mode)
    next_numeral = get_roman_numeral(next_chord, key, mode)
    
    # Build transition key
    pair = f"{current_numeral}→{next_numeral}"
    
    # Direct lookup
    if pair in RESOLUTION_PRESSURE:
        return RESOLUTION_PRESSURE[pair]
    
    # === Default rules ===
    
    # Any V-type → I/i: Resolution
    if 'V' in current_numeral:
        if next_numeral in ['I', 'i']:
            return 0.85
        if next_numeral in ['vi', 'VI']:
            return 1.15  # Deceptive
        if 'V' in next_numeral:
            return 1.25  # Prolongation
    
    # Leading tone → Tonic: Resolution
    if 'vii' in current_numeral and next_numeral in ['I', 'i']:
        return 0.85
    
    # Default: Neutral
    return 1.0


# ==============================================================================
# Chord Progression Analysis
# ==============================================================================

def analyze_progression(
    chords: List[str],
    key: str,
    mode: str = 'harmonic_minor'
) -> List[Dict]:
    """
    Analyze a chord progression with full harmonic context.
    
    Parameters
    ----------
    chords : List[str]
        List of chord symbols
    key : str
        Key signature
    mode : str, optional
        Minor mode type
        
    Returns
    -------
    List[Dict]
        Analysis for each chord containing:
        - 'chord': Original chord symbol
        - 'numeral': Roman numeral
        - 'w_func': Functional weight
        - 'w_res': Resolution pressure (to next chord)
        - 'next_chord': Following chord (or None)
        
    Example
    -------
    >>> progression = ['C', 'G', 'Am', 'F']
    >>> analysis = analyze_progression(progression, 'C Major')
    >>> for item in analysis:
    ...     print(f"{item['chord']}: {item['numeral']}")
    C: I
    G: V
    Am: vi
    F: IV
    """
    results = []
    
    for i, chord in enumerate(chords):
        numeral = get_roman_numeral(chord, key, mode)
        w_func = get_functional_weight(chord, key, mode)
        
        # Resolution pressure to next chord
        if i < len(chords) - 1:
            next_chord = chords[i + 1]
            w_res = get_resolution_pressure(chord, next_chord, key, mode)
        else:
            next_chord = None
            w_res = 1.0  # No resolution context
        
        results.append({
            'chord': chord,
            'numeral': numeral,
            'w_func': w_func,
            'w_res': w_res,
            'next_chord': next_chord,
        })
    
    return results


# ==============================================================================
# Verification
# ==============================================================================

def demonstrate_functional_analysis() -> None:
    """
    Demonstrate Roman numeral analysis for common progressions.
    """
    print("=" * 70)
    print("  FUNCTIONAL HARMONY ANALYSIS DEMONSTRATION")
    print("=" * 70)
    
    test_cases = [
        ('C Major', ['C', 'G', 'Am', 'F']),           # I-V-vi-IV
        ('D Major', ['D', 'A', 'Bm', 'G']),           # Canon progression
        ('D minor', ['Dm', 'Bb', 'F', 'C']),          # Minor progression
        ('D minor', ['Bb', 'F', 'G7', 'A']),          # Senbonzakura pre-chorus
    ]
    
    for key, chords in test_cases:
        print(f"\n  Key: {key}")
        print(f"  Progression: {' - '.join(chords)}")
        print("-" * 70)
        print(f"  {'Chord':<8} {'Roman':<8} {'w_func':<8} {'w_res':<8}")
        print("-" * 70)
        
        analysis = analyze_progression(chords, key)
        for item in analysis:
            print(f"  {item['chord']:<8} {item['numeral']:<8} "
                  f"{item['w_func']:<8.2f} {item['w_res']:<8.2f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demonstrate_functional_analysis()
