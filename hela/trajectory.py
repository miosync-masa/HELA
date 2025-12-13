"""
HELA: Trajectory Analysis Module
================================

Lagrangian mechanics for melodic trajectories.

Core Concepts:
    - Lagrangian: L = K - V (kinetic minus potential)
    - Total Action: S = Σ L(t) × Δt
    - Catharsis Index: C = |ΔV| × |ΔK| / V_init

Resolution Classification:
    - Static Resolution: S < 0 (stability-dominated)
    - Dynamic Resolution: S > 0 (motion-dominated)

See paper Section 4 for theoretical foundations.

Author: Masamichi Iizumi
License: MIT
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .consonance import harmonic_energy
from .harmony import get_roman_numeral, get_resolution_pressure
from .energy import (
    compute_harmonic_energy,
    compute_melodic_energy,
    pitch_chord_alignment,
    get_chord_energy,
    CHORD_ENERGIES,
)


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class TrajectoryPoint:
    """Single point in a melodic trajectory."""
    position: int           # Index in sequence
    time: float            # Cumulative time (beats)
    pitch: int             # MIDI pitch
    duration: float        # Note duration (beats)
    chord: str             # Current chord
    numeral: str           # Roman numeral
    velocity: float        # Pitch velocity (Δpitch/Δt)
    K: float               # Kinetic energy
    V: float               # Potential energy (harmonic)
    L: float               # Lagrangian (K - V)
    action_contrib: float  # L × duration


@dataclass  
class TrajectoryAnalysis:
    """Complete trajectory analysis results."""
    points: List[TrajectoryPoint]
    total_action: float
    mean_K: float
    mean_V: float
    mean_L: float
    max_V: float
    min_V: float
    energy_range: float
    dramatic_arc: float
    catharsis_index: float
    resolution_type: str   # 'static' or 'dynamic'
    conversion_efficiency: float  # η = |ΔK| / |ΔV|


# ==============================================================================
# Velocity and Kinetic Energy
# ==============================================================================

def compute_velocity(
    pitch_current: int,
    pitch_next: int,
    duration: float,
    tempo: float = 120.0
) -> float:
    """
    Compute melodic velocity (pitch change rate).
    
    v = Δpitch / Δt
    
    Parameters
    ----------
    pitch_current : int
        Current MIDI pitch
    pitch_next : int
        Next MIDI pitch
    duration : float
        Note duration in beats
    tempo : float, optional
        Tempo in BPM (default: 120)
        
    Returns
    -------
    float
        Velocity in semitones per second
    """
    dt = (60.0 / tempo) * duration
    if dt > 0:
        return (pitch_next - pitch_current) / dt
    return 0.0


def compute_kinetic_energy(velocity: float, mass: float = 1.0) -> float:
    """
    Compute kinetic energy from velocity.
    
    K = ½mv²
    
    Parameters
    ----------
    velocity : float
        Pitch velocity
    mass : float, optional
        Mass parameter (default: 1.0)
        
    Returns
    -------
    float
        Kinetic energy
    """
    return 0.5 * mass * velocity ** 2


# ==============================================================================
# Lagrangian Computation
# ==============================================================================

def compute_lagrangian(K: float, V: float) -> float:
    """
    Compute Lagrangian.
    
    L = K - V
    
    Parameters
    ----------
    K : float
        Kinetic energy (melodic motion)
    V : float
        Potential energy (harmonic stability)
        
    Returns
    -------
    float
        Lagrangian value
        
    Notes
    -----
    Sign interpretation:
        L < 0: Stability dominates (calm, resolved)
        L > 0: Motion dominates (energetic, tense)
    """
    return K - V


def compute_potential_energy(
    pitch: int,
    chord: str,
    key: str,
    next_chord: Optional[str] = None,
    tier: int = 3,
    mode: str = 'harmonic_minor',
    use_alignment: bool = True
) -> float:
    """
    Compute potential energy (harmonic stability term).
    
    V = E_harmonic × f_align
    
    Parameters
    ----------
    pitch : int
        MIDI pitch
    chord : str
        Current chord
    key : str
        Key signature
    next_chord : str or None, optional
        Next chord (for Tier 3)
    tier : int, optional
        Energy tier (1, 2, or 3)
    mode : str, optional
        Minor mode type
    use_alignment : bool, optional
        Apply pitch-chord alignment factor
        
    Returns
    -------
    float
        Potential energy V
    """
    E_h = compute_harmonic_energy(chord, key, next_chord, tier, mode)
    
    if use_alignment:
        f_align = pitch_chord_alignment(pitch, chord)
        E_h *= f_align
    
    return E_h


# ==============================================================================
# Trajectory Analysis
# ==============================================================================

def analyze_trajectory(
    pitches: List[int],
    durations: List[float],
    chords: List[str],
    key: str,
    tempo: float = 120.0,
    mass: float = 1.0,
    tier: int = 3,
    mode: str = 'harmonic_minor'
) -> TrajectoryAnalysis:
    """
    Perform complete trajectory analysis.
    
    This is the main analysis function for the HELA framework.
    
    Parameters
    ----------
    pitches : List[int]
        Melody as MIDI pitch sequence
    durations : List[float]
        Note durations in beats
    chords : List[str]
        Chord sequence (same length as pitches)
    key : str
        Key signature
    tempo : float, optional
        Tempo in BPM
    mass : float, optional
        Mass parameter for kinetic energy
    tier : int, optional
        Harmonic energy tier (1, 2, or 3)
    mode : str, optional
        Minor mode type
        
    Returns
    -------
    TrajectoryAnalysis
        Complete analysis including:
        - Point-by-point trajectory data
        - Total Action S
        - Energy statistics
        - Catharsis Index C
        - Resolution classification
        
    Examples
    --------
    >>> pitches = [60, 62, 64, 65, 67]
    >>> durations = [1.0, 1.0, 1.0, 1.0, 1.0]
    >>> chords = ['C', 'C', 'F', 'F', 'G']
    >>> result = analyze_trajectory(pitches, durations, chords, 'C Major')
    >>> print(f"Total Action: {result.total_action:.2f}")
    >>> print(f"Resolution: {result.resolution_type}")
    """
    n = len(pitches)
    points: List[TrajectoryPoint] = []
    cumulative_time = 0.0
    
    # Track energy changes for catharsis calculation
    V_values = []
    K_values = []
    
    for i in range(n):
        pitch = pitches[i]
        duration = durations[i]
        chord = chords[i] if i < len(chords) else chords[-1]
        next_chord = chords[i + 1] if i < len(chords) - 1 else None
        
        # Roman numeral
        numeral = get_roman_numeral(chord, key, mode)
        
        # Velocity and Kinetic energy
        if i < n - 1:
            velocity = compute_velocity(pitch, pitches[i + 1], duration, tempo)
        else:
            velocity = 0.0
        
        K = compute_kinetic_energy(velocity, mass)
        
        # Potential energy (with resolution pressure for Tier 3)
        V = compute_potential_energy(
            pitch, chord, key, next_chord, tier, mode
        )
        
        # Lagrangian
        L = compute_lagrangian(K, V)
        
        # Action contribution
        action_contrib = L * duration
        
        # Store point
        point = TrajectoryPoint(
            position=i,
            time=cumulative_time,
            pitch=pitch,
            duration=duration,
            chord=chord,
            numeral=numeral,
            velocity=velocity,
            K=K,
            V=V,
            L=L,
            action_contrib=action_contrib
        )
        points.append(point)
        
        V_values.append(V)
        K_values.append(K)
        cumulative_time += duration
    
    # === Compute Statistics ===
    total_duration = sum(durations)
    
    # Duration-weighted means
    mean_K = sum(p.K * p.duration for p in points) / total_duration
    mean_V = sum(p.V * p.duration for p in points) / total_duration
    mean_L = sum(p.L * p.duration for p in points) / total_duration
    
    # Energy range
    max_V = max(V_values)
    min_V = min(V_values)
    energy_range = max_V - min_V
    
    # Dramatic arc
    dramatic_arc = energy_range / mean_V if mean_V > 0 else 0.0
    
    # Total Action
    total_action = sum(p.action_contrib for p in points)
    
    # === Catharsis Analysis ===
    catharsis_index, conversion_efficiency = compute_catharsis_metrics(
        V_values, K_values
    )
    
    # Resolution classification
    resolution_type = classify_resolution(total_action, conversion_efficiency)
    
    return TrajectoryAnalysis(
        points=points,
        total_action=total_action,
        mean_K=mean_K,
        mean_V=mean_V,
        mean_L=mean_L,
        max_V=max_V,
        min_V=min_V,
        energy_range=energy_range,
        dramatic_arc=dramatic_arc,
        catharsis_index=catharsis_index,
        resolution_type=resolution_type,
        conversion_efficiency=conversion_efficiency
    )


# ==============================================================================
# Total Action
# ==============================================================================

def compute_total_action(
    pitches: List[int],
    durations: List[float],
    chords: List[str],
    key: str,
    tempo: float = 120.0,
    mass: float = 1.0,
    tier: int = 3,
    mode: str = 'harmonic_minor'
) -> float:
    """
    Compute Total Action S for a trajectory.
    
    S = Σ L(t) × Δt = Σ (K - V) × Δt
    
    Parameters
    ----------
    pitches : List[int]
        Melody pitches
    durations : List[float]
        Note durations
    chords : List[str]
        Chord sequence
    key : str
        Key signature
    tempo : float, optional
        Tempo in BPM
    mass : float, optional
        Mass parameter
    tier : int, optional
        Energy tier
    mode : str, optional
        Minor mode type
        
    Returns
    -------
    float
        Total Action S
        
    Notes
    -----
    Sign interpretation (Theorem 4.10):
        S < 0: Static resolution (stability-dominated)
        S > 0: Dynamic resolution (motion-dominated)
    """
    analysis = analyze_trajectory(
        pitches, durations, chords, key, tempo, mass, tier, mode
    )
    return analysis.total_action


# ==============================================================================
# Catharsis Metrics
# ==============================================================================

def compute_catharsis_metrics(
    V_values: List[float],
    K_values: List[float]
) -> Tuple[float, float]:
    """
    Compute Catharsis Index and Conversion Efficiency.
    
    Catharsis Index:
        C = |ΔV| × |ΔK| / V_initial
        
    Conversion Efficiency:
        η = |ΔK| / |ΔV|
    
    Parameters
    ----------
    V_values : List[float]
        Potential energy sequence
    K_values : List[float]
        Kinetic energy sequence
        
    Returns
    -------
    Tuple[float, float]
        (catharsis_index, conversion_efficiency)
        
    Notes
    -----
    Catharsis interpretation:
        C < 1: Low catharsis (static, gentle)
        1 ≤ C < 5: Moderate catharsis
        5 ≤ C < 20: High catharsis (dramatic)
        C ≥ 20: Extreme catharsis (epic)
    """
    if len(V_values) < 2:
        return 0.0, 0.0
    
    # Find peak tension
    V_max = max(V_values)
    V_max_idx = V_values.index(V_max)
    
    # Find minimum after peak (resolution point)
    V_after_peak = V_values[V_max_idx:]
    V_min_after = min(V_after_peak) if V_after_peak else V_values[-1]
    
    # Energy changes
    delta_V = V_max - V_min_after  # Tension drop
    
    # Kinetic energy change (from peak to resolution)
    K_at_peak = K_values[V_max_idx] if V_max_idx < len(K_values) else 0.0
    K_max = max(K_values[V_max_idx:]) if V_max_idx < len(K_values) else 0.0
    delta_K = K_max - K_at_peak
    
    # Catharsis Index
    V_initial = V_values[0] if V_values[0] > 0 else 1.0
    catharsis_index = abs(delta_V) * abs(delta_K) / V_initial
    
    # Conversion Efficiency
    if abs(delta_V) > 0:
        conversion_efficiency = abs(delta_K) / abs(delta_V)
    else:
        conversion_efficiency = 0.0
    
    return catharsis_index, conversion_efficiency


def compute_catharsis_index(
    pitches: List[int],
    durations: List[float],
    chords: List[str],
    key: str,
    tempo: float = 120.0,
    mass: float = 1.0,
    tier: int = 3,
    mode: str = 'harmonic_minor'
) -> float:
    """
    Compute Catharsis Index for a piece.
    
    Convenience function wrapping analyze_trajectory().
    
    Parameters
    ----------
    (same as analyze_trajectory)
        
    Returns
    -------
    float
        Catharsis Index C
    """
    analysis = analyze_trajectory(
        pitches, durations, chords, key, tempo, mass, tier, mode
    )
    return analysis.catharsis_index


# ==============================================================================
# Resolution Classification
# ==============================================================================

def classify_resolution(
    total_action: float,
    conversion_efficiency: float,
    action_threshold: float = 0.0,
    efficiency_threshold: float = 0.3
) -> str:
    """
    Classify resolution type based on Action and Conversion Efficiency.
    
    Parameters
    ----------
    total_action : float
        Total Action S
    conversion_efficiency : float
        Conversion efficiency η
    action_threshold : float, optional
        Threshold for S (default: 0)
    efficiency_threshold : float, optional
        Threshold for η (default: 0.3)
        
    Returns
    -------
    str
        'static' or 'dynamic'
        
    Notes
    -----
    Classification rules (Theorem 4.10):
        - Static: S < 0 (stability dominates)
        - Dynamic: S > 0 (motion dominates)
        
    Additional criterion using η:
        - Static: η < 0.3 (minimal energy conversion)
        - Dynamic: η ≥ 0.3 (substantial conversion)
    """
    # Primary criterion: Action sign
    if total_action < action_threshold:
        return 'static'
    else:
        return 'dynamic'


def get_resolution_mode_details(analysis: TrajectoryAnalysis) -> Dict[str, any]:
    """
    Get detailed resolution mode analysis.
    
    Parameters
    ----------
    analysis : TrajectoryAnalysis
        Result from analyze_trajectory()
        
    Returns
    -------
    Dict
        Detailed classification including:
        - type: 'static' or 'dynamic'
        - confidence: Classification confidence
        - characteristics: List of observed characteristics
    """
    S = analysis.total_action
    C = analysis.catharsis_index
    eta = analysis.conversion_efficiency
    arc = analysis.dramatic_arc
    
    characteristics = []
    
    # Analyze characteristics
    if S < -50:
        characteristics.append('strongly stability-dominated')
    elif S < 0:
        characteristics.append('mildly stability-dominated')
    elif S < 50:
        characteristics.append('mildly motion-dominated')
    else:
        characteristics.append('strongly motion-dominated')
    
    if C < 1:
        characteristics.append('low catharsis (gentle)')
    elif C < 5:
        characteristics.append('moderate catharsis')
    elif C < 20:
        characteristics.append('high catharsis (dramatic)')
    else:
        characteristics.append('extreme catharsis (epic)')
    
    if eta < 0.2:
        characteristics.append('minimal energy conversion')
    elif eta < 0.5:
        characteristics.append('partial energy conversion')
    else:
        characteristics.append('substantial energy conversion')
    
    if arc < 0.3:
        characteristics.append('low dramatic contrast')
    elif arc < 0.7:
        characteristics.append('moderate dramatic contrast')
    else:
        characteristics.append('high dramatic contrast')
    
    # Confidence based on consistency of indicators
    indicators = [
        S < 0,  # Static by action
        C < 1,  # Static by catharsis
        eta < 0.3,  # Static by conversion
    ]
    static_count = sum(indicators)
    
    if static_count >= 2:
        confidence = 'high' if static_count == 3 else 'medium'
        resolution_type = 'static'
    elif static_count <= 1:
        confidence = 'high' if static_count == 0 else 'medium'
        resolution_type = 'dynamic'
    else:
        confidence = 'low'
        resolution_type = 'ambiguous'
    
    return {
        'type': resolution_type,
        'confidence': confidence,
        'characteristics': characteristics,
        'metrics': {
            'total_action': S,
            'catharsis_index': C,
            'conversion_efficiency': eta,
            'dramatic_arc': arc,
        }
    }


# ==============================================================================
# Energy Profile Extraction
# ==============================================================================

def extract_energy_profile(
    analysis: TrajectoryAnalysis
) -> Dict[str, List[float]]:
    """
    Extract energy time series from trajectory analysis.
    
    Parameters
    ----------
    analysis : TrajectoryAnalysis
        Result from analyze_trajectory()
        
    Returns
    -------
    Dict[str, List[float]]
        Energy profiles:
        - 'time': Time points
        - 'K': Kinetic energy series
        - 'V': Potential energy series
        - 'L': Lagrangian series
        - 'E_total': Total energy series
    """
    return {
        'time': [p.time for p in analysis.points],
        'K': [p.K for p in analysis.points],
        'V': [p.V for p in analysis.points],
        'L': [p.L for p in analysis.points],
        'E_total': [p.K + p.V for p in analysis.points],
    }


def find_climax_point(analysis: TrajectoryAnalysis) -> Dict[str, any]:
    """
    Find the climax (maximum tension) point in trajectory.
    
    Parameters
    ----------
    analysis : TrajectoryAnalysis
        Result from analyze_trajectory()
        
    Returns
    -------
    Dict
        Climax information:
        - position: Index of climax
        - time: Time of climax
        - V: Potential energy at climax
        - chord: Chord at climax
        - numeral: Roman numeral at climax
    """
    V_max = analysis.max_V
    
    for p in analysis.points:
        if p.V == V_max:
            return {
                'position': p.position,
                'time': p.time,
                'V': p.V,
                'K': p.K,
                'chord': p.chord,
                'numeral': p.numeral,
            }
    
    return None


# ==============================================================================
# Summary Generation
# ==============================================================================

def generate_summary(analysis: TrajectoryAnalysis, title: str = '') -> str:
    """
    Generate human-readable summary of trajectory analysis.
    
    Parameters
    ----------
    analysis : TrajectoryAnalysis
        Result from analyze_trajectory()
    title : str, optional
        Piece title
        
    Returns
    -------
    str
        Formatted summary string
    """
    lines = []
    lines.append("=" * 60)
    if title:
        lines.append(f"  TRAJECTORY ANALYSIS: {title}")
    else:
        lines.append("  TRAJECTORY ANALYSIS")
    lines.append("=" * 60)
    lines.append("")
    
    # Key metrics
    lines.append("  KEY METRICS")
    lines.append("-" * 60)
    lines.append(f"  Total Action (S):         {analysis.total_action:>10.2f}")
    lines.append(f"  Catharsis Index (C):      {analysis.catharsis_index:>10.2f}")
    lines.append(f"  Conversion Efficiency (η):{analysis.conversion_efficiency:>10.2f}")
    lines.append(f"  Dramatic Arc:             {analysis.dramatic_arc:>10.2f}")
    lines.append("")
    
    # Energy statistics
    lines.append("  ENERGY STATISTICS")
    lines.append("-" * 60)
    lines.append(f"  Mean K (kinetic):         {analysis.mean_K:>10.2f}")
    lines.append(f"  Mean V (potential):       {analysis.mean_V:>10.2f}")
    lines.append(f"  Mean L (Lagrangian):      {analysis.mean_L:>10.2f}")
    lines.append(f"  Max V (peak tension):     {analysis.max_V:>10.2f}")
    lines.append(f"  Min V (resolution):       {analysis.min_V:>10.2f}")
    lines.append(f"  Energy Range:             {analysis.energy_range:>10.2f}")
    lines.append("")
    
    # Classification
    lines.append("  CLASSIFICATION")
    lines.append("-" * 60)
    lines.append(f"  Resolution Type:          {analysis.resolution_type.upper()}")
    
    if analysis.resolution_type == 'static':
        lines.append("  → Stability-dominated (calm, resolved)")
    else:
        lines.append("  → Motion-dominated (energetic, climactic)")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


# ==============================================================================
# Demonstration
# ==============================================================================

def demonstrate_trajectory_analysis() -> None:
    """
    Demonstrate trajectory analysis on sample data.
    """
    print("=" * 60)
    print("  TRAJECTORY ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Sample: Simple ascending scale over I-IV-V-I
    pitches = [60, 62, 64, 65, 67, 65, 64, 60]
    durations = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]
    chords = ['C', 'C', 'F', 'F', 'G', 'G', 'C', 'C']
    key = 'C Major'
    
    print(f"\n  Melody: C-D-E-F-G-F-E-C")
    print(f"  Chords: {' - '.join(chords)}")
    print(f"  Key: {key}")
    print()
    
    # Analyze
    analysis = analyze_trajectory(pitches, durations, chords, key)
    
    # Print summary
    print(generate_summary(analysis, "Simple Scale"))
    
    # Point-by-point
    print("\n  POINT-BY-POINT ANALYSIS")
    print("-" * 60)
    print(f"  {'Pos':<4} {'Pitch':<6} {'Chord':<6} {'K':<8} {'V':<8} {'L':<8}")
    print("-" * 60)
    
    for p in analysis.points:
        print(f"  {p.position:<4} {p.pitch:<6} {p.chord:<6} "
              f"{p.K:<8.2f} {p.V:<8.2f} {p.L:<8.2f}")
    
    print()
    
    # Find climax
    climax = find_climax_point(analysis)
    if climax:
        print(f"  Climax at position {climax['position']}: "
              f"chord={climax['chord']} ({climax['numeral']}), V={climax['V']:.2f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demonstrate_trajectory_analysis()
