"""
HELA: Harmonic Energy Landscape Analyzer
========================================

A framework for interpretable music analysis using energy landscape theory.

Core Concepts:
    - Three-tier harmonic energy (Consonance × Function × Resolution)
    - Lagrangian mechanics for melodic trajectories (L = K - V)
    - Total Action classification (Static vs Dynamic resolution)
    - Catharsis Index for emotional intensity quantification

Quick Start:
    >>> from hela import analyze_trajectory
    >>> 
    >>> pitches = [60, 62, 64, 65, 67]
    >>> durations = [1.0, 1.0, 1.0, 1.0, 1.0]
    >>> chords = ['C', 'C', 'F', 'G', 'C']
    >>> 
    >>> result = analyze_trajectory(pitches, durations, chords, 'C Major')
    >>> print(f"Total Action: {result.total_action:.2f}")
    >>> print(f"Resolution: {result.resolution_type}")

Modules:
    - consonance: Continued fraction analysis, Consonance Tensor
    - harmony: Roman numeral analysis, functional weights
    - energy: Three-tier energy computation
    - trajectory: Lagrangian analysis, Action, Catharsis
    - visualization: Energy landscape plots

References:
    Iizumi, M. (2025). "Beyond Tags: An Energy Landscape Approach to 
    Interpretable Training Data for Music AI"
    
    Iizumi, M. (2025). "On the Consonance of Prime Factorization"
    DOI: 10.5281/zenodo.17920155

    Iizumi, M. (2025). ""The Consonance Tensor"
    DOI: 10.5281/zenodo.17920244

Author: Masamichi Iizumi
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Masamichi Iizumi"
__license__ = "MIT"


# ==============================================================================
# Core Imports
# ==============================================================================

# Consonance (Tier 1)
from .consonance import (
    continued_fraction,
    consonance_degree,
    build_consonance_tensor,
    harmonic_energy,
    CHORD_RATIOS,
)

# Harmony (Tier 2-3)
from .harmony import (
    get_roman_numeral,
    get_functional_weight,
    get_resolution_pressure,
    analyze_progression,
    FUNCTIONAL_WEIGHTS,
    RESOLUTION_PRESSURE,
)

# Energy
from .energy import (
    compute_tier1_energy,
    compute_tier2_energy,
    compute_tier3_energy,
    compute_harmonic_energy,
    compute_melodic_energy,
    compute_total_energy,
    analyze_progression_energy,
    pitch_chord_alignment,
    get_chord_energy,
    CHORD_ENERGIES,
)

# Trajectory (Main Analysis)
from .trajectory import (
    # Data structures
    TrajectoryPoint,
    TrajectoryAnalysis,
    # Main functions
    analyze_trajectory,
    compute_total_action,
    compute_catharsis_index,
    classify_resolution,
    # Utilities
    compute_velocity,
    compute_kinetic_energy,
    compute_lagrangian,
    compute_potential_energy,
    extract_energy_profile,
    find_climax_point,
    generate_summary,
    get_resolution_mode_details,
)

# Visualization
from .visualization import (
    plot_2d_energy_landscape,
    plot_energy_profile,
    plot_three_tier_comparison,
    plot_multi_piece_comparison,
    plot_summary_comparison,
    plot_resolution_diagram,
    setup_style,
    COLORS,
    PIECE_COLORS,
)


# ==============================================================================
# Public API
# ==============================================================================

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    
    # === Consonance (Tier 1) ===
    "continued_fraction",
    "consonance_degree",
    "build_consonance_tensor",
    "harmonic_energy",
    "CHORD_RATIOS",
    
    # === Harmony (Tier 2-3) ===
    "get_roman_numeral",
    "get_functional_weight",
    "get_resolution_pressure",
    "analyze_progression",
    "FUNCTIONAL_WEIGHTS",
    "RESOLUTION_PRESSURE",
    
    # === Energy ===
    "compute_tier1_energy",
    "compute_tier2_energy",
    "compute_tier3_energy",
    "compute_harmonic_energy",
    "compute_melodic_energy",
    "compute_total_energy",
    "analyze_progression_energy",
    "pitch_chord_alignment",
    "get_chord_energy",
    "CHORD_ENERGIES",
    
    # === Trajectory (Main) ===
    "TrajectoryPoint",
    "TrajectoryAnalysis",
    "analyze_trajectory",
    "compute_total_action",
    "compute_catharsis_index",
    "classify_resolution",
    "compute_velocity",
    "compute_kinetic_energy",
    "compute_lagrangian",
    "compute_potential_energy",
    "extract_energy_profile",
    "find_climax_point",
    "generate_summary",
    "get_resolution_mode_details",
    
    # === Visualization ===
    "plot_2d_energy_landscape",
    "plot_energy_profile",
    "plot_three_tier_comparison",
    "plot_multi_piece_comparison",
    "plot_summary_comparison",
    "plot_resolution_diagram",
    "setup_style",
    "COLORS",
    "PIECE_COLORS",
]


# ==============================================================================
# Convenience Functions
# ==============================================================================

def quick_analysis(
    pitches: list,
    durations: list,
    chords: list,
    key: str,
    tempo: float = 120.0,
    title: str = "",
    verbose: bool = True,
) -> TrajectoryAnalysis:
    """
    Quick one-line analysis with optional summary output.
    
    Parameters
    ----------
    pitches : list
        Melody as MIDI pitch sequence
    durations : list
        Note durations in beats
    chords : list
        Chord sequence
    key : str
        Key signature (e.g., 'C Major', 'D minor')
    tempo : float, optional
        Tempo in BPM (default: 120)
    title : str, optional
        Piece title for summary
    verbose : bool, optional
        Print summary to console (default: True)
        
    Returns
    -------
    TrajectoryAnalysis
        Complete analysis result
        
    Examples
    --------
    >>> from hela import quick_analysis
    >>> result = quick_analysis(
    ...     pitches=[60, 62, 64, 65, 67],
    ...     durations=[1.0, 1.0, 1.0, 1.0, 1.0],
    ...     chords=['C', 'C', 'F', 'G', 'C'],
    ...     key='C Major',
    ...     title='Simple Scale'
    ... )
    """
    analysis = analyze_trajectory(
        pitches=pitches,
        durations=durations,
        chords=chords,
        key=key,
        tempo=tempo,
    )
    
    if verbose:
        print(generate_summary(analysis, title))
    
    return analysis


def compare_pieces(
    pieces: dict,
    verbose: bool = True,
) -> dict:
    """
    Compare multiple pieces and return analyses.
    
    Parameters
    ----------
    pieces : dict
        Dictionary with structure:
        {
            'Piece Name': {
                'pitches': [...],
                'durations': [...],
                'chords': [...],
                'key': 'X Major/minor',
                'tempo': 120.0,  # optional
            },
            ...
        }
    verbose : bool, optional
        Print comparison table (default: True)
        
    Returns
    -------
    dict
        Dictionary mapping piece names to TrajectoryAnalysis objects
        
    Examples
    --------
    >>> from hela import compare_pieces
    >>> pieces = {
    ...     'Canon': {'pitches': [...], 'durations': [...], ...},
    ...     'Let It Be': {'pitches': [...], 'durations': [...], ...},
    ... }
    >>> analyses = compare_pieces(pieces)
    """
    analyses = {}
    
    for name, data in pieces.items():
        tempo = data.get('tempo', 120.0)
        analyses[name] = analyze_trajectory(
            pitches=data['pitches'],
            durations=data['durations'],
            chords=data['chords'],
            key=data['key'],
            tempo=tempo,
        )
    
    if verbose:
        print("=" * 70)
        print("  MULTI-PIECE COMPARISON")
        print("=" * 70)
        print()
        print(f"  {'Piece':<20} {'Action (S)':<12} {'Catharsis':<12} {'Type':<10}")
        print("-" * 70)
        
        for name, analysis in analyses.items():
            print(f"  {name:<20} {analysis.total_action:<12.2f} "
                  f"{analysis.catharsis_index:<12.2f} {analysis.resolution_type:<10}")
        
        print()
        print("=" * 70)
    
    return analyses


# Add to __all__
__all__.extend([
    "quick_analysis",
    "compare_pieces",
])


# ==============================================================================
# Package Info
# ==============================================================================

def info():
    """Print package information."""
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  HELA: Harmonic Energy Landscape Analyzer                        ║
║  Version: {__version__:<54} ║
║  Author: {__author__:<55} ║
║  License: {__license__:<54} ║
╠══════════════════════════════════════════════════════════════════╣
║  Core Functions:                                                 ║
║    • analyze_trajectory()  - Full Lagrangian analysis            ║
║    • quick_analysis()      - One-line convenience function       ║
║    • compare_pieces()      - Multi-piece comparison              ║
║                                                                  ║
║  Energy Functions:                                               ║
║    • compute_tier1_energy() - Base consonance (λ_max)            ║
║    • compute_tier2_energy() - + Functional weight                ║
║    • compute_tier3_energy() - + Resolution pressure              ║
║                                                                  ║
║  Visualization:                                                  ║
║    • plot_2d_energy_landscape() - Time×Pitch heatmap             ║
║    • plot_energy_profile()      - K, V, L time series            ║
║    • plot_three_tier_comparison() - Tier comparison bars         ║
╚══════════════════════════════════════════════════════════════════╝
    """)
