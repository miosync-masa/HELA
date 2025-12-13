"""
HELA: Visualization Module
==========================

Energy landscape visualization for melodic trajectories.

Visualization Types:
    1. 2D Energy Landscape (Time × Pitch heatmap)
    2. Energy Profile (Time series)
    3. Three-Tier Comparison
    4. Multi-piece Comparison
    5. Lagrangian Trajectory

Author: Masamichi Iizumi
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Tuple, Union

from .trajectory import TrajectoryAnalysis, TrajectoryPoint
from .energy import (
    pitch_chord_alignment,
    get_chord_energy,
    compute_tier1_energy,
    compute_tier2_energy,
    compute_tier3_energy,
    CHORD_ENERGIES,
)
from .harmony import get_roman_numeral


# ==============================================================================
# Style Configuration
# ==============================================================================

# Color palette
COLORS = {
    'tier1': '#3498DB',      # Blue
    'tier2': '#E74C3C',      # Red
    'tier3': '#9B59B6',      # Purple
    'lagrangian': '#27AE60', # Green
    'melody': '#FF1744',     # Bright red
    'climax': '#FFD700',     # Gold
    'static': '#2ECC71',     # Green (calm)
    'dynamic': '#E74C3C',    # Red (energetic)
}

# Piece-specific colors
PIECE_COLORS = {
    'Canon': '#3498DB',
    'Let It Be': '#27AE60',
    'Senbonzakura': '#E74C3C',
}

# MIDI pitch to note name
PITCH_NAMES = {
    60: 'C4', 61: 'C#4', 62: 'D4', 63: 'D#4', 64: 'E4', 65: 'F4',
    66: 'F#4', 67: 'G4', 68: 'G#4', 69: 'A4', 70: 'A#4', 71: 'B4',
    72: 'C5', 73: 'C#5', 74: 'D5', 75: 'D#5', 76: 'E5', 77: 'F5',
}

# Simplified pitch labels
PITCH_LABELS_SIMPLE = {
    60: 'C', 62: 'D', 64: 'E', 65: 'F', 67: 'G', 69: 'A', 71: 'B', 72: 'C'
}


def setup_style():
    """Configure matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


# ==============================================================================
# 2D Energy Landscape
# ==============================================================================

def plot_2d_energy_landscape(
    analysis: TrajectoryAnalysis,
    title: str = '',
    pitch_range: Tuple[int, int] = (60, 73),
    cmap: str = 'RdYlGn_r',
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None,
    show_climax: bool = True,
    show_chord_boundaries: bool = True,
    show_chord_labels: bool = True,
) -> plt.Figure:
    """
    Plot 2D energy landscape with melody trajectory overlay.
    
    This creates a Time × Pitch heatmap showing the energy landscape,
    with the actual melody path drawn on top.
    
    Parameters
    ----------
    analysis : TrajectoryAnalysis
        Result from analyze_trajectory()
    title : str, optional
        Plot title
    pitch_range : Tuple[int, int], optional
        MIDI pitch range (min, max). Default: (60, 73) = C4 to C5
    cmap : str, optional
        Colormap name. Default: 'RdYlGn_r' (red=unstable, green=stable)
    figsize : Tuple[int, int], optional
        Figure size
    save_path : str or None, optional
        Path to save figure (None = don't save)
    show_climax : bool, optional
        Highlight climax point with gold star
    show_chord_boundaries : bool, optional
        Show vertical lines at chord changes
    show_chord_labels : bool, optional
        Show chord names at top
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
        
    Notes
    -----
    The heatmap shows V(pitch, chord) for each time-pitch combination.
    Darker colors indicate more stable (lower energy) regions.
    The melody trajectory shows how the actual melody navigates
    through this energy landscape.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    points = analysis.points
    n_points = len(points)
    
    # === 1. Build energy landscape grid ===
    times = np.arange(n_points)
    pitches = np.arange(pitch_range[0], pitch_range[1])
    
    T, P = np.meshgrid(times, pitches)
    Z = np.zeros_like(T, dtype=float)
    
    for i, t in enumerate(times):
        chord = points[t].chord
        base_energy = get_chord_energy(chord)
        
        for j, pitch in enumerate(pitches):
            f = pitch_chord_alignment(pitch, chord)
            # Negative for visualization (darker = more stable)
            Z[j, i] = -base_energy * f
    
    # === 2. Plot heatmap ===
    im = ax.imshow(
        Z, 
        aspect='auto', 
        origin='lower',
        extent=[0, n_points - 1, pitch_range[0], pitch_range[1] - 1],
        cmap=cmap, 
        alpha=0.7
    )
    
    # === 3. Plot melody trajectory ===
    melody_times = [p.position for p in points]
    melody_pitches = [p.pitch for p in points]
    
    ax.plot(
        melody_times, melody_pitches,
        'o-', 
        linewidth=4, 
        markersize=12,
        color=COLORS['melody'], 
        markeredgecolor='black',
        markeredgewidth=2, 
        label='Melody',
        zorder=50
    )
    
    # === 4. Chord boundaries ===
    if show_chord_boundaries:
        for i in range(1, n_points):
            if points[i].chord != points[i - 1].chord:
                ax.axvline(
                    i - 0.5, 
                    color='white', 
                    linestyle='--',
                    linewidth=2, 
                    alpha=0.6
                )
    
    # === 5. Climax marker ===
    if show_climax:
        # Find minimum Lagrangian (maximum stability point)
        min_L_idx = min(range(n_points), key=lambda i: points[i].L)
        ax.plot(
            points[min_L_idx].position,
            points[min_L_idx].pitch,
            marker='*', 
            markersize=35, 
            color=COLORS['climax'],
            markeredgecolor='red', 
            markeredgewidth=3,
            zorder=100,
            label='Energy Minimum'
        )
        
        # Also mark maximum tension
        max_V_idx = max(range(n_points), key=lambda i: points[i].V)
        ax.plot(
            points[max_V_idx].position,
            points[max_V_idx].pitch,
            marker='X', 
            markersize=20, 
            color='red',
            markeredgecolor='white', 
            markeredgewidth=2,
            zorder=99,
            label='Peak Tension'
        )
    
    # === 6. Chord labels ===
    if show_chord_labels:
        for i, p in enumerate(points):
            if i == 0 or points[i].chord != points[i - 1].chord:
                ax.text(
                    i, pitch_range[1] - 0.5, 
                    p.chord,
                    fontsize=11, 
                    ha='center',
                    fontweight='bold',
                    bbox=dict(
                        boxstyle='round',
                        facecolor='white', 
                        alpha=0.8
                    )
                )
    
    # === 7. Formatting ===
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Energy (darker = more stable)', fontsize=12)
    
    ax.set_xlabel('Note Position', fontsize=14)
    ax.set_ylabel('Pitch', fontsize=14)
    
    # Y-axis pitch labels
    yticks = [p for p in pitches if p in PITCH_LABELS_SIMPLE]
    ax.set_yticks(yticks)
    ax.set_yticklabels([PITCH_LABELS_SIMPLE[p] for p in yticks])
    
    if title:
        ax.set_title(
            f'{title}\nMelody Trajectory Through Energy Landscape',
            fontsize=16, 
            weight='bold'
        )
    else:
        ax.set_title(
            'Melody Trajectory Through Energy Landscape',
            fontsize=16, 
            weight='bold'
        )
    
    ax.legend(fontsize=11, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ==============================================================================
# Energy Profile (Time Series)
# ==============================================================================

def plot_energy_profile(
    analysis: TrajectoryAnalysis,
    title: str = '',
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
    show_components: bool = True,
) -> plt.Figure:
    """
    Plot energy components over time.
    
    Shows K (kinetic), V (potential), and L (Lagrangian) as time series.
    
    Parameters
    ----------
    analysis : TrajectoryAnalysis
        Result from analyze_trajectory()
    title : str, optional
        Plot title
    figsize : Tuple[int, int], optional
        Figure size
    save_path : str or None, optional
        Path to save figure
    show_components : bool, optional
        Show K and V separately (True) or just L (False)
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    setup_style()
    
    points = analysis.points
    times = [p.time for p in points]
    
    if show_components:
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Kinetic Energy
        K_values = [p.K for p in points]
        axes[0].plot(times, K_values, 'o-', color=COLORS['tier1'], linewidth=2)
        axes[0].fill_between(times, K_values, alpha=0.3, color=COLORS['tier1'])
        axes[0].set_ylabel('K (Kinetic)', fontsize=12)
        axes[0].set_title('Kinetic Energy (Melodic Motion)', fontsize=13, fontweight='bold')
        
        # Potential Energy
        V_values = [p.V for p in points]
        axes[1].plot(times, V_values, 'o-', color=COLORS['tier2'], linewidth=2)
        axes[1].fill_between(times, V_values, alpha=0.3, color=COLORS['tier2'])
        axes[1].set_ylabel('V (Potential)', fontsize=12)
        axes[1].set_title('Potential Energy (Harmonic Tension)', fontsize=13, fontweight='bold')
        
        # Add chord labels
        for i, p in enumerate(points):
            if i == 0 or points[i].chord != points[i - 1].chord:
                axes[1].annotate(
                    p.chord, 
                    (p.time, p.V),
                    textcoords='offset points',
                    xytext=(0, 10),
                    ha='center',
                    fontsize=9,
                    fontweight='bold',
                    color='red'
                )
        
        # Lagrangian
        L_values = [p.L for p in points]
        axes[2].plot(times, L_values, 'o-', color=COLORS['lagrangian'], linewidth=2)
        axes[2].fill_between(times, L_values, alpha=0.3, color=COLORS['lagrangian'])
        axes[2].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[2].set_ylabel('L = K - V', fontsize=12)
        axes[2].set_xlabel('Time (beats)', fontsize=12)
        axes[2].set_title('Lagrangian', fontsize=13, fontweight='bold')
        
    else:
        fig, ax = plt.subplots(figsize=figsize)
        
        L_values = [p.L for p in points]
        ax.plot(times, L_values, 'o-', color=COLORS['lagrangian'], linewidth=2)
        ax.fill_between(times, L_values, alpha=0.3, color=COLORS['lagrangian'])
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_ylabel('L = K - V', fontsize=12)
        ax.set_xlabel('Time (beats)', fontsize=12)
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ==============================================================================
# Three-Tier Comparison
# ==============================================================================

def plot_three_tier_comparison(
    chords: List[str],
    key: str,
    mode: str = 'harmonic_minor',
    title: str = '',
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare energy across three tiers for a chord progression.
    
    Parameters
    ----------
    chords : List[str]
        Chord sequence
    key : str
        Key signature
    mode : str, optional
        Minor mode type
    title : str, optional
        Plot title
    figsize : Tuple[int, int], optional
        Figure size
    save_path : str or None, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    setup_style()
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    n = len(chords)
    x = np.arange(n)
    
    # Compute energies
    E_tier1 = [compute_tier1_energy(ch) for ch in chords]
    E_tier2 = [compute_tier2_energy(ch, key, mode) for ch in chords]
    E_tier3 = []
    for i, ch in enumerate(chords):
        next_ch = chords[i + 1] if i < n - 1 else None
        E_tier3.append(compute_tier3_energy(ch, next_ch, key, mode))
    
    # Get numerals
    numerals = [get_roman_numeral(ch, key, mode) for ch in chords]
    
    # Tier 1
    axes[0].bar(x, E_tier1, color=COLORS['tier1'], alpha=0.7, 
                edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Energy', fontsize=12)
    axes[0].set_title('Tier 1: Consonance Only (λ_max)', fontsize=13, fontweight='bold')
    axes[0].set_ylim(0, max(E_tier3) * 1.1)
    
    # Tier 2
    axes[1].bar(x, E_tier2, color=COLORS['tier2'], alpha=0.7,
                edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Energy', fontsize=12)
    axes[1].set_title('Tier 2: Consonance × Function', fontsize=13, fontweight='bold')
    axes[1].set_ylim(0, max(E_tier3) * 1.1)
    
    # Add numerals
    for i, (num, e) in enumerate(zip(numerals, E_tier2)):
        axes[1].text(i, e + 0.5, num, ha='center', fontsize=10, 
                     fontweight='bold', color='red')
    
    # Tier 3
    axes[2].bar(x, E_tier3, color=COLORS['tier3'], alpha=0.7,
                edgecolor='black', linewidth=1.5)
    axes[2].set_ylabel('Energy', fontsize=12)
    axes[2].set_xlabel('Chord', fontsize=12)
    axes[2].set_title('Tier 3: Consonance × Function × Resolution', 
                      fontsize=13, fontweight='bold')
    axes[2].set_ylim(0, max(E_tier3) * 1.1)
    
    # X-axis labels
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(chords, fontsize=11, fontweight='bold')
    
    if title:
        fig.suptitle(f'{title} ({key})', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ==============================================================================
# Multi-piece Comparison
# ==============================================================================

def plot_multi_piece_comparison(
    analyses: Dict[str, TrajectoryAnalysis],
    metric: str = 'V',
    figsize: Tuple[int, int] = (16, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare energy profiles across multiple pieces.
    
    Parameters
    ----------
    analyses : Dict[str, TrajectoryAnalysis]
        Dictionary mapping piece names to analyses
    metric : str, optional
        Which metric to plot: 'V', 'K', 'L', or 'E_total'
    figsize : Tuple[int, int], optional
        Figure size
    save_path : str or None, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    metric_labels = {
        'V': 'Potential Energy (V)',
        'K': 'Kinetic Energy (K)',
        'L': 'Lagrangian (L = K - V)',
        'E_total': 'Total Energy (K + V)',
    }
    
    for name, analysis in analyses.items():
        points = analysis.points
        
        # Normalize time to [0, 1] for comparison
        max_time = points[-1].time if points[-1].time > 0 else 1.0
        times_normalized = [p.time / max_time for p in points]
        
        if metric == 'V':
            values = [p.V for p in points]
        elif metric == 'K':
            values = [p.K for p in points]
        elif metric == 'L':
            values = [p.L for p in points]
        else:  # E_total
            values = [p.K + p.V for p in points]
        
        color = PIECE_COLORS.get(name, None)
        ax.plot(times_normalized, values, 'o-', linewidth=2, 
                markersize=6, label=name, color=color)
    
    ax.set_xlabel('Normalized Time', fontsize=12)
    ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12)
    ax.set_title(f'Multi-piece Comparison: {metric_labels.get(metric, metric)}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    
    if metric == 'L':
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ==============================================================================
# Summary Statistics Bar Chart
# ==============================================================================

def plot_summary_comparison(
    analyses: Dict[str, TrajectoryAnalysis],
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart comparing key metrics across pieces.
    
    Parameters
    ----------
    analyses : Dict[str, TrajectoryAnalysis]
        Dictionary mapping piece names to analyses
    figsize : Tuple[int, int], optional
        Figure size
    save_path : str or None, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    names = list(analyses.keys())
    x = np.arange(len(names))
    colors = [PIECE_COLORS.get(n, '#888888') for n in names]
    
    # Total Action
    ax = axes[0, 0]
    values = [analyses[n].total_action for n in names]
    bars = ax.bar(x, values, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_ylabel('Total Action (S)', fontsize=11)
    ax.set_title('Total Action', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    
    # Catharsis Index
    ax = axes[0, 1]
    values = [analyses[n].catharsis_index for n in names]
    ax.bar(x, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Catharsis Index (C)', fontsize=11)
    ax.set_title('Catharsis Index', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    
    # Energy Range
    ax = axes[1, 0]
    values = [analyses[n].energy_range for n in names]
    ax.bar(x, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Energy Range', fontsize=11)
    ax.set_title('Energy Range (Dynamic Contrast)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    
    # Dramatic Arc
    ax = axes[1, 1]
    values = [analyses[n].dramatic_arc for n in names]
    ax.bar(x, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Dramatic Arc', fontsize=11)
    ax.set_title('Dramatic Arc (Range / Mean)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    
    fig.suptitle('Comparative Analysis Summary', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ==============================================================================
# Resolution Type Visualization
# ==============================================================================

def plot_resolution_diagram(
    analysis: TrajectoryAnalysis,
    title: str = '',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Diagram showing static vs dynamic resolution characteristics.
    
    Parameters
    ----------
    analysis : TrajectoryAnalysis
        Result from analyze_trajectory()
    title : str, optional
        Plot title
    figsize : Tuple[int, int], optional
        Figure size
    save_path : str or None, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get metrics
    S = analysis.total_action
    C = analysis.catharsis_index
    
    # Determine color based on resolution type
    color = COLORS['static'] if analysis.resolution_type == 'static' else COLORS['dynamic']
    
    # Plot point
    ax.scatter([S], [C], s=500, c=[color], edgecolors='black', linewidths=2, zorder=10)
    
    # Annotate
    if title:
        ax.annotate(title, (S, C), textcoords='offset points', xytext=(15, 15),
                    fontsize=12, fontweight='bold')
    
    # Quadrant lines
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Quadrant labels
    ax.text(ax.get_xlim()[0] * 0.8, ax.get_ylim()[1] * 0.8, 
            'Static\n(Low C, S<0)', ha='center', fontsize=10, alpha=0.7)
    ax.text(ax.get_xlim()[1] * 0.8, ax.get_ylim()[1] * 0.8,
            'Dynamic\n(Low C, S>0)', ha='center', fontsize=10, alpha=0.7)
    ax.text(ax.get_xlim()[0] * 0.8, ax.get_ylim()[0] * 0.2,
            'Static+Cathartic\n(High C, S<0)', ha='center', fontsize=10, alpha=0.7)
    ax.text(ax.get_xlim()[1] * 0.8, ax.get_ylim()[0] * 0.2,
            'Dynamic+Cathartic\n(High C, S>0)', ha='center', fontsize=10, alpha=0.7)
    
    ax.set_xlabel('Total Action (S)', fontsize=12)
    ax.set_ylabel('Catharsis Index (C)', fontsize=12)
    ax.set_title('Resolution Type Classification', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ==============================================================================
# Demonstration
# ==============================================================================

def demonstrate_visualizations():
    """Demonstrate visualization capabilities."""
    print("=" * 60)
    print("  VISUALIZATION DEMONSTRATION")
    print("=" * 60)
    print()
    print("  Available functions:")
    print("  - plot_2d_energy_landscape(): Time×Pitch heatmap")
    print("  - plot_energy_profile(): K, V, L time series")
    print("  - plot_three_tier_comparison(): Tier 1/2/3 bars")
    print("  - plot_multi_piece_comparison(): Cross-piece profiles")
    print("  - plot_summary_comparison(): Key metrics comparison")
    print("  - plot_resolution_diagram(): S vs C classification")
    print()
    print("  Run with actual TrajectoryAnalysis objects for plots.")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_visualizations()
