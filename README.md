# HELA: Harmonic Energy Landscape Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A framework for **interpretable music analysis** using energy landscape theory.

## Overview

HELA provides a physics-inspired approach to analyzing music, treating melodic trajectories as paths through a harmonic energy landscape. The framework enables:

- **Quantitative emotional analysis** via Catharsis Index
- **Resolution classification** (Static vs Dynamic)
- **Three-tier energy model** combining acoustics and music theory
- **Interpretable features** for Music AI training data

## Key Concepts

### Three-Tier Energy Model

| Tier | Component | Description | Nature |
|------|-----------|-------------|--------|
| 1 | Consonance | λ_max from Consonance Tensor | **Rigorous** |
| 2 | + Function | Tonic/Dominant/Subdominant weighting | Heuristic |
| 3 | + Resolution | V→I resolution pressure | Heuristic |

### Lagrangian Mechanics
```
L = K - V

where:
  K = ½m(Δpitch)²     (melodic motion energy)
  V = E_harmonic       (harmonic stability)
```

### Total Action & Classification
```
S = Σ L(t) × Δt

S < 0 → Static Resolution  (stability-dominated)
S > 0 → Dynamic Resolution (motion-dominated)
```

## Installation
```bash
git clone https://github.com/miosync-masa/HELA.git
cd HELA
pip install -r requirements.txt
```

## Quick Start
```python
from hela import quick_analysis

# Analyze a simple progression
result = quick_analysis(
    pitches=[60, 62, 64, 65, 67, 65, 64, 60],
    durations=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
    chords=['C', 'C', 'F', 'F', 'G', 'G', 'C', 'C'],
    key='C Major',
    title='Simple Scale'
)

# Output:
# Total Action (S):         -62.34
# Catharsis Index (C):        0.47
# Resolution Type:          STATIC
```

## Core Functions

### Analysis
```python
from hela import analyze_trajectory

result = analyze_trajectory(
    pitches=[...],      # MIDI pitch sequence
    durations=[...],    # Note durations (beats)
    chords=[...],       # Chord symbols
    key='C Major',      # Key signature
    tempo=120.0,        # BPM
)

print(f"Total Action: {result.total_action:.2f}")
print(f"Catharsis: {result.catharsis_index:.2f}")
print(f"Type: {result.resolution_type}")
```

### Energy Computation
```python
from hela import compute_tier1_energy, compute_tier2_energy, compute_tier3_energy

# Tier 1: Base consonance only
E1 = compute_tier1_energy('G7')  # → 15.18

# Tier 2: + Functional weight
E2 = compute_tier2_energy('G7', 'C Major')  # → 24.28 (V7 function)

# Tier 3: + Resolution pressure
E3 = compute_tier3_energy('G7', 'C', 'C Major')  # → 19.42 (V7→I resolves)
```

### Visualization
```python
from hela import plot_2d_energy_landscape, plot_energy_profile

# 2D Energy Landscape (Time × Pitch heatmap)
plot_2d_energy_landscape(result, title='My Piece')

# Energy Profile (K, V, L time series)
plot_energy_profile(result, title='My Piece')
```

### Multi-piece Comparison
```python
from hela import compare_pieces

analyses = compare_pieces({
    'Canon': {'pitches': [...], 'durations': [...], 'chords': [...], 'key': 'D Major'},
    'Let It Be': {'pitches': [...], 'durations': [...], 'chords': [...], 'key': 'C Major'},
    'Senbonzakura': {'pitches': [...], 'durations': [...], 'chords': [...], 'key': 'D minor'},
})
```

## Empirical Results

| Piece | Total Action (S) | Catharsis (C) | Classification |
|-------|-----------------|---------------|----------------|
| Canon in D | -62.34 | 0.47 | Static |
| Let It Be | -77.09 | 0.13 | Static |
| Senbonzakura | +182.81 | 6.02 | **Dynamic** |

Senbonzakura shows **40× higher catharsis** than Canon/Let It Be, quantifying the perceived dramatic intensity difference.

## Project Structure
```
HELA/
├── README.md
├── LICENSE
├── requirements.txt
├── hela/
│   ├── __init__.py        # Package interface
│   ├── consonance.py      # Tier 1: Continued fractions, eigenvalues
│   ├── harmony.py         # Tier 2-3: Roman numerals, weights
│   ├── energy.py          # Three-tier energy computation
│   ├── trajectory.py      # Lagrangian, Action, Catharsis
│   └── visualization.py   # Energy landscape plots
└── examples/
    └── (sample scripts)
```

## Theoretical Foundation

This framework is based on:

1. **Consonance Tensor** - Spectral theory for interval consonance
   - Iizumi, M. (2025). "On the Consonance of Prime Factorization"
   - DOI: 10.5281/zenodo.XXXXXXX

2. **Energy Landscape Theory** - Lagrangian mechanics for music
   - See paper: "Beyond Tags: An Energy Landscape Approach to Interpretable Training Data for Music AI"

### Why Three Tiers?

- **Tier 1 alone** cannot distinguish V from I (same consonance)
- **Tier 2** adds functional context (V has dominant tension)
- **Tier 3** adds resolution dynamics (V→I releases tension)

See Supplementary Information SI-B.5 for mathematical necessity proofs.

## Citation
```bibtex
@software{hela2025,
  author = {Iizumi, Masamichi},
  title = {HELA: Harmonic Energy Landscape Analyzer},
  year = {2025},
  url = {https://github.com/miosync-masa/HELA}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**Masamichi Iizumi** - Miosync Inc.
---

*"Physics = Hierarchical Constraint Satisfaction on Λ-space"*
