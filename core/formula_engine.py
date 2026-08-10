"""
Formula Inference Engine — Compute from Physical Laws
======================================================
Instead of looking up "the speed of light is 299792458 m/s",
this engine stores formulas and derives answers:

  "What is the kinetic energy of a 2kg object at 3 m/s?"
  → Formula: E = ½mv²
  → E = 0.5 * 2 * 3² = 9 J

This is the "teach basic laws, derive everything" approach.
"""

import re
import math
from typing import Optional, Dict, List


# ============================================================
# Physical Formulas
# ============================================================

FORMULAS = {
    'kinetic_energy': {
        'name': 'Kinetic Energy',
        'formula': 'E = ½mv²',
        'compute': lambda m, v: 0.5 * m * v**2,
        'unit': 'J',
        'variables': {'m': 'mass (kg)', 'v': 'velocity (m/s)'},
        'patterns': [
            r'kinetic\s+energy.*?(\d+\.?\d*)\s*(?:kg|kilogram).*?(\d+\.?\d*)\s*(?:m/s|meters?\s*/?\s*s)',
            r'kinetic\s+energy.*?mass\s*(?:of\s+)?(\d+\.?\d*).*?(?:speed|velocity)\s*(?:of\s+)?(\d+\.?\d*)',
        ],
    },
    'potential_energy': {
        'name': 'Gravitational Potential Energy',
        'formula': 'E = mgh',
        'compute': lambda m, g, h: m * g * h,
        'unit': 'J',
        'variables': {'m': 'mass (kg)', 'g': 'gravity (m/s²)', 'h': 'height (m)'},
        'patterns': [
            r'potential\s+energy.*?(\d+\.?\d*)\s*kg.*?(\d+\.?\d*)\s*m',
        ],
    },
    'force': {
        'name': "Newton's Second Law",
        'formula': 'F = ma',
        'compute': lambda m, a: m * a,
        'unit': 'N',
        'variables': {'m': 'mass (kg)', 'a': 'acceleration (m/s²)'},
        'patterns': [
            r'force.*?(\d+\.?\d*)\s*kg.*?(\d+\.?\d*)\s*m/s',
            r'force.*?mass\s*(?:of\s+)?(\d+\.?\d*).*?acceleration\s*(?:of\s+)?(\d+\.?\d*)',
        ],
    },
    'work': {
        'name': 'Work',
        'formula': 'W = Fd',
        'compute': lambda f, d: f * d,
        'unit': 'J',
        'variables': {'F': 'force (N)', 'd': 'distance (m)'},
        'patterns': [
            r'work.*?(\d+\.?\d*)\s*(?:N|newton).*?(\d+\.?\d*)\s*m',
            r'work.*?force\s*(?:of\s+)?(\d+\.?\d*).*?distance\s*(?:of\s+)?(\d+\.?\d*)',
            r'work\s+done.*?(\d+\.?\d*)\s*(?:N|newton).*?(\d+\.?\d*)\s*m',
        ],
    },
    'power': {
        'name': 'Power',
        'formula': 'P = W/t',
        'compute': lambda w, t: w / t if t != 0 else float('inf'),
        'unit': 'W',
        'variables': {'W': 'work (J)', 't': 'time (s)'},
        'patterns': [
            r'power.*?(\d+\.?\d*)\s*J.*?(\d+\.?\d*)\s*s',
        ],
    },
    'ohms_law': {
        'name': "Ohm's Law",
        'formula': 'V = IR',
        'compute': lambda i, r: i * r,
        'unit': 'V',
        'variables': {'I': 'current (A)', 'R': 'resistance (Ω)'},
        'patterns': [
            r'voltage.*?(\d+\.?\d*)\s*(?:A|amp).*?(\d+\.?\d*)\s*(?:Ω|ohm)',
            r'voltage.*?current\s*(?:of\s+)?(\d+\.?\d*).*?resistance\s*(?:of\s+)?(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*(?:A|amp).*?(\d+\.?\d*)\s*(?:Ω|ohm)',
        ],
    },
    'density': {
        'name': 'Density',
        'formula': 'ρ = m/V',
        'compute': lambda m, v: m / v if v != 0 else float('inf'),
        'unit': 'kg/m³',
        'variables': {'m': 'mass (kg)', 'V': 'volume (m³)'},
        'patterns': [
            r'density.*?(\d+\.?\d*)\s*kg.*?(\d+\.?\d*)\s*(?:m³|m\^3|liter)',
        ],
    },
    'speed': {
        'name': 'Speed',
        'formula': 'v = d/t',
        'compute': lambda d, t: d / t if t != 0 else float('inf'),
        'unit': 'm/s',
        'variables': {'d': 'distance (m)', 't': 'time (s)'},
        'patterns': [
            r'(?:speed|velocity).*?(\d+\.?\d*)\s*(?:m|km|meter|kilometer).*?(\d+\.?\d*)\s*(?:s|second|hour)',
        ],
    },
    'momentum': {
        'name': 'Momentum',
        'formula': 'p = mv',
        'compute': lambda m, v: m * v,
        'unit': 'kg⋅m/s',
        'variables': {'m': 'mass (kg)', 'v': 'velocity (m/s)'},
        'patterns': [
            r'momentum.*?(\d+\.?\d*)\s*kg.*?(\d+\.?\d*)\s*m/s',
        ],
    },
    'wavelength_frequency': {
        'name': 'Wave Equation',
        'formula': 'v = fλ (or c = fλ for light)',
        'compute': lambda f, lam: f * lam,
        'unit': 'm/s',
        'variables': {'f': 'frequency (Hz)', 'λ': 'wavelength (m)'},
        'patterns': [
            r'(?:wavelength|frequency).*?(\d+\.?\d*(?:e[+-]?\d+)?)\s*(?:Hz|hertz).*?(\d+\.?\d*(?:e[+-]?\d+)?)\s*(?:m|nm)',
        ],
    },
    'pressure': {
        'name': 'Pressure',
        'formula': 'P = F/A',
        'compute': lambda f, a: f / a if a != 0 else float('inf'),
        'unit': 'Pa',
        'variables': {'F': 'force (N)', 'A': 'area (m²)'},
        'patterns': [
            r'pressure.*?(\d+\.?\d*)\s*N.*?(\d+\.?\d*)\s*(?:m²|m\^2)',
        ],
    },
    'einstein_energy': {
        'name': 'Mass-Energy Equivalence',
        'formula': 'E = mc²',
        'compute': lambda m: m * (299792458 ** 2),
        'unit': 'J',
        'variables': {'m': 'mass (kg)'},
        'patterns': [
            r'(?:energy|e\s*=\s*mc).*?(\d+\.?\d*(?:e[+-]?\d+)?)\s*kg',
        ],
    },
}

# Unit conversions for flexibility
UNIT_CONVERSIONS = {
    'km': ('m', 1000),
    'cm': ('m', 0.01),
    'mm': ('m', 0.001),
    'km/h': ('m/s', 1/3.6),
    'mph': ('m/s', 0.44704),
    'g': ('kg', 0.001),
    'mg': ('kg', 1e-6),
    'ton': ('kg', 1000),
    'lb': ('kg', 0.453592),
    'nm': ('m', 1e-9),
    'μm': ('m', 1e-6),
    'liter': ('m³', 0.001),
    'mL': ('m³', 1e-6),
    'minute': ('s', 60),
    'hour': ('s', 3600),
    'day': ('s', 86400),
}


class FormulaEngine:
    """Computes answers from physical formulas."""

    def __init__(self):
        self._formulas = FORMULAS

    def solve(self, question: str) -> Optional[Dict]:
        """Try to solve a physics question using stored formulas."""
        q_lower = question.lower()

        for key, formula in self._formulas.items():
            for pattern in formula['patterns']:
                m = re.search(pattern, q_lower, re.IGNORECASE)
                if m:
                    values = [float(g) for g in m.groups()]
                    try:
                        result = formula['compute'](*values)
                        # Format result nicely
                        if result == int(result):
                            result_str = str(int(result))
                        elif abs(result) > 1e6 or abs(result) < 0.001:
                            result_str = f'{result:.3e}'
                        else:
                            result_str = f'{result:.2f}'

                        return {
                            'answer': f'{result_str} {formula["unit"]}',
                            'formula_name': formula['name'],
                            'formula': formula['formula'],
                            'values': values,
                            'result': result,
                            'unit': formula['unit'],
                        }
                    except (TypeError, ValueError, ZeroDivisionError):
                        continue

        return None

    def explain_formula(self, formula_name: str) -> Optional[str]:
        """Explain a formula."""
        name_lower = formula_name.lower()
        for key, formula in self._formulas.items():
            if name_lower in key or name_lower in formula['name'].lower():
                parts = [f"{formula['name']}: {formula['formula']}"]
                parts.append("Variables:")
                for var, desc in formula['variables'].items():
                    parts.append(f"  {var} = {desc}")
                parts.append(f"Unit: {formula['unit']}")
                return '\n'.join(parts)
        return None

    def list_formulas(self) -> List[str]:
        """List all known formulas."""
        return [f"{f['name']}: {f['formula']}" for f in self._formulas.values()]
