"""
Generate FCIDUMP scans for all supported diatomic molecules.
No command-line arguments required.

Output:
  - FCIDUMP files are written directly into ./fcidump/
    e.g., fcidump_CO_1.13.fcidump

"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

from pyscf import gto, scf
from pyscf.tools import fcidump

# ----------------------------
# Configuration (edit if needed)
# ----------------------------
BASIS = "sto-3g"
DELTA = 0.10  # 5-point scan: re-2*DELTA, re-DELTA, re, re+DELTA, re+2*DELTA

BASE_DIR = os.path.dirname(__file__)
FCIDUMP_DIR = os.path.join(BASE_DIR, "fcidump")
os.makedirs(FCIDUMP_DIR, exist_ok=True)

# Equilibrium bond lengths (Å)
EQUILIBRIUM_BOND: Dict[str, float] = {
    #"H2": 0.74,
    "N2": 1.10,
    "O2": 1.21,
    "F2": 1.41,
    "Cl2": 1.99,
    "CO": 1.13,
    "NO": 1.15,
    #"HF": 0.92,
    #"HCl": 1.27,
    #"LiH": 1.60,
    #"BeH": 1.34,
    #"CH": 1.12,
    #"NH": 1.04,
    #"OH": 0.97,
}

# Spin multiplicity (2S+1)
# IMPORTANT: BeH must be doublet (odd electron count => cannot be singlet)
MULTIPLICITY: Dict[str, int] = {
    "H2": 1,
    "N2": 1,
    "O2": 3,
    "F2": 1,
    "Cl2": 1,
    "CO": 1,
    "NO": 2,
    "HF": 1,
    "HCl": 1,
    "LiH": 1,
    "BeH": 2,  # fixed
    "CH": 2,
    "NH": 3,
    "OH": 2,
}


def parse_atoms(name: str) -> Tuple[str, str]:
    """
    Parse a diatomic formula into (atom1, atom2).

    Handles:
      - Homonuclear with digit suffix: N2, Cl2 -> (N, N), (Cl, Cl)
      - Heteronuclear: CO, NO, HF, HCl, BeH, LiH -> (C, O), (N, O), (H, F), (H, Cl), (Be, H), (Li, H)
    """
    if not name or len(name) < 2:
        raise ValueError(f"Unrecognized diatomic format: {name}")

    # Case 1: homonuclear X2 form (ends with digit)
    if name[-1].isdigit():
        element = name[:-1]
        if not re.fullmatch(r"[A-Z][a-z]?", element):
            raise ValueError(f"Unrecognized element symbol in: {name}")
        return element, element

    # Case 2: exactly two element symbols
    symbols = re.findall(r"[A-Z][a-z]?", name)
    if len(symbols) == 2 and "".join(symbols) == name:
        return symbols[0], symbols[1]

    raise ValueError(f"Unrecognized diatomic format: {name}")


def build_diatomic(name: str, bond: float) -> gto.Mole:
    """Build a PySCF molecule with correct spin and geometry."""
    if name not in EQUILIBRIUM_BOND or name not in MULTIPLICITY:
        raise ValueError(f"Unknown molecule: {name}")

    atom1, atom2 = parse_atoms(name)

    multiplicity = MULTIPLICITY[name]
    spin = multiplicity - 1  # PySCF uses spin = Nalpha - Nbeta = 2S

    mol = gto.Mole()
    mol.atom = f"{atom1} 0 0 0; {atom2} 0 0 {bond}"
    mol.unit = "Angstrom"
    mol.basis = BASIS
    mol.charge = 0
    mol.spin = spin
    mol.build()
    return mol


def bond_grid(re_eq: float) -> List[float]:
    """Generate 5 bond lengths around equilibrium, rounded to 2 decimals."""
    vals = [re_eq - 2 * DELTA, re_eq - DELTA, re_eq, re_eq + DELTA, re_eq + 2 * DELTA]
    return [round(v, 2) for v in vals]


def main() -> None:
    molecules = sorted(EQUILIBRIUM_BOND.keys())

    print("=== FCIDUMP generation started ===")
    print(f"Basis = {BASIS}")
    print(f"Output folder = {FCIDUMP_DIR}")
    print(f"Molecules = {', '.join(molecules)}")
    print()

    for name in molecules:
        re_eq = EQUILIBRIUM_BOND[name]
        bonds = bond_grid(re_eq)

        print(f"=== {name}: 5-point scan ===")
        print("Bond lengths =", ", ".join(f"{b:.2f}" for b in bonds))

        for bond in bonds:
            try:
                mol = build_diatomic(name, bond)

                # RHF for closed-shell, ROHF for open-shell
                if MULTIPLICITY[name] == 1:
                    mf = scf.RHF(mol)
                else:
                    mf = scf.ROHF(mol)

                energy = float(mf.kernel())

                outfile = os.path.join(FCIDUMP_DIR, f"fcidump_{name}_{bond:.2f}.fcidump")
                fcidump.from_scf(mf, outfile)

                print(f"[DONE] {name} | r={bond:.2f} Å | E={energy:.6f} | File={outfile}")

            except Exception as e:
                # Do not stop the whole sweep if one point fails.
                print(f"[SKIP] {name} | r={bond:.2f} Å | Reason: {type(e).__name__}: {e}")

        print()

    print("=== FCIDUMP generation completed ===")


if __name__ == "__main__":
    main()
