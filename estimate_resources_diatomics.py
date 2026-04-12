"""
Estimate DF-QPE resources for all diatomic FCIDUMP files in ./fcidump/.
No command-line arguments required.

Input:
  - ./fcidump/fcidump_<MOL>_<bond>.fcidump

Output:
  - ./result/results_<MOL>.json   (one JSON per molecule)
  - ./result/results_all.json     (combined)

All comments in English.
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from benchq.problem_embeddings.qpe import (
    get_double_factorized_qpe_toffoli_and_qubit_cost,
)

BASE_DIR = os.path.dirname(__file__)
FCIDUMP_DIR = os.path.join(BASE_DIR, "fcidump")
RESULT_DIR = os.path.join(BASE_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

THRESHOLD = 1e-10


def parse_fcidump(filename: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Parse an FCIDUMP file to extract one-electron and two-electron integrals.
    Returns (h1_matrix, eri_tensor, e_core).
    """
    with open(filename, "r") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                break
            header_lines.append(line)
            if "&END" in line or "/" in line:
                break
        header = "".join(header_lines)

        match = re.search(r"NORB\s*=\s*(\d+)", header, re.IGNORECASE)
        if not match:
            raise ValueError(f"NORB not found in FCIDUMP header of {filename}")
        norb = int(match.group(1))

        h1 = np.zeros((norb, norb), dtype=float)
        eri = np.zeros((norb, norb, norb, norb), dtype=float)
        ecore = 0.0

        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            val_str = parts[0].replace("D", "E").replace("d", "E")
            try:
                val = float(val_str)
            except ValueError:
                continue

            i, j, k, l = map(int, parts[1:5])

            if k == 0:
                if j != 0:
                    h1[i - 1, j - 1] = val
                    h1[j - 1, i - 1] = val
                else:
                    ecore = val
            else:
                I, J, K, L = i - 1, j - 1, k - 1, l - 1
                eri[I, J, K, L] = val
                eri[J, I, K, L] = val
                eri[I, J, L, K] = val
                eri[J, I, L, K] = val
                eri[K, L, I, J] = val
                eri[L, K, I, J] = val
                eri[K, L, J, I] = val
                eri[L, K, J, I] = val

        return h1, eri, float(ecore)


def parse_name_and_bond_from_filename(path: str) -> Tuple[str, float]:
    """
    Parse:
      fcidump_<MOL>_<bond>.fcidump
    Example:
      fcidump_N2_1.30.fcidump -> ("N2", 1.30)

    This avoids matching the "2" inside "N2" by anchoring to the final numeric suffix.
    """
    base = os.path.basename(path)
    m = re.match(r"^fcidump_([A-Za-z0-9]+)_([0-9]+(?:\.[0-9]+)?)\.fcidump$", base)
    if not m:
        raise ValueError(f"Unrecognized FCIDUMP filename format: {base}")
    mol = m.group(1)
    bond = float(m.group(2))
    return mol, bond


def main() -> None:
    pattern = os.path.join(FCIDUMP_DIR, "fcidump_*.fcidump")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No FCIDUMP files found under: {FCIDUMP_DIR}")
        return

    print("=== Resource estimation started ===")
    print(f"Input folder = {FCIDUMP_DIR}")
    print(f"Output folder = {RESULT_DIR}")
    print(f"Found {len(files)} FCIDUMP files")
    print()

    by_mol: Dict[str, List[dict]] = {}
    all_results: List[dict] = []

    for fpath in files:
        try:
            mol_name, bond = parse_name_and_bond_from_filename(fpath)
            print(f"Processing: {os.path.basename(fpath)}")

            h1, eri, ecore = parse_fcidump(fpath)

            # Metrics (same style as your reference)
            n_orbitals = int(h1.shape[0])
            n_h1_nonzero = int(np.count_nonzero(h1))
            n_eri_nonzero = int(np.count_nonzero(eri))

            lambda_approx = float(np.sum(np.abs(h1)) + np.sum(np.abs(eri)) + abs(ecore))
            lambda_per_orbital = float(lambda_approx / n_orbitals) if n_orbitals > 0 else 0.0

            h1_fro_norm = float(np.linalg.norm(h1))
            eri_fro_norm = float(np.linalg.norm(eri))

            h1_max_abs = float(np.max(np.abs(h1))) if n_h1_nonzero > 0 else 0.0
            eri_max_abs = float(np.max(np.abs(eri))) if n_eri_nonzero > 0 else 0.0

            total_h1 = n_orbitals * n_orbitals
            total_eri = n_orbitals**4
            sparsity_h1 = float(n_h1_nonzero) / float(total_h1) if total_h1 > 0 else 0.0
            sparsity_eri = float(n_eri_nonzero) / float(total_eri) if total_eri > 0 else 0.0

            toffoli_count, logical_qubits = get_double_factorized_qpe_toffoli_and_qubit_cost(
                h1, eri, THRESHOLD
            )

            entry = {
                "file": fpath,
                "molecule": mol_name,
                "bond_length": bond,
                "n_orbitals": n_orbitals,
                "n_h1_nonzero": n_h1_nonzero,
                "n_eri_nonzero": n_eri_nonzero,
                "lambda_approx": lambda_approx,
                "lambda_per_orbital": lambda_per_orbital,
                "ecore": float(ecore),
                "h1_fro_norm": h1_fro_norm,
                "eri_fro_norm": eri_fro_norm,
                "h1_max_abs": h1_max_abs,
                "eri_max_abs": eri_max_abs,
                "sparsity_h1": sparsity_h1,
                "sparsity_eri": sparsity_eri,
                "threshold": THRESHOLD,
                "toffoli_count": int(toffoli_count),
                "logical_qubits": int(logical_qubits),
            }

            by_mol.setdefault(mol_name, []).append(entry)
            all_results.append(entry)

        except Exception as e:
            print(f"[SKIP] {os.path.basename(fpath)} | Reason: {type(e).__name__}: {e}")

    # Write per-molecule JSON files (sorted by bond length)
    for mol, entries in by_mol.items():
        entries_sorted = sorted(entries, key=lambda x: float(x["bond_length"]))
        outpath = os.path.join(RESULT_DIR, f"results_{mol}.json")
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(entries_sorted, f, indent=2)
        print(f"[WRITE] {outpath}")

    # Write combined JSON
    all_outpath = os.path.join(RESULT_DIR, "results_all.json")
    with open(all_outpath, "w", encoding="utf-8") as f:
        json.dump(sorted(all_results, key=lambda x: (x["molecule"], float(x["bond_length"]))), f, indent=2)
    print(f"[WRITE] {all_outpath}")

    print("=== Resource estimation completed ===")


if __name__ == "__main__":
    main()
