import numpy as np
import re
import json
import os
import glob

# BenchQ import for DFQPE resource estimation
from benchq.problem_embeddings.qpe import get_double_factorized_qpe_toffoli_and_qubit_cost


def parse_fcidump(filename):
    """
    Parse an FCIDUMP file to extract one-electron and two-electron integrals.
    Returns (h1_matrix, eri_tensor, e_core).
    """
    with open(filename, 'r') as f:
        # Read header lines until end of header marker (&END or /)
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                break
            header_lines.append(line)
            if "&END" in line or "/" in line:
                break
        header = "".join(header_lines)

        # Extract number of orbitals (NORB)
        match = re.search(r"NORB\s*=\s*(\d+)", header, re.IGNORECASE)
        if match:
            norb = int(match.group(1))
        else:
            raise ValueError(f"NORB not found in FCIDUMP header of {filename}")

        # Prepare containers for integrals
        h1 = np.zeros((norb, norb), dtype=float)
        eri = np.zeros((norb, norb, norb, norb), dtype=float)
        ecore = 0.0

        # Read integral lines
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            # Parse value and indices
            val_str = parts[0].replace('D', 'E').replace('d', 'E')
            try:
                val = float(val_str)
            except ValueError:
                continue

            i, j, k, l = map(int, parts[1:5])

            if k == 0:
                if j != 0:
                    h1[i-1, j-1] = val
                    h1[j-1, i-1] = val
                else:
                    ecore = val
            else:
                I, J, K, L = i-1, j-1, k-1, l-1
                eri[I, J, K, L] = val
                eri[J, I, K, L] = val
                eri[I, J, L, K] = val
                eri[J, I, L, K] = val
                eri[K, L, I, J] = val
                eri[L, K, I, J] = val
                eri[K, L, J, I] = val
                eri[L, K, J, I] = val

        return h1, eri, ecore


def extract_bond_length_from_filename(path):
    """
    Extract bond length from filename like 'fcidump_N2_1.10.fcidump'
    """
    base = os.path.basename(path)
    match = re.search(r"(\d+(?:\.\d+)?)", base)
    if match:
        return float(match.group(1))
    return None


def main():
    """
    Automatically scan all FCIDUMP files in fcidumps/,
    compute DF-QPE resource estimates, and output extended metrics.
    """

    # Scan FCIDUMP directory
    fcidump_files = sorted(glob.glob("fcidumps/*.fcidump"))
    if not fcidump_files:
        print("No FCIDUMP files found in 'fcidumps/'")
        return

    print("Found FCIDUMP files:")
    for f in fcidump_files:
        print("   ", f)
    print()

    results = []
    threshold = 1e-6

    for file_path in fcidump_files:
        print(f"Processing: {file_path}")

        # Parse integrals
        h1, eri, ecore = parse_fcidump(file_path)

        # Extra Hamiltonian-level info (Level 2)
        n_orbitals = h1.shape[0]
        n_h1_nonzero = int(np.count_nonzero(h1))
        n_eri_nonzero = int(np.count_nonzero(eri))
        lambda_approx = float(np.sum(np.abs(h1)) + np.sum(np.abs(eri)) + abs(ecore))

        # Extract bond length from filename
        bond_length = extract_bond_length_from_filename(file_path)

        # BenchQ DF-QPE estimation
        toffoli_count, logical_qubits = get_double_factorized_qpe_toffoli_and_qubit_cost(
            h1, eri, threshold
        )

        result_entry = {
            "file": file_path,
            "n_orbitals": int(n_orbitals),
            "n_h1_nonzero": n_h1_nonzero,
            "n_eri_nonzero": n_eri_nonzero,
            "lambda_approx": lambda_approx,
            "ecore": float(ecore),
            "toffoli_count": int(toffoli_count),
            "logical_qubits": int(logical_qubits)
        }

        if bond_length is not None:
            result_entry["bond_length"] = bond_length

        results.append(result_entry)

    # Save to JSON
    with open("results.json", "w") as fout:
        json.dump(results, fout, indent=2)

    print("\nExtended resource estimates saved to results.json")


if __name__ == "__main__":
    main()

