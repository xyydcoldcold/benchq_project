import numpy as np
import re
import json
import argparse

# BenchQ import for DFQPE resource estimation
from benchq.problem_embeddings.qpe import get_double_factorized_qpe_toffoli_and_qubit_cost  # :contentReference[oaicite:19]{index=19}

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
        # Extract number of orbitals (NORB) from header
        match = re.search(r"NORB\s*=\s*(\d+)", header, re.IGNORECASE)
        if match:
            norb = int(match.group(1))
        else:
            raise ValueError(f"NORB not found in FCIDUMP header of {filename}")
        # Prepare containers for integrals
        h1 = np.zeros((norb, norb), dtype=float)       # one-electron integrals matrix
        eri = np.zeros((norb, norb, norb, norb), dtype=float)  # two-electron integrals tensor
        ecore = 0.0
        # Read integral lines
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # skip if line is empty or incomplete
            # Parse value and indices
            # Handle scientific notation 'D' exponent if present by replacing with 'E'
            val_str = parts[0].replace('D', 'E').replace('d', 'E')
            try:
                val = float(val_str)
            except ValueError:
                # Skip lines that don't parse as float (if any)
                continue
            i, j, k, l = map(int, parts[1:5])
            if k == 0:
                # This is a one-electron integral or core energy line
                if j != 0:
                    # One-electron term h(i,j)
                    h1[i-1, j-1] = val
                    h1[j-1, i-1] = val  # ensure symmetry
                else:
                    # Core energy term (all indices 0)
                    ecore = val
            else:
                # Two-electron integral (i,j|k,l)
                I, J, K, L = i-1, j-1, k-1, l-1
                # Assign the value to all symmetric permutations to maintain (ij|kl) = (kl|ij) symmetry:contentReference[oaicite:20]{index=20}
                eri[I, J, K, L] = val
                eri[J, I, K, L] = val
                eri[I, J, L, K] = val
                eri[J, I, L, K] = val
                eri[K, L, I, J] = val
                eri[L, K, I, J] = val
                eri[K, L, J, I] = val
                eri[L, K, J, I] = val
        return h1, eri, ecore

def main():
    """
    Automatically scan fcidumps/ directory and estimate resources
    for every FCIDUMP file inside.
    """

    import glob

    # Automatically find all FCIDUMP files
    fcidump_files = sorted(glob.glob("fcidumps/*.fcidump"))

    if not fcidump_files:
        print("No FCIDUMP files found in directory 'fcidumps/'")
        return

    print("Found FCIDUMP files:")
    for f in fcidump_files:
        print("   ", f)
    print()

    results = []
    threshold = 1e-6  # You can adjust this if needed

    for file_path in fcidump_files:
        print(f"Processing: {file_path}")

        # Parse integrals
        h1, eri, e_core = parse_fcidump(file_path)

        # BenchQ DF-QPE resource estimation
        toffoli_count, logical_qubits = get_double_factorized_qpe_toffoli_and_qubit_cost(
            h1, eri, threshold
        )

        results.append({
            "file": file_path,
            "toffoli_count": int(toffoli_count),
            "logical_qubits": int(logical_qubits),
            "ecore": float(e_core)
        })

    # Always write to results.json
    with open("results.json", "w") as fout:
        json.dump(results, fout, indent=2)

    print("\nResource estimates saved to results.json")


if __name__ == "__main__":
    main()
