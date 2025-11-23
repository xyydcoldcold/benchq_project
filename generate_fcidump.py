"""
Compute FCIDUMP files for N2 at varying bond lengths using PySCF.
All comments in English.

The script:
1. Loops over a list of bond lengths (in Angstrom).
2. Builds the N2 molecule for each bond length.
3. Runs RHF.
4. Writes an FCIDUMP file for each geometry.

Output:
    fcidump_N2_<bond_length>.fcidump
in the same directory as the script.
"""

from pyscf import gto, scf
from pyscf.tools import fcidump

def compute_fcidump_for_bond_length(bond_length):
    """Build N2, run SCF, and write FCIDUMP for a given bond length."""
    mol = gto.Mole()
    mol.atom = f"N 0 0 0; N 0 0 {bond_length}"
    mol.basis = "sto-3g"
    mol.unit = "Angstrom"
    mol.build()

    mf = scf.RHF(mol)
    energy = mf.kernel()

    output_name = f"fcidump_N2_{bond_length:.2f}.fcidump"
    fcidump.from_scf(mf, output_name)

    print(f"[DONE] Bond length = {bond_length:.2f} Å | Energy = {energy:.6f} | File = {output_name}")


def main():
    # Bond lengths to scan (Angstrom)
    bond_lengths = [0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3]

    print("=== N2 FCIDUMP Scan Started ===")
    for L in bond_lengths:
        compute_fcidump_for_bond_length(L)

    print("=== All FCIDUMPs generated successfully ===")


if __name__ == "__main__":
    main()
