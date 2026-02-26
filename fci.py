from pathlib import Path
import numpy as np
from pyscf import gto, scf, mcscf

out_path = Path("outputs/H4_trotterized/CASCI_output.txt")
# Molecule/setup
mol = gto.M(
    atom="H 0 0 0; H 0 0 3; H 0 0 6; H 0 0 9",
    basis="sto-3g",
    unit="angstrom",
    charge=0,
    spin=0,
    verbose=0,
)

# RHF reference
mf = scf.RHF(mol)
mf.kernel()

# CASCI active space (same as your HF_0.793 workflow)
ncas = 4
nelecas = 4

# Number of excited states to save
n_excited = 26

# CASCI roots: ground + excited
mc = mcscf.CASCI(mf, ncas, nelecas)
mc.fcisolver.nroots = n_excited + 1
energies = np.atleast_1d(mc.kernel()[0]).astype(float)

# Excited-state energies only
excited = energies[1:]

out_path.parent.mkdir(parents=True, exist_ok=True)
np.savetxt(out_path, excited, fmt="%.12f")

print(f"Saved CASCI excited-state energies to {out_path}")
