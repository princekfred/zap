from pathlib import Path
import numpy as np
from pyscf import gto, scf, mcscf

out_path = Path("outputs/CH+ Trotterized/CASCI_output.txt")
# Molecule/setup
mol = gto.M(
    atom="C 0 0 0; H 0 0 2.261840985",
    basis="6-31g",
    unit="angstrom",
    charge=1,
    spin=0,
    verbose=0,
)

# RHF reference
mf = scf.RHF(mol)
mf.kernel()

# CASCI active space (same as your HF_0.793 workflow)
ncas = 6
nelecas = 6

# Number of excited states to save
n_excited = 117

# CASCI roots: ground + excited
mc = mcscf.CASCI(mf, ncas, nelecas)
mc.fcisolver.nroots = n_excited + 1
energies = np.atleast_1d(mc.kernel()[0]).astype(float)

# Excited-state energies only
excited = energies[1:]

out_path.parent.mkdir(parents=True, exist_ok=True)
np.savetxt(out_path, excited, fmt="%.12f")

print(f"Saved CASCI excited-state energies to {out_path}")
