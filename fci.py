import numpy as np
from pyscf import gto, scf, mcscf

# Molecule/setup
mol = gto.M(
    atom="H 0 0 0; F 0 0 0.793766",
    basis="6-31g",
    unit="angstrom",
    charge=0,
    spin=0,
    verbose=0,
)

# RHF reference
mf = scf.RHF(mol)
mf.kernel()

# CASCI active space (same as your HF_0.793 workflow)
ncas = 6
nelecas = 6

# Number of excited states to print
n_excited = 117

# CASCI roots: ground + excited
mc = mcscf.CASCI(mf, ncas, nelecas)
mc.fcisolver.nroots = n_excited + 1
energies = np.atleast_1d(mc.kernel()[0]).astype(float)

# Excited-state energies only
e0 = energies[0]
excited = energies[1:]

print("CASCI excited-state energies (Hartree):")
print(excited)
