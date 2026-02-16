from pyscf import gto, scf
import numpy as np

symbols = ['H', 'H', 'H', 'H']
geometry = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 3.0],
    [0.0, 0.0, 6.0],
    [0.0, 0.0, 9.0]])
charge = 0

# Build molecule
mol = gto.Mole()
mol.atom = [[symbols[i], geometry[i]] for i in range(len(symbols))]
mol.basis = 'sto-3g'
mol.charge = charge
mol.build()

# Run Hartree-Fock
mf = scf.RHF(mol)
mf.kernel()

# SCF orbitals
orbital_energies = np.array(mf.mo_energy)

# Total spatial orbitals
n_spatial_orbitals = mol.nao_nr()
n_elec = mol.nelectron
n_occ = n_elec // 2
n_vir = n_spatial_orbitals - n_occ

# Print info
print("Number of spatial orbitals:", n_spatial_orbitals)
print("Number of occupied orbitals:", n_occ)
print("Number of virtual orbitals:", n_vir)
print("SCF Orbital energies (Hartree):", orbital_energies)

# Save output to npy file
output = {
    "n_spatial_orbitals": n_spatial_orbitals,
    "n_occ": n_occ,
    "n_vir": n_vir,
    "orbital_energies": orbital_energies}

#save to text file
with open("fock.txt", "w") as f:
    f.write(f"Number of spatial orbitals: {n_spatial_orbitals}\n")
    f.write(f"Number of occupied orbitals: {n_occ}\n")
    f.write(f"Number of virtual orbitals: {n_vir}\n")
    f.write(f"SCF Orbital energies (Hartree): {orbital_energies}")




## 2e- integrals
from pyscf import ao2mo
# --- RHF ---
mf = scf.RHF(mol).run()
C = mf.mo_coeff
norb = C.shape[1]
nspin = 2 * norb

# --- MO ERIs in chemist notation (pq|rs) ---
eri = ao2mo.kernel(mol, C, aosym="s1")
eri = ao2mo.restore(1, eri, norb)  # full 4-index (p,q,r,s)

def so(i):
    """spin-orbital index -> (spatial index, spin:0 alpha / 1 beta)"""
    return i // 2, i % 2

lines = []
lines.append("Spin-orbital antisymmetrized integrals <ij||kl> (MO basis)")

thr = 1e-10
count = 0

for i in range(nspin):
    I, si = so(i)
    for j in range(nspin):
        J, sj = so(j)
        for k in range(nspin):
            K, sk = so(k)
            for l in range(nspin):
                L, sl = so(l)

                # <ij||kl> = (ik|jl) - (il|jk) with spin deltas
                term1 = eri[I, K, J, L] if (si == sk and sj == sl) else 0.0
                term2 = eri[I, L, J, K] if (si == sl and sj == sk) else 0.0
                val = term1 - term2
                val = -val
                if abs(val) > thr:
                    lines.append(f"{i}^ {j}^ {k} {l}\t|\t{val}")
                    count += 1

print("printed:", count)

with open("two_elec.txt", "w") as f:
    f.write("\n".join(lines))
