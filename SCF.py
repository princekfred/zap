"""SCF utilities for the exact workflow.

This module exposes a single callable entrypoint (`run_scf`) so it can be
orchestrated by `run_exact.py` without triggering work on import.
"""


def run_scf(
    symbols,
    geometry,
    charge=0,
    basis="sto-3g",
    unit="Bohr",
    fock_output="fock.txt",
    two_e_output="two_elec.txt",
    threshold=1e-10,
    max_cycle=100,
    conv_tol=1e-9,
):
    """Run RHF and optionally write orbital and two-electron integral reports."""
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'numpy'. Install with:\n  python -m pip install numpy"
        ) from exc

    try:
        from pyscf import ao2mo, gto, scf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'pyscf'. Install with:\n  python -m pip install pyscf"
        ) from exc

    coords = np.asarray(geometry, dtype=float)

    mol = gto.Mole()
    mol.atom = [[symbols[i], coords[i].tolist()] for i in range(len(symbols))]
    mol.basis = basis
    mol.charge = int(charge)
    mol.unit = unit
    mol.build()

    mf = scf.RHF(mol)
    mf.max_cycle = int(max_cycle)
    mf.conv_tol = float(conv_tol)
    mf.kernel()

    if not mf.converged:
        print(
            "Initial RHF did not converge; retrying with atom guess, damping, and level shift."
        )
        dm0 = mf.make_rdm1()
        mf = scf.RHF(mol)
        mf.max_cycle = max(int(max_cycle) * 2, 200)
        mf.conv_tol = float(conv_tol)
        mf.level_shift = 0.5
        mf.damping = 0.2
        mf.diis_start_cycle = 1
        mf.init_guess = "atom"
        mf.kernel(dm0=dm0)
        if not mf.converged:
            print("Retry did not converge; attempting Newton solver.")
            newton_mf = scf.newton(scf.RHF(mol))
            newton_mf.max_cycle = max(int(max_cycle) * 2, 200)
            newton_mf.conv_tol = float(conv_tol)
            newton_mf.kernel(dm0=mf.make_rdm1())
            mf = newton_mf
            if not mf.converged:
                print("Warning: RHF still not converged after Newton fallback.")

    orbital_energies = np.array(mf.mo_energy)
    n_spatial_orbitals = mol.nao_nr()
    n_elec = mol.nelectron
    n_occ = n_elec // 2
    n_vir = n_spatial_orbitals - n_occ

    print("Number of spatial orbitals:", n_spatial_orbitals)
    print("Number of occupied orbitals:", n_occ)
    print("Number of virtual orbitals:", n_vir)
    print("SCF Orbital energies (Hartree):", orbital_energies)

    if fock_output:
        with open(fock_output, "w", encoding="utf-8") as f:
            f.write(f"Number of spatial orbitals: {n_spatial_orbitals}\n")
            f.write(f"Number of occupied orbitals: {n_occ}\n")
            f.write(f"Number of virtual orbitals: {n_vir}\n")
            f.write(f"SCF Orbital energies (Hartree): {orbital_energies}\n")

    # Two-electron spin-orbital antisymmetrized integrals in MO basis.
    c_mo = mf.mo_coeff
    norb = c_mo.shape[1]
    nspin = 2 * norb
    eri = ao2mo.kernel(mol, c_mo, aosym="s1")
    eri = ao2mo.restore(1, eri, norb)

    def so(idx):
        return idx // 2, idx % 2

    lines = ["Spin-orbital antisymmetrized integrals <ij||kl> (MO basis)"]
    count = 0

    for i in range(nspin):
        i_spatial, s_i = so(i)
        for j in range(nspin):
            j_spatial, s_j = so(j)
            for k in range(nspin):
                k_spatial, s_k = so(k)
                for l in range(nspin):
                    l_spatial, s_l = so(l)
                    term1 = (
                        eri[i_spatial, k_spatial, j_spatial, l_spatial]
                        if (s_i == s_k and s_j == s_l)
                        else 0.0
                    )
                    term2 = (
                        eri[i_spatial, l_spatial, j_spatial, k_spatial]
                        if (s_i == s_l and s_j == s_k)
                        else 0.0
                    )
                    value = -(term1 - term2)
                    if abs(value) > threshold:
                        lines.append(f"{i}^ {j}^ {k} {l}\t|\t{value}")
                        count += 1

    print("printed:", count)
    if two_e_output:
        with open(two_e_output, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    return {
        "n_spatial_orbitals": n_spatial_orbitals,
        "n_occ": n_occ,
        "n_vir": n_vir,
        "orbital_energies": orbital_energies,
        "two_electron_terms_written": count,
        "fock_output": fock_output,
        "two_e_output": two_e_output,
        "converged": bool(mf.converged),
    }


  
