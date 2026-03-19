"""CASCI/FCI helpers for building reusable active-space Hamiltonians."""

from __future__ import annotations

from pathlib import Path


def _as_numpy_coordinates(geometry):
    import numpy as np

    coords = np.asarray(geometry, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("`geometry` must have shape (n_atoms, 3).")
    return coords


def build_casci_hamiltonian(
    symbols,
    geometry,
    *,
    active_electrons,
    active_orbitals,
    charge=0,
    basis="sto-3g",
    unit="angstrom",
    n_excited=0,
    casci_output_path=None,
):
    """Build an active-space qubit Hamiltonian from a CASCI reference.

    Returns
    -------
    tuple
        (hamiltonian, qubits, energies) where `energies` contains
        ground + excited CASCI roots.
    """

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'numpy'. Install with:\n  python -m pip install numpy"
        ) from exc

    try:
        import pennylane as qml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'pennylane'. Install with:\n"
            "  python -m pip install pennylane pennylane-lightning pyscf"
        ) from exc

    try:
        from pyscf import ao2mo, gto, mcscf, scf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'pyscf'. Install with:\n  python -m pip install pyscf"
        ) from exc

    coords = _as_numpy_coordinates(geometry)
    if len(symbols) != coords.shape[0]:
        raise ValueError("`symbols` length must match geometry rows.")

    ncas = int(active_orbitals)
    nelecas = int(active_electrons)
    if ncas <= 0 or nelecas <= 0:
        raise ValueError("`active_orbitals` and `active_electrons` must be positive.")

    mol = gto.Mole()
    mol.atom = [[symbols[i], coords[i].tolist()] for i in range(len(symbols))]
    mol.basis = basis
    mol.charge = int(charge)
    mol.unit = unit
    mol.build()

    mf = scf.RHF(mol)
    mf.max_cycle = 200
    mf.conv_tol = 1e-9
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge before CASCI Hamiltonian build.")

    mc = mcscf.CASCI(mf, ncas, nelecas)
    n_roots = int(n_excited) + 1 if int(n_excited) >= 0 else 1
    mc.fcisolver.nroots = n_roots
    energies = np.atleast_1d(mc.kernel()[0]).astype(float)
    excited = energies[1:]

    if casci_output_path:
        out_path = Path(casci_output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out_path, excited, fmt="%.12f")

    h1_cas, core_energy = mc.get_h1cas()
    h2_cas = ao2mo.restore(1, mc.get_h2cas(), ncas)
    h_fermion = qml.qchem.fermionic_observable(
        np.array([float(core_energy)]),
        np.asarray(h1_cas, dtype=float),
        np.asarray(h2_cas, dtype=float),
    )
    h_qubit = qml.jordan_wigner(h_fermion, tol=1.0e-10)
    try:
        h_qubit.simplify()
    except Exception:
        pass

    qubits = len(h_qubit.wires)
    return h_qubit, qubits, energies


def save_default_ch_plus_casci():
    """Preserve original fci.py behavior when run as a script."""

    out_path = Path("outputs/CH+ Trotterized/CASCI_output.txt")
    symbols = ["C", "H"]
    coords = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.261840985],
    ]
    build_casci_hamiltonian(
        symbols,
        coords,
        active_electrons=6,
        active_orbitals=6,
        charge=1,
        basis="6-31g",
        unit="angstrom",
        n_excited=117,
        casci_output_path=out_path,
    )
    print(f"Saved CASCI excited-state energies to {out_path}")


if __name__ == "__main__":
    save_default_ch_plus_casci()
