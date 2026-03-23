"""C2v-symmetry CASCI/FCI helpers for reusable active-space Hamiltonians."""

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
    point_group="C2v",
    n_excited=None,
    casci_output_path=None,
):
    """Build an active-space qubit Hamiltonian from a CASCI reference.

    Returns
    -------
    tuple
        (hamiltonian, qubits, energies) where `energies` contains
        ground + excited CASCI roots.

    Notes
    -----
    This C2v variant enforces ``mol.symmetry = "C2v"`` by default.
    If `n_excited` is `None`, all roots in the fixed-Sz active-space FCI
    determinant manifold are requested automatically.
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
        from pyscf.fci import cistring
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

    point_group = str(point_group).strip()
    if point_group.upper() != "C2V":
        raise ValueError(
            f"casci_c2v supports point_group='C2v' only, got `{point_group}`."
        )

    mol = gto.Mole()
    mol.atom = [[symbols[i], coords[i].tolist()] for i in range(len(symbols))]
    mol.basis = basis
    mol.charge = int(charge)
    mol.unit = unit
    mol.symmetry = "C2v"
    mol.build()

    mf = scf.RHF(mol)
    mf.max_cycle = 200
    mf.conv_tol = 1e-9
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge before CASCI Hamiltonian build.")

    mc = mcscf.CASCI(mf, ncas, nelecas)
    n_alpha = (nelecas + int(mol.spin)) // 2
    n_beta = nelecas - n_alpha
    max_roots = int(cistring.num_strings(ncas, n_alpha) * cistring.num_strings(ncas, n_beta))

    if n_excited is None:
        n_roots = max_roots
    else:
        n_roots = int(n_excited) + 1 if int(n_excited) >= 0 else 1
        if n_roots > max_roots:
            n_roots = max_roots

    mc.fcisolver.nroots = n_roots
    energies = np.atleast_1d(mc.kernel()[0]).astype(float)
    excited = energies[0:]

    if casci_output_path:
        out_path = Path(casci_output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "point_group C2v\n"
            f"converged_scf_energy_hartree {float(mf.e_tot):.12f}\n"
            "casci_excited_state_energies_hartree"
        )
        np.savetxt(out_path, excited, fmt="%.12f", header=header, comments="# ")

    h1_cas, core_energy = mc.get_h1cas()
    h2_cas = ao2mo.restore(1, mc.get_h2cas(), ncas)
    # Match PennyLane's expected physicists' ordering for two-electron tensors.
    h2_cas = np.swapaxes(h2_cas, 1, 3)
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


def build_casci_hamiltonian_from_problem(
    problem_cfg,
    *,
    point_group="C2v",
    n_excited=None,
    casci_output_path=None,
):
    """Build CASCI Hamiltonian from a run-script problem dictionary."""

    required = {"symbols", "geometry", "active_electrons", "active_orbitals"}
    missing = sorted(required.difference(problem_cfg))
    if missing:
        raise KeyError(f"Missing required problem keys for CASCI Hamiltonian: {missing}")

    return build_casci_hamiltonian(
        problem_cfg["symbols"],
        problem_cfg["geometry"],
        active_electrons=problem_cfg["active_electrons"],
        active_orbitals=problem_cfg["active_orbitals"],
        charge=problem_cfg.get("charge", 0),
        basis=problem_cfg.get("basis", "sto-3g"),
        unit=problem_cfg.get("unit", "angstrom"),
        point_group=point_group,
        n_excited=n_excited,
        casci_output_path=casci_output_path,
    )

if __name__ == "__main__":
    raise SystemExit(
        "casci_c2v.py is a library module. Call "
        "`build_casci_hamiltonian_from_problem(...)` from a run script."
    )
