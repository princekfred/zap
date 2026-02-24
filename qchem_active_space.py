"""Utilities to enforce requested active-space size in PennyLane qchem builds."""

from __future__ import annotations


def _build_pyscf_with_forced_active_space(
    qml,
    symbols,
    coordinates,
    *,
    basis,
    unit,
    charge,
    active_electrons,
    active_orbitals,
    mult,
):
    """Build a PySCF Hamiltonian and explicitly enforce active-space slicing."""
    import numpy as np

    if active_orbitals is None:
        raise RuntimeError("Forced active-space builder requires `active_orbitals`.")

    coords = np.asarray(coordinates, dtype=float).reshape(-1)
    unit_norm = str(unit).strip().lower()
    if unit_norm not in {"bohr", "angstrom"}:
        raise ValueError("`unit` must be either 'bohr' or 'angstrom'.")
    if unit_norm == "angstrom":
        coords = coords / qml.qchem.openfermion_pyscf.bohr_angs

    core_const, one_mo, two_mo = qml.qchem.openfermion_pyscf._pyscf_integrals(
        symbols,
        coords,
        charge=charge,
        mult=mult,
        basis=basis,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )

    n_active = int(active_orbitals)
    if one_mo.shape[0] != n_active:
        # Upstream PySCF path may skip slicing when no core orbitals are frozen.
        one_mo = one_mo[:n_active, :n_active]
        two_mo = two_mo[:n_active, :n_active, :n_active, :n_active]

    hf = qml.qchem.fermionic_observable(core_const, one_mo, two_mo)
    h_pl = qml.jordan_wigner(hf, tol=1.0e-10)
    try:
        h_pl.simplify()
    except Exception:
        pass
    return h_pl, len(h_pl.wires)


def build_molecular_hamiltonian_enforcing_active_space(
    qml,
    symbols,
    coordinates,
    *,
    basis,
    method,
    unit,
    charge,
    active_electrons,
    active_orbitals,
    mult=1,
):
    """Build molecular Hamiltonian and ensure qubit count matches active orbitals.

    Returns
    -------
    tuple
        (Hamiltonian, qubits, used_method)
    """
    target_qubits = None if active_orbitals is None else 2 * int(active_orbitals)

    def _build(method_name):
        return qml.qchem.molecular_hamiltonian(
            symbols,
            coordinates,
            basis=basis,
            method=method_name,
            unit=unit,
            charge=charge,
            mult=mult,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
        )

    hamiltonian, qubits = _build(method)
    if target_qubits is None or qubits == target_qubits:
        return hamiltonian, qubits, method

    # PennyLane's PySCF path can return full-space qubits for some closed-shell cations.
    # Rebuild directly from PySCF integrals and enforce requested active slicing.
    if method == "pyscf":
        h_fix, q_fix = _build_pyscf_with_forced_active_space(
            qml,
            symbols,
            coordinates,
            basis=basis,
            unit=unit,
            charge=charge,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            mult=mult,
        )
        if q_fix == target_qubits:
            print(
                "Active-space safeguard: method='pyscf' returned",
                qubits,
                "qubits; expected",
                target_qubits,
                ". Rebuilding with forced PySCF active-space slicing.",
            )
            return h_fix, q_fix, "pyscf"

    raise RuntimeError(
        "Requested active space is not honored: expected "
        f"{target_qubits} qubits from active_orbitals={active_orbitals}, got {qubits}."
    )
