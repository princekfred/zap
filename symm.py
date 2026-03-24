"""Symmetry analysis helpers for QSC-EOM determinant-space eigenvectors."""

from __future__ import annotations

from typing import Sequence


def _normalize_irrep_name(name: str) -> str:
    value = str(name).strip().upper()
    aliases = {
        "A_1": "A1",
        "A_2": "A2",
        "B_1": "B1",
        "B_2": "B2",
    }
    return aliases.get(value, value)


def _mul_irreps_by_xor(irrep_ids: Sequence[int]) -> int:
    if not irrep_ids:
        raise ValueError("irrep_ids must not be empty.")
    acc = int(irrep_ids[0])
    for ir in irrep_ids[1:]:
        acc ^= int(ir)
    return int(acc)


def build_active_space_irreps(
    symbols,
    geometry,
    *,
    charge,
    basis,
    unit,
    active_electrons,
    active_orbitals,
    point_group="C2v",
):
    """Return `(groupname, active_orbital_irreps)` from symmetry RHF."""
    try:
        import numpy as np
        from pyscf import gto, scf, symm as pyscf_symm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency for symmetry analysis. Install with:\n"
            "  python -m pip install numpy pyscf"
        ) from exc

    coords = np.asarray(geometry, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("geometry must have shape (n_atoms, 3)")
    if len(symbols) != coords.shape[0]:
        raise ValueError("len(symbols) must match geometry rows")

    mol = gto.Mole()
    mol.atom = [[symbols[i], coords[i].tolist()] for i in range(len(symbols))]
    mol.basis = basis
    mol.charge = int(charge)
    mol.unit = unit
    mol.symmetry = str(point_group)
    mol.build()

    mf = scf.RHF(mol)
    mf.max_cycle = 200
    mf.conv_tol = 1e-9
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("Symmetry RHF did not converge while building orbital irreps.")

    ncas = int(active_orbitals)
    nelecas = int(active_electrons)
    inactive_electrons = int(mol.nelectron) - nelecas
    if inactive_electrons < 0:
        raise ValueError("active_electrons exceeds total electron count")
    if inactive_electrons % 2 != 0:
        raise ValueError("Odd inactive electron count is incompatible with RHF slicing")
    ncore = inactive_electrons // 2

    active_coeff = mf.mo_coeff[:, ncore : ncore + ncas]
    if active_coeff.shape[1] != ncas:
        raise ValueError("Active-space window exceeds available RHF orbitals")

    orb_syms = pyscf_symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, active_coeff)
    active_irreps = [_normalize_irrep_name(name) for name in orb_syms]
    return str(mol.groupname), active_irreps


def analyze_qsceom_eigenvector(
    vector,
    det_list,
    *,
    symbols,
    geometry,
    active_electrons,
    active_orbitals,
    charge,
    basis,
    unit,
    point_group="C2v",
    amp_cutoff=1e-10,
):
    """Decompose a QSC-EOM eigenvector into symmetry weights by determinant."""
    try:
        import numpy as np
        from pyscf import symm as pyscf_symm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency for symmetry analysis. Install with:\n"
            "  python -m pip install numpy pyscf"
        ) from exc

    vec = np.asarray(vector)
    if vec.ndim != 1:
        raise ValueError("vector must be 1D")
    if len(vec) != len(det_list):
        raise ValueError(f"len(vector)={len(vec)} does not match len(det_list)={len(det_list)}")

    groupname, active_irreps = build_active_space_irreps(
        symbols,
        geometry,
        charge=charge,
        basis=basis,
        unit=unit,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        point_group=point_group,
    )

    weights = {}
    total = 0.0
    for coeff, det in zip(vec, det_list):
        if abs(coeff) < float(amp_cutoff):
            continue

        det_irrep_ids = []
        for spin_orb in det:
            spatial = int(spin_orb) // 2
            if spatial < 0 or spatial >= len(active_irreps):
                raise ValueError(
                    f"Spin orbital index {spin_orb} maps to spatial index {spatial}, "
                    f"outside active range [0, {len(active_irreps)-1}]"
                )
            det_irrep_ids.append(pyscf_symm.irrep_name2id(groupname, active_irreps[spatial]))

        ir_id = _mul_irreps_by_xor(det_irrep_ids)
        ir_name = str(pyscf_symm.irrep_id2name(groupname, ir_id))
        w = float(abs(coeff) ** 2)
        weights[ir_name] = weights.get(ir_name, 0.0) + w
        total += w

    if total > 0.0:
        for ir in list(weights.keys()):
            weights[ir] = weights[ir] / total
        dominant = max(weights, key=weights.get)
    else:
        dominant = "unknown"

    return {
        "groupname": groupname,
        "active_orbital_irreps": active_irreps,
        "weights_by_irrep": weights,
        "dominant_irrep": dominant,
    }


def format_weights(weights_by_irrep: dict[str, float]) -> str:
    if not weights_by_irrep:
        return ""
    items = sorted(weights_by_irrep.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{ir}:{wt:.6f}" for ir, wt in items)


def print_sym_info_old_format(weights_by_irrep: dict[str, float], groupname: str) -> str:
    """Print symmetry decomposition in the legacy format used by `print_sym_info`."""
    from pyscf import symm as pyscf_symm

    weights = {
        int(pyscf_symm.irrep_name2id(groupname, ir_name)): float(weight)
        for ir_name, weight in weights_by_irrep.items()
    }

    total = sum(weights.values())
    print(weights.items())
    print("\n=== Symmetry Decomposition ===")
    for ir, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(ir)
        print(f"{pyscf_symm.irrep_id2name(groupname, ir):9s} : {w/total:.4f}")
    print(groupname)
    dominant = max(weights, key=weights.get)
    info = pyscf_symm.irrep_id2name(groupname, dominant)
    print("\nDominant symmetry:", pyscf_symm.irrep_id2name(groupname, dominant))
    return str(info)
