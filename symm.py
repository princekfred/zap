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


def qsceom_detvec_to_r1r2(
    vector,
    det_list,
    *,
    active_electrons: int,
    active_orbitals: int,
    ncore: int = 0,
    nocc_ze: int | None = None,
    nmo_ze: int | None = None,
    combine: str = "norm",
):
    """Convert determinant-basis QSC-EOM coefficients into ze-style ``R1/R2`` layout.

    Parameters
    ----------
    vector
        1D QSC-EOM eigenvector in determinant basis (same order as ``det_list``).
    det_list
        Determinant occupations in spin-orbital indices (for example from ``exc.inite``).
    active_electrons, active_orbitals
        Active-space definition used to build ``det_list``.
    ncore
        Number of inactive core spatial orbitals. For all-electron active spaces use ``0``.
    nocc_ze, nmo_ze
        Target ze-layout occupied/MO counts. Defaults to active-space layout
        (``nocc_ze = ncore + active_electrons//2``, ``nmo_ze = nocc_ze + active_nvir``).
    combine
        ``"norm"`` aggregates spin-resolved determinants by root-sum-square (preserves
        symmetry weights). ``"sum"`` aggregates coherently by plain coefficient sum.
    """
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency for R1/R2 conversion. Install with:\n"
            "  python -m pip install numpy"
        ) from exc

    vec = np.asarray(vector)
    if vec.ndim != 1:
        raise ValueError("vector must be 1D")
    if len(vec) != len(det_list):
        raise ValueError(f"len(vector)={len(vec)} does not match len(det_list)={len(det_list)}")

    nele = int(active_electrons)
    norb = int(active_orbitals)
    if nele <= 0 or norb <= 0:
        raise ValueError("active_electrons and active_orbitals must be positive")
    if nele % 2 != 0:
        raise ValueError("ze-style R1/R2 layout requires an even active_electrons count.")

    nocc_active = nele // 2
    nvir_active = norb - nocc_active
    if nvir_active <= 0:
        raise ValueError("active_orbitals must be greater than active_electrons//2.")

    ncore = int(ncore)
    if ncore < 0:
        raise ValueError("ncore must be >= 0")

    if nocc_ze is None:
        nocc_ze = ncore + nocc_active
    if nmo_ze is None:
        nmo_ze = int(nocc_ze) + nvir_active

    nocc_ze = int(nocc_ze)
    nmo_ze = int(nmo_ze)
    nvir_ze = nmo_ze - nocc_ze

    if nocc_ze < ncore + nocc_active:
        raise ValueError(
            "nocc_ze is too small for the requested active-space embedding: "
            f"need >= {ncore + nocc_active}, got {nocc_ze}."
        )
    if nvir_ze < nvir_active:
        raise ValueError(
            "nmo_ze - nocc_ze is too small for active virtual orbitals: "
            f"need >= {nvir_active}, got {nvir_ze}."
        )

    mode = str(combine).strip().lower()
    if mode not in {"norm", "sum"}:
        raise ValueError("combine must be 'norm' or 'sum'")

    if mode == "norm":
        r1_acc = np.zeros((nocc_ze, nvir_ze), dtype=float)
        r2_acc = np.zeros((nocc_ze, nocc_ze, nvir_ze, nvir_ze), dtype=float)
    else:
        r1_acc = np.zeros((nocc_ze, nvir_ze), dtype=complex)
        r2_acc = np.zeros((nocc_ze, nocc_ze, nvir_ze, nvir_ze), dtype=complex)

    hf_occ = list(range(nele))
    hf_set = set(hf_occ)
    singles = 0
    doubles = 0
    skipped = 0

    for coeff, det in zip(vec, det_list):
        det_int = [int(x) for x in det]
        det_set = set(det_int)
        holes = [h for h in hf_occ if h not in det_set]
        particles = sorted(p for p in det_int if p not in hf_set)
        rank = len(holes)

        if rank == 1 and len(particles) == 1:
            i_active = int(holes[0]) // 2
            a_active = int(particles[0]) // 2 - nocc_active
            i = ncore + i_active
            a = a_active
            if i < 0 or i >= nocc_ze or a < 0 or a >= nvir_ze:
                raise ValueError(
                    f"Mapped single index out of bounds for ze layout: i={i}, a={a}, "
                    f"shape=({nocc_ze}, {nvir_ze})."
                )
            if mode == "norm":
                r1_acc[i, a] += float(abs(coeff) ** 2)
            else:
                r1_acc[i, a] += coeff
            singles += 1
            continue

        if rank == 2 and len(particles) == 2:
            holes = sorted(int(h) for h in holes)
            i_active = holes[0] // 2
            j_active = holes[1] // 2
            a_active = int(particles[0]) // 2 - nocc_active
            b_active = int(particles[1]) // 2 - nocc_active
            i = ncore + i_active
            j = ncore + j_active
            a = a_active
            b = b_active
            if (
                i < 0
                or i >= nocc_ze
                or j < 0
                or j >= nocc_ze
                or a < 0
                or a >= nvir_ze
                or b < 0
                or b >= nvir_ze
            ):
                raise ValueError(
                    "Mapped double index out of bounds for ze layout: "
                    f"(i,j,a,b)=({i},{j},{a},{b}), "
                    f"shape=({nocc_ze}, {nocc_ze}, {nvir_ze}, {nvir_ze})."
                )
            if mode == "norm":
                r2_acc[i, j, a, b] += float(abs(coeff) ** 2)
            else:
                r2_acc[i, j, a, b] += coeff
            doubles += 1
            continue

        skipped += 1

    if mode == "norm":
        r1 = np.sqrt(r1_acc)
        r2 = np.sqrt(r2_acc)
    else:
        r1 = r1_acc
        r2 = r2_acc

    packed = np.concatenate([r1.ravel(), r2.ravel()])
    return {
        "R": packed,
        "R1": r1,
        "R2": r2,
        "nocc": nocc_ze,
        "nvir": nvir_ze,
        "nmo": nmo_ze,
        "ncore": ncore,
        "combine": mode,
        "counts": {"singles": singles, "doubles": doubles, "skipped": skipped},
    }


def analyze_qsceom_eigenvector_r1r2_precomputed(
    vector,
    det_list,
    *,
    groupname,
    active_orbital_irreps,
    active_electrons,
    active_orbitals,
    amp_cutoff=1e-10,
    combine="norm",
):
    """ze-style symmetry decomposition using precomputed active-space irreps."""
    try:
        import numpy as np
        from pyscf import symm as pyscf_symm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency for symmetry analysis. Install with:\n"
            "  python -m pip install numpy pyscf"
        ) from exc

    active_irreps = [_normalize_irrep_name(name) for name in active_orbital_irreps]
    conv = qsceom_detvec_to_r1r2(
        vector,
        det_list,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        ncore=0,
        nocc_ze=active_electrons // 2,
        nmo_ze=active_orbitals,
        combine=combine,
    )

    nocc = int(conv["nocc"])
    nvir = int(conv["nvir"])
    r1 = np.asarray(conv["R1"])
    r2 = np.asarray(conv["R2"])

    weights = {}
    total = 0.0

    for i in range(nocc):
        for a in range(nvir):
            amp = r1[i, a]
            if abs(amp) < float(amp_cutoff):
                continue
            ir_i = pyscf_symm.irrep_name2id(groupname, active_irreps[i])
            ir_a = pyscf_symm.irrep_name2id(groupname, active_irreps[nocc + a])
            ir_name = str(pyscf_symm.irrep_id2name(groupname, int(ir_a) ^ int(ir_i)))
            w = float(abs(amp) ** 2)
            weights[ir_name] = weights.get(ir_name, 0.0) + w
            total += w

    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    amp = r2[i, j, a, b]
                    if abs(amp) < float(amp_cutoff):
                        continue
                    ir = (
                        int(pyscf_symm.irrep_name2id(groupname, active_irreps[nocc + a]))
                        ^ int(pyscf_symm.irrep_name2id(groupname, active_irreps[nocc + b]))
                        ^ int(pyscf_symm.irrep_name2id(groupname, active_irreps[i]))
                        ^ int(pyscf_symm.irrep_name2id(groupname, active_irreps[j]))
                    )
                    ir_name = str(pyscf_symm.irrep_id2name(groupname, ir))
                    w = float(abs(amp) ** 2)
                    weights[ir_name] = weights.get(ir_name, 0.0) + w
                    total += w

    if total > 0.0:
        for ir in list(weights.keys()):
            weights[ir] = weights[ir] / total
        dominant = max(weights, key=weights.get)
    else:
        dominant = "unknown"

    return {
        "groupname": str(groupname),
        "active_orbital_irreps": active_irreps,
        "weights_by_irrep": weights,
        "dominant_irrep": dominant,
        "r_layout_counts": conv["counts"],
    }


def analyze_qsceom_eigenvector_r1r2(
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
    combine="norm",
):
    """Symmetry decomposition via converted ze-style R1/R2 layout (no ze.py dependency)."""
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
    return analyze_qsceom_eigenvector_r1r2_precomputed(
        vector,
        det_list,
        groupname=groupname,
        active_orbital_irreps=active_irreps,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        amp_cutoff=amp_cutoff,
        combine=combine,
    )


def summarize_lowest_qsceom_roots_r1r2(
    eigvals,
    eigvecs,
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
    n_roots=25,
    amp_cutoff=1e-10,
    combine="norm",
):
    """Summarize dominant irreps for the lowest-energy QSC-EOM roots (ze-style R1/R2)."""
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency for symmetry analysis. Install with:\n"
            "  python -m pip install numpy"
        ) from exc

    evals = np.asarray(eigvals)
    evecs = np.asarray(eigvecs)
    if evals.ndim != 1:
        raise ValueError("eigvals must be 1D")
    if evecs.ndim != 2:
        raise ValueError("eigvecs must be 2D")
    if evecs.shape[0] != len(det_list):
        raise ValueError(
            f"eigvecs first dimension {evecs.shape[0]} must match len(det_list)={len(det_list)}"
        )
    if evecs.shape[1] != evals.shape[0]:
        raise ValueError(
            "eigvecs second dimension must match number of eigenvalues: "
            f"{evecs.shape[1]} vs {evals.shape[0]}"
        )

    limit = max(int(n_roots), 0)
    n_take = min(limit, evals.shape[0], evecs.shape[1])
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

    roots = []
    for root_idx in range(n_take):
        info = analyze_qsceom_eigenvector_r1r2_precomputed(
            np.asarray(evecs[:, root_idx]),
            det_list,
            groupname=groupname,
            active_orbital_irreps=active_irreps,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            amp_cutoff=amp_cutoff,
            combine=combine,
        )
        energy_raw = evals[root_idx]
        try:
            energy = float(np.real(energy_raw))
        except Exception:
            energy = float(energy_raw)
        roots.append(
            {
                "root_index": int(root_idx),
                "energy": energy,
                "dominant_irrep": str(info["dominant_irrep"]),
                "weights_by_irrep": dict(info["weights_by_irrep"]),
            }
        )

    return {
        "groupname": str(groupname),
        "active_orbital_irreps": list(active_irreps),
        "roots": roots,
    }


def _legacy_symmetry_report(weights_by_irrep: dict[str, float], groupname: str) -> tuple[str, str]:
    """Return `(report_text, dominant_irrep)` in the legacy ze.py print format."""
    from pyscf import symm as pyscf_symm

    weights = {
        int(pyscf_symm.irrep_name2id(groupname, ir_name)): float(weight)
        for ir_name, weight in weights_by_irrep.items()
    }

    total = sum(weights.values())
    lines = [str(weights.items()), "", "=== Symmetry Decomposition ==="]
    for ir, w in sorted(weights.items(), key=lambda x: -x[1]):
        lines.append(str(ir))
        frac = (w / total) if total else 0.0
        lines.append(f"{pyscf_symm.irrep_id2name(groupname, ir):9s} : {frac:.4f}")
    lines.append(groupname)

    if weights:
        dominant = max(weights, key=weights.get)
        info = str(pyscf_symm.irrep_id2name(groupname, dominant))
    else:
        info = "unknown"

    lines.extend(["", f"Dominant symmetry: {info}"])
    return "\n".join(lines), info


def format_sym_info_(weights_by_irrep: dict[str, float], groupname: str) -> str:
    """Format symmetry decomposition exactly like ze.py legacy terminal output."""
    report, _ = _legacy_symmetry_report(weights_by_irrep, groupname)
    return report + "\n"


def print_sym_info_(weights_by_irrep: dict[str, float], groupname: str) -> str:
    """Print symmetry decomposition in the legacy format used by `print_sym_info`."""
    report, info = _legacy_symmetry_report(weights_by_irrep, groupname)
    print(report)
    return info
