"""C2v-irrep-filtered VQE helpers (PennyLane)."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

_C2V_NAME_TO_BITS = {"A1": 0, "A2": 1, "B1": 2, "B2": 3}
_C2V_BITS_TO_NAME = {value: key for key, value in _C2V_NAME_TO_BITS.items()}


def _normalize_irrep_name(name: str) -> str:
    value = str(name).strip().upper()
    aliases = {
        "1": "A1",
        "2": "A2",
        "3": "B1",
        "4": "B2",
        "A_1": "A1",
        "A_2": "A2",
        "B_1": "B1",
        "B_2": "B2",
    }
    value = aliases.get(value, value)
    if value not in _C2V_NAME_TO_BITS:
        raise ValueError(
            f"Invalid C2v irrep `{name}`. Allowed: A1, A2, B1, B2 (or combinations like A1+A2)."
        )
    return value


def _parse_target_irreps(target_irrep) -> list[str]:
    if target_irrep is None:
        return ["A1"]
    if isinstance(target_irrep, str):
        if "+" in target_irrep:
            raw_items = [item for item in target_irrep.split("+") if item.strip()]
        elif "," in target_irrep:
            raw_items = [item for item in target_irrep.split(",") if item.strip()]
        else:
            raw_items = [target_irrep]
    elif isinstance(target_irrep, Iterable):
        raw_items = [str(item) for item in target_irrep]
    else:
        raw_items = [str(target_irrep)]

    parsed = []
    for item in raw_items:
        name = _normalize_irrep_name(item)
        if name not in parsed:
            parsed.append(name)
    if not parsed:
        raise ValueError("At least one target C2v irrep must be specified.")
    return parsed


def _build_active_space_irreps(
    symbols: Sequence[str],
    geometry,
    *,
    charge: int,
    basis: str,
    unit: str,
    active_electrons: int,
    active_orbitals: int,
    point_group: str,
) -> list[str]:
    try:
        import numpy as np
        from pyscf import gto, scf, symm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "C2v symmetry filtering requires numpy and pyscf. Install with:\n"
            "  python -m pip install numpy pyscf"
        ) from exc

    point_group = str(point_group).strip()
    if point_group.upper() != "C2V":
        raise ValueError(
            f"This module currently supports point_group='C2v' only, got `{point_group}`."
        )

    coords = np.asarray(geometry, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("`geometry` must have shape (n_atoms, 3).")
    if len(symbols) != coords.shape[0]:
        raise ValueError("`symbols` length must match geometry rows.")

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
        raise RuntimeError("Symmetry RHF did not converge while building C2v labels.")

    nelecas = int(active_electrons)
    ncas = int(active_orbitals)
    if nelecas <= 0 or ncas <= 0:
        raise ValueError("`active_orbitals` and `active_electrons` must be positive.")

    inactive_electrons = int(mol.nelectron) - nelecas
    if inactive_electrons < 0:
        raise ValueError(
            "active_electrons exceeds total electron count for the requested molecule."
        )
    if inactive_electrons % 2 != 0:
        raise ValueError(
            "active-space partition is incompatible with RHF (odd number of inactive electrons)."
        )
    ncore = inactive_electrons // 2

    nmo = int(mf.mo_coeff.shape[1])
    if ncore + ncas > nmo:
        raise ValueError(
            f"Active-space window exceeds RHF MO count: ncore({ncore}) + ncas({ncas}) > nmo({nmo})."
        )

    active_coeff = mf.mo_coeff[:, ncore : ncore + ncas]
    orb_syms = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, active_coeff)
    return [_normalize_irrep_name(name) for name in orb_syms]


def _det_irrep_from_occ(occ, spatial_orb_irreps: Sequence[str]) -> str:
    bits = 0
    for spin_orb in occ:
        spatial_idx = int(spin_orb) // 2
        if spatial_idx < 0 or spatial_idx >= len(spatial_orb_irreps):
            raise ValueError(
                f"Spin orbital index {spin_orb} maps to spatial index {spatial_idx}, "
                f"outside active-orbital range [0, {len(spatial_orb_irreps) - 1}]."
            )
        bits ^= _C2V_NAME_TO_BITS[spatial_orb_irreps[spatial_idx]]
    return _C2V_BITS_TO_NAME[bits]


def _filter_excitations_by_target_irrep(
    singles,
    doubles,
    *,
    hf_occ: Sequence[int],
    spatial_orb_irreps: Sequence[str],
    target_irreps: Sequence[str],
):
    target_set = set(target_irreps)
    kept_singles = []
    kept_doubles = []

    hf_occ_set = set(int(i) for i in hf_occ)

    for i, a in singles:
        occ = set(hf_occ_set)
        occ.remove(int(i))
        occ.add(int(a))
        det_irrep = _det_irrep_from_occ(sorted(occ), spatial_orb_irreps)
        if det_irrep in target_set:
            kept_singles.append((int(i), int(a)))

    for i, j, a, b in doubles:
        occ = set(hf_occ_set)
        occ.remove(int(i))
        occ.remove(int(j))
        occ.add(int(a))
        occ.add(int(b))
        det_irrep = _det_irrep_from_occ(sorted(occ), spatial_orb_irreps)
        if det_irrep in target_set:
            kept_doubles.append((int(i), int(j), int(a), int(b)))

    return kept_singles, kept_doubles


def gs_exact(
    symbols,
    geometry,
    active_electrons,
    active_orbitals,
    charge,
    method="pyscf",
    basis=None,
    unit=None,
    shots=None,
    max_iter=500,
    amplitudes_outfile=None,
    hamiltonian=None,
    qubits=None,
    target_irrep="A1",
    point_group="C2v",
    active_orbital_irreps: Optional[Sequence[str]] = None,
):
    """Run a C2v-irrep-filtered UCCSD-VQE ground-state calculation.

    Returns the optimized parameter vector, ordered as:
    1) all allowed singles
    2) all allowed doubles
    """
    try:
        import pennylane as qml
        from pennylane import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'pennylane'. Install with:\n"
            "  python -m pip install pennylane pennylane-lightning pyscf"
        ) from exc

    if hamiltonian is None:
        raise ValueError(
            "`hamiltonian` is required. Internal Hamiltonian building has been disabled."
        )
    if basis is None or unit is None:
        raise ValueError("`basis` and `unit` must be provided by the caller.")

    H = hamiltonian
    if qubits is None:
        try:
            qubits = len(H.wires)
        except Exception as exc:
            raise ValueError(
                "Could not infer `qubits` from the provided Hamiltonian. Pass `qubits` explicitly."
            ) from exc
    qubits = int(qubits)

    expected_qubits = 2 * int(active_orbitals)
    if qubits != expected_qubits:
        raise ValueError(
            "C2v irrep filtering assumes a spin-orbital active-space mapping "
            f"with qubits == 2*active_orbitals. Got qubits={qubits}, "
            f"active_orbitals={active_orbitals}."
        )

    if active_orbital_irreps is None:
        spatial_orb_irreps = _build_active_space_irreps(
            symbols=symbols,
            geometry=geometry,
            charge=charge,
            basis=basis,
            unit=unit,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            point_group=point_group,
        )
    else:
        spatial_orb_irreps = [_normalize_irrep_name(name) for name in active_orbital_irreps]
        if len(spatial_orb_irreps) != int(active_orbitals):
            raise ValueError(
                "len(active_orbital_irreps) must equal active_orbitals. "
                f"Got {len(spatial_orb_irreps)} vs {active_orbitals}."
            )

    target_irreps = _parse_target_irreps(target_irrep)

    hf_state = qml.qchem.hf_state(active_electrons, qubits)
    hf_occ = list(range(int(active_electrons)))
    hf_irrep = _det_irrep_from_occ(hf_occ, spatial_orb_irreps)

    print("C2v active-space orbital irreps:", spatial_orb_irreps)
    print("HF determinant irrep:", hf_irrep)
    print("Requested VQE target irrep sector(s):", "+".join(target_irreps))

    def make_device(*, shots_):
        try:
            return qml.device("lightning.qubit", wires=qubits, shots=shots_)
        except Exception:
            return qml.device("default.qubit", wires=qubits, shots=shots_)

    # HF energy
    dev_hf = make_device(shots_=None)

    @qml.qnode(dev_hf)
    def hf_energy():
        qml.BasisState(hf_state, wires=range(qubits))
        return qml.expval(H)

    hf_e = hf_energy()
    print("HF energy:", hf_e)

    all_singles, all_doubles = qml.qchem.excitations(active_electrons, qubits)
    singles, doubles = _filter_excitations_by_target_irrep(
        all_singles,
        all_doubles,
        hf_occ=hf_occ,
        spatial_orb_irreps=spatial_orb_irreps,
        target_irreps=target_irreps,
    )
    print(
        "Symmetry-allowed excitations:",
        f"singles {len(singles)}/{len(all_singles)},",
        f"doubles {len(doubles)}/{len(all_doubles)}",
    )

    if singles or doubles:
        s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
    else:
        s_wires, d_wires = [], []
    params = np.zeros(len(singles) + len(doubles), dtype=float)

    # Device
    dev = make_device(shots_=shots)

    @qml.qnode(dev, interface="autograd", diff_method="best")
    def circuit(params, wires, s_wires, d_wires, hf_state):
        qml.UCCSD(params, wires, s_wires, d_wires, hf_state)
        return qml.expval(H)

    if max_iter < 1:
        raise ValueError("`max_iter` must be >= 1")

    if len(params) == 0:
        energy = hf_e
    else:
        optimizer = qml.GradientDescentOptimizer(stepsize=0.1)
        for step in range(1, int(max_iter) + 1):
            params, energy = optimizer.step_and_cost(
                circuit,
                params,
                wires=range(qubits),
                s_wires=s_wires,
                d_wires=d_wires,
                hf_state=hf_state,
            )
            if step % 10 == 0:
                print(f"VQE iter {step}: energy = {float(energy):.12f}")

    print("\nOptimal parameters:\n", list(params))
    print("UCCSD energy = ", energy)

    # Print amplitudes in exact excitation ordering (singles then doubles)
    print("\nPrinting amplitudes")
    print("Operator\tAmplitude")
    print("++++++++++++++++++++++++++++++")

    n_s = len(singles)
    t1 = params[:n_s]
    t2 = params[n_s:]

    def _canon_pair(p, q):
        """Put mixed-spin pairs in (even, odd) order; return (p2, q2, sign)."""
        p, q = int(p), int(q)
        sign = 1.0
        if (p % 2) != (q % 2):
            if p % 2 == 1:  # (odd, even) -> swap
                p, q = q, p
                sign *= -1.0
        else:
            if p > q:  # keep same-spin pairs ascending (should already be)
                p, q = q, p
                sign *= -1.0
        return p, q, sign

    amp_lines = []
    for (i, j, a, b), amp in zip(doubles, t2):
        a2, b2, s_ab = _canon_pair(a, b)
        i2, j2, s_ij = _canon_pair(i, j)
        line = f"{a2}^ {b2}^ {i2} {j2} \t| {s_ab * s_ij * amp}"
        amp_lines.append(line)
        print(line)

    for (i, a), amp in zip(singles, t1):
        line = f"{a}^ {i} \t| {amp}"
        amp_lines.append(line)
        print(line)

    if amplitudes_outfile:
        with open(amplitudes_outfile, "w", encoding="utf-8") as f:
            f.write(f"Active electrons: {int(active_electrons)}\n")
            f.write(f"Active orbitals: {int(active_orbitals)}\n")
            f.write(f"HF energy: {float(hf_e)}\n")
            f.write(f"UCCSD energy: {float(energy)}\n")
            f.write(f"Point group: C2v\n")
            f.write(f"HF irrep: {hf_irrep}\n")
            f.write(f"Target irreps: {','.join(target_irreps)}\n")
            f.write(
                "Symmetry-allowed excitations: "
                f"singles={len(singles)}, doubles={len(doubles)}\n"
            )
            f.write("\n")
            f.write("Operator\tAmplitude\n")
            f.write("++++++++++++++++++++++++++++++\n")
            f.write("\n".join(amp_lines))
            f.write("\n")

    return params
