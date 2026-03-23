"""C2v-irrep-filtered QSC-EOM solver.

This module mirrors ``qsceom.py`` but filters the QSC-EOM determinant
manifold by C2v irreps (A1/A2/B1/B2) before building and diagonalizing the
M-matrix.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Iterable, Optional, Sequence

try:
    from mpi4py import MPI as _MPI
except ImportError:  # pragma: no cover
    _MPI = None

qml = None
np = None
_inite = None

_C2V_NAME_TO_BITS = {"A1": 0, "A2": 1, "B1": 2, "B2": 3}
_C2V_BITS_TO_NAME = {value: key for key, value in _C2V_NAME_TO_BITS.items()}


def _require_quantum_deps():
    global qml, np
    if qml is not None and np is not None:
        return
    try:
        import pennylane as _qml
        from pennylane import numpy as _np
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "qsc_eom requires PennyLane and a quantum chemistry backend. "
            "Install with: `pip install pennylane pyscf` "
            "(and optionally `pip install pennylane-lightning`)."
        ) from exc
    qml = _qml
    np = _np


def _load_inite():
    global _inite
    if _inite is not None:
        return _inite

    try:
        from .exc import inite as _inite_local

        _inite = _inite_local
        return _inite
    except Exception:
        pass

    try:
        from exc import inite as _inite_local

        _inite = _inite_local
        return _inite
    except Exception:
        pass

    exc_path = Path(__file__).with_name("exc.py")
    spec = importlib.util.spec_from_file_location("qsceom_c2v_exc", exc_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load inite() helper from {exc_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _inite = mod.inite
    return _inite


def _normalize_shots(shots: Optional[int]) -> Optional[int]:
    if shots is None or shots == 0:
        return None
    if shots < 0:
        raise ValueError("shots must be >= 0")
    return int(shots)


def _make_device(qubits: int, shots: Optional[int]):
    _require_quantum_deps()
    try:
        return qml.device("lightning.qubit", wires=qubits, shots=shots)
    except Exception:
        return qml.device("default.qubit", wires=qubits, shots=shots)


def _mpi_context():
    if _MPI is None:
        return None, 1, 0, None
    comm = _MPI.COMM_WORLD
    return comm, comm.Get_size(), comm.Get_rank(), _MPI.SUM


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
    coordinates,
    *,
    charge: int,
    basis: str,
    unit: str,
    active_electrons: int,
    active_orbitals: int,
    point_group: str,
) -> list[str]:
    try:
        from pyscf import gto, scf, symm
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "C2v-irrep filtering requires PySCF. Install with: `pip install pyscf`."
        ) from exc

    point_group = str(point_group).strip()
    if point_group.upper() != "C2V":
        raise ValueError(
            f"This module currently supports point_group='C2v' only, got `{point_group}`."
        )

    mol = gto.Mole()
    mol.atom = [
        [symbols[i], [float(x) for x in coordinates[i]]]
        for i in range(len(symbols))
    ]
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
        raise ValueError("active_electrons and active_orbitals must be positive.")

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


def _filter_configs_by_irrep(
    excitation_configs,
    *,
    target_irreps: Sequence[str],
    spatial_orb_irreps: Sequence[str],
):
    keep_indices = []
    kept_configs = []
    kept_irreps = []
    for idx, occ in enumerate(excitation_configs):
        occ_irrep = _det_irrep_from_occ(occ, spatial_orb_irreps)
        if occ_irrep in target_irreps:
            keep_indices.append(idx)
            kept_configs.append(occ)
            kept_irreps.append(occ_irrep)
    return keep_indices, kept_configs, kept_irreps


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


def qsc_eom(
    symbols: Sequence[str],
    coordinates,
    active_electrons: int,
    active_orbitals: int,
    charge: int,
    params,
    ash_excitation=None,
    shots: int = 0,
    method: str = "pyscf",
    basis: Optional[str] = None,
    unit: Optional[str] = None,
    hamiltonian=None,
    qubits: Optional[int] = None,
    target_irrep="A1",
    ansatz_target_irrep="A1",
    point_group: str = "C2v",
    active_orbital_irreps: Optional[Sequence[str]] = None,
    return_metadata: bool = False,
):
    """Build and diagonalize a C2v-irrep-filtered QSC-EOM M-matrix.

    Parameters
    ----------
    target_irrep
        One C2v irrep (``"A1"``, ``"A2"``, ``"B1"``, ``"B2"``), a combination
        string (for example ``"A1+A2"``), or an iterable of irreps.
    ansatz_target_irrep
        C2v irrep sector(s) used to define the UCCSD parameterization that
        produced ``params``. This must match the VQE ansatz symmetry filter.
    active_orbital_irreps
        Optional explicit active-space irrep labels in spinless orbital order
        (length ``active_orbitals``). If omitted, labels are built from a C2v
        RHF run with the same molecular inputs.

    Returns
    -------
    tuple
        ``(eigvals, eigvecs)`` sorted by ascending eigenvalue.
    """

    _require_quantum_deps()
    inite = _load_inite()
    norm_shots = _normalize_shots(shots)
    params = np.asarray(params)
    coordinates = np.asarray(coordinates, dtype=float)

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape (n_atoms, 3)")
    if len(symbols) != coordinates.shape[0]:
        raise ValueError("len(symbols) must match number of coordinate rows")

    if params.ndim != 1:
        raise ValueError("params must be a 1D array-like")
    if ash_excitation is not None and len(ash_excitation) != len(params):
        raise ValueError("len(ash_excitation) must match len(params)")
    if basis is None or unit is None:
        raise ValueError("`basis` and `unit` must be provided by the caller.")

    if hamiltonian is None:
        raise ValueError(
            "`hamiltonian` is required. Internal Hamiltonian building has been disabled."
        )

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

    target_irreps = _parse_target_irreps(target_irrep)
    ansatz_irreps = _parse_target_irreps(ansatz_target_irrep)
    if active_orbital_irreps is None:
        spatial_orb_irreps = _build_active_space_irreps(
            symbols=symbols,
            coordinates=coordinates,
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

    all_singles, all_doubles = qml.qchem.excitations(active_electrons, qubits)
    hf_occ = list(range(int(active_electrons)))
    ansatz_singles, ansatz_doubles = _filter_excitations_by_target_irrep(
        all_singles,
        all_doubles,
        hf_occ=hf_occ,
        spatial_orb_irreps=spatial_orb_irreps,
        target_irreps=ansatz_irreps,
    )
    expected_params = len(ansatz_singles) + len(ansatz_doubles)
    if params.size != expected_params:
        raise ValueError(
            "QSC-EOM ansatz parameter length mismatch. "
            f"Expected {expected_params} parameters from ansatz_target_irrep={ansatz_irreps}, "
            f"got {params.size}. "
            "Use the same symmetry sector in VQE and QSC-EOM ansatz construction."
        )

    if ansatz_singles or ansatz_doubles:
        s_wires, d_wires = qml.qchem.excitations_to_wires(ansatz_singles, ansatz_doubles)
    else:
        s_wires, d_wires = [], []
    wires = range(qubits)

    null_state = np.zeros(qubits, int)
    all_excitation_configs = inite(active_electrons, qubits)

    keep_idx, excitation_configs, config_irreps = _filter_configs_by_irrep(
        all_excitation_configs,
        target_irreps=target_irreps,
        spatial_orb_irreps=spatial_orb_irreps,
    )

    if not excitation_configs:
        raise ValueError(
            "No QSC-EOM determinants remain after C2v irrep filtering. "
            f"Requested target_irrep={target_irreps}."
        )

    dev = _make_device(qubits, norm_shots)

    @qml.qnode(dev)
    def circuit_d(params, occ, wires, s_wires, d_wires, hf_state):
        for w in occ:
            qml.X(wires=w)
        qml.UCCSD(params, wires, s_wires, d_wires, hf_state)
        return qml.expval(H)

    @qml.qnode(dev)
    def circuit_od(params, occ1, occ2, wires, s_wires, d_wires, hf_state):
        for w in occ1:
            qml.X(wires=w)

        first = -1
        for v in occ2:
            if v not in occ1:
                if first == -1:
                    first = v
                    qml.Hadamard(wires=v)
                else:
                    qml.CNOT(wires=[first, v])
        for v in occ1:
            if v not in occ2:
                if first == -1:
                    first = v
                    qml.Hadamard(wires=v)
                else:
                    qml.CNOT(wires=[first, v])

        qml.UCCSD(params, wires, s_wires, d_wires, hf_state)
        return qml.expval(H)

    comm, size, rank, mpi_sum = _mpi_context()
    mat_size = len(excitation_configs)

    m_diag_local = np.zeros(mat_size)
    for i in range(rank, mat_size, size):
        m_diag_local[i] = circuit_d(
            params, excitation_configs[i], wires, s_wires, d_wires, null_state
        )

    if comm is None:
        m_diag = m_diag_local
    else:
        m_diag = np.zeros(mat_size)
        comm.Allreduce(m_diag_local, m_diag, op=mpi_sum)

    m_local = np.zeros((mat_size, mat_size))
    flat_idx = 0
    for i in range(mat_size):
        for j in range(i + 1):
            if flat_idx % size == rank:
                if i == j:
                    m_tmp = m_diag[i]
                else:
                    m_tmp = (
                        circuit_od(
                            params,
                            excitation_configs[i],
                            excitation_configs[j],
                            wires,
                            s_wires,
                            d_wires,
                            null_state,
                        )
                        - m_diag[i] / 2.0
                        - m_diag[j] / 2.0
                    )
                m_local[i, j] = m_tmp
                m_local[j, i] = m_tmp
            flat_idx += 1

    if comm is None:
        m_matrix = m_local
    else:
        m_matrix = np.zeros_like(m_local)
        comm.Allreduce(m_local, m_matrix, op=mpi_sum)

    eigvals, eigvecs = np.linalg.eigh(m_matrix)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    if return_metadata:
        metadata = {
            "target_irreps": list(target_irreps),
            "ansatz_target_irreps": list(ansatz_irreps),
            "point_group": "C2v",
            "spatial_orbital_irreps": list(spatial_orb_irreps),
            "selected_config_indices": list(keep_idx),
            "selected_config_irreps": list(config_irreps),
            "selected_excitation_configs": [list(cfg) for cfg in excitation_configs],
            "ansatz_singles_count": int(len(ansatz_singles)),
            "ansatz_doubles_count": int(len(ansatz_doubles)),
            "total_configs_before_filter": int(len(all_excitation_configs)),
            "total_configs_after_filter": int(len(excitation_configs)),
        }
        return eigvals, eigvecs, metadata

    return eigvals, eigvecs


def ee_exact(
    symbols,
    geometry,
    active_electrons,
    active_orbitals,
    charge,
    params,
    shots=0,
    method="pyscf",
    basis=None,
    unit=None,
    state_idx=1,
    r1r2_outfile="out_r1_r2.txt",
    hamiltonian=None,
    qubits=None,
    target_irrep="A1",
    ansatz_target_irrep="A1",
    point_group="C2v",
    active_orbital_irreps=None,
):
    """Compatibility wrapper with C2v irrep filtering for QSC-EOM."""
    _require_quantum_deps()

    eigvals, eigvecs, metadata = qsc_eom(
        symbols=symbols,
        coordinates=geometry,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        charge=charge,
        params=params,
        shots=shots,
        method=method,
        basis=basis,
        unit=unit,
        hamiltonian=hamiltonian,
        qubits=qubits,
        target_irrep=target_irrep,
        ansatz_target_irrep=ansatz_target_irrep,
        point_group=point_group,
        active_orbital_irreps=active_orbital_irreps,
        return_metadata=True,
    )

    if state_idx < 0 or state_idx >= len(eigvals):
        raise ValueError(
            f"`state_idx` must be between 0 and {len(eigvals) - 1}, got {state_idx}."
        )

    vector = eigvecs[:, state_idx]
    det_list = metadata["selected_excitation_configs"]
    det_irreps = metadata["selected_config_irreps"]
    hf_state = np.array(range(active_electrons))

    if r1r2_outfile:
        lines = [
            "R1/R2",
            "Excitations | Coefficients | C2v irrep",
            f"# target_irreps: {', '.join(metadata['target_irreps'])}",
            f"# ansatz_target_irreps: {', '.join(metadata['ansatz_target_irreps'])}",
            f"# ansatz_singles: {metadata['ansatz_singles_count']}",
            f"# ansatz_doubles: {metadata['ansatz_doubles_count']}",
            f"# selected_configs: {metadata['total_configs_after_filter']}/"
            f"{metadata['total_configs_before_filter']}",
        ]
        for det_idx, coeff in enumerate(vector):
            det = det_list[det_idx]
            holes = [hf for hf in hf_state if hf not in det]
            particles = [virt for virt in det if virt not in hf_state]
            labels = [f"{p}^ {h}" for p, h in zip(particles, holes)]
            label = "; ".join(labels) if labels else "reference"
            try:
                cval = coeff.item()
            except Exception:
                cval = coeff
            lines.append(f"{label}\t| {cval}\t| {det_irreps[det_idx]}")
        with open(r1r2_outfile, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            f.write("\n")

    print(
        "C2v-filtered QSC-EOM eigenvalues "
        f"(target_irrep={'+'.join(metadata['target_irreps'])}, "
        f"ansatz_irrep={'+'.join(metadata['ansatz_target_irreps'])}):\n",
        eigvals,
    )
    return eigvals
