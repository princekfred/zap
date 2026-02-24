"""MPI-parallel QSC-EOM solver for ground-state ansatzes."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional, Sequence

from qchem_active_space import build_molecular_hamiltonian_enforcing_active_space

try:
    from mpi4py import MPI as _MPI
except ImportError:  # pragma: no cover
    _MPI = None

qml = None
np = None
_inite = None


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
    spec = importlib.util.spec_from_file_location("qsceom_ground_exc", exc_path)
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
    basis: str = "sto-3g",
    unit: str = "angstrom",
):
    """Build and diagonalize the QSC-EOM M-matrix.

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

    H, qubits, used_method = build_molecular_hamiltonian_enforcing_active_space(
        qml,
        symbols,
        coordinates,
        basis=basis,
        method=method,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        charge=charge,
        unit=unit,
    )
    if used_method != method:
        print("Hamiltonian backend used:", used_method)
    singles, doubles = qml.qchem.excitations(active_electrons, qubits)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
    wires = range(qubits)
    hf_state = qml.qchem.hf_state(active_electrons, qubits)

    null_state = np.zeros(qubits,int)
    excitation_configs = inite(active_electrons, qubits)
    #dev = _make_device(qubits, norm_shots)
    dev = qml.device("lightning.qubit", wires=qubits)

    @qml.qnode(dev)
    def circuit_d(params, occ, wires, s_wires, d_wires, hf_state):
        for w in occ:
            qml.X(wires=w)
        qml.UCCSD(params, wires, s_wires, d_wires, hf_state)
        return qml.expval(H)

    @qml.qnode(dev)
    def circuit_od(params, occ1, occ2,wires, s_wires, d_wires, hf_state):
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
        m_diag_local[i] = circuit_d(params, excitation_configs[i], wires, s_wires, d_wires, null_state)

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
                        circuit_od(params, excitation_configs[i], excitation_configs[j], wires, s_wires, d_wires, null_state)
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
    return eigvals[order], eigvecs[:, order]


def ee_exact(
    symbols,
    geometry,
    active_electrons,
    active_orbitals,
    charge,
    params,
    shots=0,
    method="pyscf",
    basis="sto-3g",
    unit="bohr",
    state_idx=1,
    r1r2_outfile="out_r1_r2.txt",
):
    """Compatibility wrapper used by existing run scripts.

    Computes QSC-EOM eigenvalues/eigenvectors and optionally writes R1/R2 coefficients.
    """
    _require_quantum_deps()
    inite = _load_inite()

    eigvals, eigvecs = qsc_eom(
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
    )

    if state_idx < 0 or state_idx >= len(eigvals):
        raise ValueError(
            f"`state_idx` must be between 0 and {len(eigvals) - 1}, got {state_idx}."
        )

    vector = eigvecs[:, state_idx]
    hf_state = np.array(range(active_electrons))
    # QSC-EOM determinant list uses spin-orbital count (2 * active_orbitals).
    det_list = inite(active_electrons, 2 * active_orbitals)

    if r1r2_outfile:
        lines = ["R1/R2", "Excitations | Coefficients"]
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
            lines.append(f"{label}\t| {cval}")
        with open(r1r2_outfile, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            f.write("\n")

    print("QSC-EOM eigenvalues:\n", eigvals)
    return eigvals
