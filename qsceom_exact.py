"""Exact QSC-EOM helpers consistent with the non-Trotterized `vqee` ansatz."""

import exc
import vqee
from qchem_active_space import build_molecular_hamiltonian_enforcing_active_space


def _occ_to_index(occ, n_qubits, wire0_is_msb):
    idx = 0
    for w in occ:
        bit = (n_qubits - 1 - int(w)) if wire0_is_msb else int(w)
        idx |= 1 << bit
    return idx


def _format_coeff(x):
    try:
        xr = float(x.real)
        xi = float(x.imag)
    except AttributeError:
        return str(x)
    if abs(xi) < 1e-10:
        return str(xr)
    return str(complex(xr, xi))


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
    """Run exact QSC-EOM using the same exact UCCSD unitary as `vqee`."""
    try:
        import pennylane as qml
        from pennylane import numpy as pnp
    except ModuleNotFoundError as exc_mod:
        raise ModuleNotFoundError(
            "Missing dependency 'pennylane'. Install with:\n"
            "  python -m pip install pennylane pennylane-lightning pyscf"
        ) from exc_mod

    try:
        import numpy as np
    except ModuleNotFoundError as exc_mod:
        raise ModuleNotFoundError(
            "Missing dependency 'numpy'. Install with:\n"
            "  python -m pip install numpy"
        ) from exc_mod

    try:
        from scipy.linalg import expm
    except ModuleNotFoundError as exc_mod:
        raise ModuleNotFoundError(
            "Missing dependency 'scipy'. Install with:\n"
            "  python -m pip install scipy"
        ) from exc_mod

    if shots and int(shots) > 0:
        print("Note: `shots` is ignored in `qsceom_exact` (statevector exact evaluation).")

    try:
        geometry = pnp.array(geometry, dtype=float, requires_grad=False)
    except TypeError:
        geometry = pnp.array(geometry, dtype=float)

    H, qubits, used_method = build_molecular_hamiltonian_enforcing_active_space(
        qml,
        symbols,
        geometry,
        basis=basis,
        method=method,
        unit=unit,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        charge=charge,
    )
    if used_method != method:
        print("Hamiltonian backend used:", used_method)
    h_mat = np.asarray(qml.matrix(H, wire_order=list(range(qubits))), dtype=complex)

    singles, doubles = qml.qchem.excitations(active_electrons, qubits)
    list1 = exc.inite(active_electrons, qubits)
    n_det = len(list1)

    wire_order = list(range(qubits))
    wire0_is_msb = vqee._wire0_is_msb(qml, wire_order)
    a_ops, adag_ops = vqee._creation_annihilation_mats(
        n_modes=qubits, wire_order=wire_order, wire0_is_msb=wire0_is_msb
    )

    k_mats = []
    for (i, a) in singles:
        k_mats.append(adag_ops[a] @ a_ops[i] - adag_ops[i] @ a_ops[a])
    for (i, j, a, b) in doubles:
        t = adag_ops[a] @ adag_ops[b] @ a_ops[j] @ a_ops[i]
        k_mats.append(t - t.conj().T)

    params = np.asarray(params, dtype=float)
    if params.size != len(k_mats):
        raise ValueError(
            f"Expected {len(k_mats)} parameters (singles+doubles), got {params.size}."
        )

    k_stack = np.stack(k_mats, axis=0)
    k_total = np.tensordot(params, k_stack, axes=(0, 0))
    u_mat = expm(k_total)

    dim = 1 << qubits
    det_vectors = np.zeros((dim, n_det), dtype=complex)
    for j, occ in enumerate(list1):
        det_vectors[_occ_to_index(occ, qubits, wire0_is_msb), j] = 1.0

    psi = u_mat @ det_vectors
    m_mat = psi.conj().T @ h_mat @ psi
    m_mat = 0.5 * (m_mat + m_mat.conj().T)

    eig, evec = np.linalg.eigh(m_mat)
    idx = np.argsort(eig.real)
    eig = eig[idx].real
    evec = evec[:, idx]

    if state_idx < 0 or state_idx >= len(eig):
        raise ValueError(
            f"`state_idx` must be between 0 and {len(eig) - 1}, got {state_idx}."
        )

    vector = evec[:, state_idx]
    pivot = int(np.argmax(np.abs(vector)))
    if abs(vector[pivot]) > 0:
        vector = vector / (vector[pivot] / abs(vector[pivot]))

    hf_state = list(range(active_electrons))
    if r1r2_outfile:
        lines = ["R1/R2", "Excitations | Coefficients"]
        for det_idx, coeff in enumerate(vector):
            det = list1[det_idx]
            holes = [hf for hf in hf_state if hf not in det]
            particles = [virt for virt in det if virt not in hf_state]
            labels = [f"{p}^ {h}" for p, h in zip(particles, holes)]
            label = "; ".join(labels) if labels else "reference"
            lines.append(f"{label}\t| {_format_coeff(coeff)}")
        with open(r1r2_outfile, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            f.write("\n")

    print("QSC-EOM exact eigenvalues:\n", eig)
    return eig
