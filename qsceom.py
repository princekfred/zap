"""QSC-EOM post-VQE utilities with deterministic R1/R2 output."""

import exc


def _excitation_label(det, hf_occ):
    holes = [h for h in hf_occ if h not in det]
    particles = [p for p in det if p not in hf_occ]
    pairs = [f"{p}^ {h}" for p, h in zip(particles, holes)]
    return "; ".join(pairs) if pairs else "reference"


def _build_permutation_operator(determinants, mapping):
    import numpy as np

    det_to_idx = {tuple(sorted(det)): idx for idx, det in enumerate(determinants)}
    n_det = len(determinants)
    op = np.zeros((n_det, n_det), dtype=complex)

    for i, det in enumerate(determinants):
        mapped = tuple(sorted(mapping(orb) for orb in det))
        j = det_to_idx.get(mapped)
        op[i, j if j is not None else i] = 1.0
    return op


def _symmetry_adapt_evecs(eigvals, eigvecs, determinants, n_spatial, deg_tol=1e-6):
    import numpy as np

    # Spin-flip symmetry: alpha <-> beta within each spatial orbital.
    spin_flip = _build_permutation_operator(determinants, lambda x: x ^ 1)

    # Reflection in spatial-orbital index: p -> (n_spatial-1-p), keep spin.
    def reflect_spin_orb(orb):
        spatial = orb // 2
        spin = orb % 2
        mapped_spatial = n_spatial - 1 - spatial
        return 2 * mapped_spatial + spin

    reflection = _build_permutation_operator(determinants, reflect_spin_orb)

    out = eigvecs.copy()
    n_roots = out.shape[1]

    start = 0
    while start < n_roots:
        end = start + 1
        while end < n_roots and abs(float((eigvals[end] - eigvals[start]).real)) <= deg_tol:
            end += 1

        if end - start > 1:
            sub = out[:, start:end]
            for op in (spin_flip, reflection):
                proj = sub.conj().T @ op @ sub
                proj = 0.5 * (proj + proj.conj().T)
                w, u = np.linalg.eigh(proj)
                order = np.argsort(-np.real(w))
                sub = sub @ u[:, order]
            out[:, start:end] = sub

        start = end

    return out


def _canonicalize_vector_phase(vector, labels):
    import numpy as np

    anchors = ("6^ 0", "7^ 1", "4^ 2", "5^ 3")
    idx = None
    for target in anchors:
        try:
            cand = labels.index(target)
        except ValueError:
            continue
        if abs(vector[cand]) > 1e-12:
            idx = cand
            break

    if idx is None:
        idx = int(np.argmax(np.abs(vector)))
    ref = vector[idx]
    if abs(ref) <= 1e-14:
        return vector

    phase = np.exp(-1j * np.angle(ref))
    vec = vector * phase
    if np.real(vec[idx]) < 0:
        vec = -vec
    return vec


def _write_r1_r2(eigvals, eigvecs, determinants, hf_occ, state_idx=1, outfile="out_r1_r2.txt"):
    import numpy as np

    order = np.argsort(np.real(eigvals))
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    state_idx = max(0, min(int(state_idx), eigvecs.shape[1] - 1))
    labels = [_excitation_label(det, hf_occ) for det in determinants]
    vector = _canonicalize_vector_phase(eigvecs[:, state_idx], labels)

    lines = ["R1/R2", "Excitations | Coefficients"]
    for label, coeff in zip(labels, vector):
        cval = complex(coeff.item())
        if abs(cval.imag) < 1e-12:
            cval = complex(cval.real, 0.0)
        lines.append(f"{label}\t| {cval}")

    if outfile:
        with open(outfile, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

    return eigvals, eigvecs, vector


def ee_exact(
    symbols,
    geometry,
    active_electrons,
    active_orbitals,
    charge,
    params,
    shots=0,
    method="pyscf",
    state_idx=1,
    r1r2_outfile="out_r1_r2.txt",
):
    try:
        import numpy as np
    except ModuleNotFoundError as exc_mod:
        raise ModuleNotFoundError(
            "Missing dependency 'numpy'. Install with:\n  python -m pip install numpy"
        ) from exc_mod

    try:
        import pennylane as qml
        from pennylane import numpy as pnp
    except ModuleNotFoundError as exc_mod:
        raise ModuleNotFoundError(
            "Missing dependency 'pennylane'. Install with:\n"
            "  python -m pip install pennylane pennylane-lightning"
        ) from exc_mod

    if shots not in (0, None):
        print("Note: shots are ignored in exact statevector QSC-EOM evaluation.")

    try:
        geometry = pnp.array(geometry, dtype=float, requires_grad=False)
    except TypeError:
        geometry = pnp.array(geometry, dtype=float)

    hamiltonian, qubits = qml.qchem.molecular_hamiltonian(
        symbols,
        geometry,
        basis="sto-3g",
        method=method,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        charge=charge,
    )
    singles, doubles = qml.qchem.excitations(active_electrons, qubits)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
    wires = list(range(qubits))

    determinants = exc.inite(active_electrons, qubits)
    if not determinants:
        raise ValueError("No determinants were generated by exc.inite.")

    try:
        dev = qml.device("lightning.qubit", wires=qubits)
    except Exception:
        dev = qml.device("default.qubit", wires=qubits)

    params = np.asarray(params, dtype=float)
    h_mat = np.asarray(qml.matrix(hamiltonian, wire_order=wires), dtype=complex)

    @qml.qnode(dev)
    def state_for_det(occ):
        init_state = np.zeros(qubits, dtype=int)
        for w in occ:
            init_state[w] = 1
        qml.UCCSD(
            params,
            wires=wires,
            s_wires=s_wires,
            d_wires=d_wires,
            init_state=init_state,
        )
        return qml.state()

    states = []
    for occ in determinants:
        states.append(np.asarray(state_for_det(occ), dtype=complex))
    states = np.asarray(states, dtype=complex)

    m_matrix = states.conj() @ h_mat @ states.T
    m_matrix = 0.5 * (m_matrix + m_matrix.conj().T)

    eigvals, eigvecs = np.linalg.eigh(m_matrix)
    eigvecs = _symmetry_adapt_evecs(
        eigvals=eigvals,
        eigvecs=eigvecs,
        determinants=determinants,
        n_spatial=qubits // 2,
    )

    hf_occ = list(range(active_electrons))
    eigvals, eigvecs, vector = _write_r1_r2(
        eigvals=eigvals,
        eigvecs=eigvecs,
        determinants=determinants,
        hf_occ=hf_occ,
        state_idx=state_idx,
        outfile=r1r2_outfile,
    )
    print("exact eigenvalues:\n", eigvals)

    return {
        "matrix": m_matrix,
        "eigenvalues": eigvals,
        "vector": vector,
        "excitation_energies": np.real(eigvals - eigvals[0]),
    }