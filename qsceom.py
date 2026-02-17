"""QSC-EOM post-VQE utilities with deterministic R1/R2 output."""

import exc


def _wire0_is_msb(qml, wire_order):
    """Infer computational-basis bit ordering for the given wire order."""
    if len(wire_order) <= 1:
        return True

    import numpy as np

    x0 = np.asarray(
        qml.matrix(qml.PauliX(wire_order[0]), wire_order=wire_order), dtype=complex
    )
    dim = x0.shape[0]

    idx_msb = 1 << (len(wire_order) - 1)
    if idx_msb < dim and abs(x0[idx_msb, 0] - 1) < 1e-12:
        return True
    if dim > 1 and abs(x0[1, 0] - 1) < 1e-12:
        return False
    return True


def _creation_annihilation_mats(*, n_modes, wire_order, wire0_is_msb):
    """Dense Jordan-Wigner creation/annihilation matrices for each mode."""
    import numpy as np

    dim = 1 << n_modes
    wire_to_pos = {w: idx for idx, w in enumerate(wire_order)}

    def bitpos(mode):
        pos = wire_to_pos[mode]
        return (n_modes - 1 - pos) if wire0_is_msb else pos

    parity_masks = []
    mask = 0
    for p in range(n_modes):
        parity_masks.append(mask)
        mask |= 1 << bitpos(p)

    a = []
    adag = []
    for p in range(n_modes):
        a_p = np.zeros((dim, dim), dtype=complex)
        adag_p = np.zeros((dim, dim), dtype=complex)

        bp = bitpos(p)
        flip = 1 << bp
        parity_mask = parity_masks[p]

        for basis in range(dim):
            parity = (basis & parity_mask).bit_count() & 1
            sign = -1.0 if parity else 1.0
            occ = (basis >> bp) & 1

            if occ:
                new_state = basis ^ flip
                a_p[new_state, basis] = sign
            else:
                new_state = basis | flip
                adag_p[new_state, basis] = sign

        a.append(a_p)
        adag.append(adag_p)

    return a, adag


def _ordered_excitations(qml, active_electrons, n_qubits):
    """Return the exact-UCCSD excitation ordering used in vqeex.gs_exact."""

    def order_pair(i, j):
        if (i % 2) != (j % 2):
            return (i, j) if (i % 2) == 0 else (j, i)
        return (i, j) if i <= j else (j, i)

    def normalize_double(ex):
        i, j, a, b = ex
        i, j = order_pair(i, j)
        a, b = order_pair(a, b)
        return (i, j, a, b)

    def double_sort_key(ex):
        i, j, a, b = ex
        mixed_spin = (i % 2) != (j % 2)
        if mixed_spin:
            category = 0
        else:
            category = 1 if (i % 2) == 0 else 2
        return (category, a // 2, b // 2, i // 2, j // 2, a, b, i, j)

    def single_sort_key(ex):
        i, a = ex
        return (i // 2, a // 2, i % 2, a, i)

    singles, doubles = qml.qchem.excitations(active_electrons, n_qubits)
    singles = sorted(singles, key=single_sort_key)
    doubles = sorted((normalize_double(ex) for ex in doubles), key=double_sort_key)
    return singles, doubles


def _exact_uccsd_k_stack(qml, active_electrons, n_qubits, wire_order):
    """Build anti-Hermitian exact-UCCSD generators K_k."""
    import numpy as np

    singles, doubles = _ordered_excitations(qml, active_electrons, n_qubits)
    wire0_is_msb = _wire0_is_msb(qml, wire_order)
    a_ops, adag_ops = _creation_annihilation_mats(
        n_modes=n_qubits, wire_order=wire_order, wire0_is_msb=wire0_is_msb
    )

    k_mats = []
    for (i, a) in singles:
        k_mats.append(adag_ops[a] @ a_ops[i] - adag_ops[i] @ a_ops[a])
    for (i, j, a, b) in doubles:
        t = adag_ops[a] @ adag_ops[b] @ a_ops[j] @ a_ops[i]
        k_mats.append(t - t.conj().T)

    if not k_mats:
        dim = 1 << n_qubits
        return np.zeros((0, dim, dim), dtype=complex)
    return np.stack(k_mats, axis=0)


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
    basis="sto-3g",
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
    try:
        from scipy.linalg import expm
    except ModuleNotFoundError as exc_mod:
        raise ModuleNotFoundError(
            "Missing dependency 'scipy'. Install with:\n  python -m pip install scipy"
        ) from exc_mod
    try:
        from scipy.sparse.linalg import expm_multiply
    except Exception:
        expm_multiply = None

    if shots not in (0, None):
        print("Note: shots are ignored in exact statevector QSC-EOM evaluation.")

    try:
        geometry = pnp.array(geometry, dtype=float, requires_grad=False)
    except TypeError:
        geometry = pnp.array(geometry, dtype=float)

    hamiltonian, qubits = qml.qchem.molecular_hamiltonian(
        symbols,
        geometry,
        basis=basis,
        method=method,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        charge=charge,
    )
    wires = list(range(qubits))

    determinants = exc.inite(active_electrons, qubits)
    if not determinants:
        raise ValueError("No determinants were generated by exc.inite.")

    try:
        dev = qml.device("lightning.qubit", wires=qubits)
    except Exception:
        dev = qml.device("default.qubit", wires=qubits)

    params = np.asarray(params, dtype=float).reshape(-1)
    h_mat = np.asarray(qml.matrix(hamiltonian, wire_order=wires), dtype=complex)
    k_stack = _exact_uccsd_k_stack(
        qml=qml, active_electrons=active_electrons, n_qubits=qubits, wire_order=wires
    )

    expected_params = k_stack.shape[0]
    if params.size != expected_params:
        raise ValueError(
            f"Parameter length mismatch: got {params.size}, expected {expected_params} "
            "for exact UCCSD excitations."
        )

    if expected_params == 0:
        k_total = np.zeros_like(h_mat)
    else:
        k_total = np.tensordot(params, k_stack, axes=(0, 0))
    u_mat = None if expm_multiply is not None else expm(k_total)

    @qml.qnode(dev)
    def _basis_state_from_occ(occ):
        init_state = np.zeros(qubits, dtype=int)
        for w in occ:
            init_state[w] = 1
        qml.BasisState(init_state, wires=wires)
        return qml.state()

    def _apply_exact_uccsd(ref_state):
        if expm_multiply is not None:
            return np.asarray(expm_multiply(k_total, ref_state), dtype=complex)
        return u_mat @ ref_state

    def _energy_expectation(state):
        return float(np.real(np.vdot(state, h_mat @ state)))

    # Precompute determinant basis states |D_i>.
    det_states = [np.asarray(_basis_state_from_occ(occ), dtype=complex) for occ in determinants]

    # Mirror the reference formulation:
    #   diagonal:   M_ii = <U D_i| H |U D_i>
    #   off-diag:   M_ij = E[(U D_i + U D_j)/sqrt(2)] - M_ii/2 - M_jj/2
    n_det = len(determinants)
    m_matrix = np.zeros((n_det, n_det), dtype=float)

    def circuit_d(i):
        psi = _apply_exact_uccsd(det_states[i])
        return _energy_expectation(psi)

    def circuit_od(i, j):
        ref_super = (det_states[i] + det_states[j]) / np.sqrt(2.0)
        psi_super = _apply_exact_uccsd(ref_super)
        return _energy_expectation(psi_super)

    for i in range(n_det):
        m_matrix[i, i] = circuit_d(i)

    for i in range(n_det):
        for j in range(n_det):
            if i != j:
                mtmp = circuit_od(i, j)
                m_matrix[i, j] = mtmp - 0.5 * m_matrix[i, i] - 0.5 * m_matrix[j, j]

    m_matrix = 0.5 * (m_matrix + m_matrix.T)
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
