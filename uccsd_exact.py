"""Exact (non-Trotterized) UCCSD VQE with dense exp(T - T^dagger)."""


def _canon_pair(p, q):
    """Put mixed-spin pairs in (even, odd) order; return (p2, q2, sign)."""
    p, q = int(p), int(q)
    sign = 1.0
    if (p % 2) != (q % 2):
        if p % 2 == 1:  # (odd, even) -> swap
            p, q = q, p
            sign *= -1.0
    else:
        if p > q:  # keep same-spin pairs ascending
            p, q = q, p
            sign *= -1.0
    return p, q, sign


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
    opt_method="Powell",
    amplitudes_outfile=None,
    hamiltonian=None,
    qubits=None,
):
    """Run exact dense UCCSD with U = exp(T - T^dagger)."""
    try:
        import pennylane as qml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'pennylane'. Install with:\n"
            "  python -m pip install pennylane pennylane-lightning pyscf"
        ) from exc

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'numpy'. Install with:\n  python -m pip install numpy"
        ) from exc

    try:
        from scipy.linalg import expm
        from scipy.optimize import minimize
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'scipy'. Install with:\n  python -m pip install scipy"
        ) from exc

    if hamiltonian is None:
        raise ValueError(
            "`hamiltonian` is required. Internal Hamiltonian building has been disabled."
        )
    if basis is None or unit is None:
        raise ValueError("`basis` and `unit` must be provided by the caller.")
    if max_iter < 1:
        raise ValueError("`max_iter` must be >= 1")

    H = hamiltonian
    if qubits is None:
        try:
            qubits = len(H.wires)
        except Exception as exc:
            raise ValueError(
                "Could not infer `qubits` from the provided Hamiltonian. Pass `qubits` explicitly."
            ) from exc
    qubits = int(qubits)

    wire_order = list(range(qubits))
    hf_state = np.asarray(qml.qchem.hf_state(active_electrons, qubits), dtype=int)

    # Excitations and parameters (same ordering as vqe.py)
    singles, doubles = qml.qchem.excitations(active_electrons, qubits)
    n_s = len(singles)
    n_d = len(doubles)
    n_params = n_s + n_d
    params0 = np.zeros(n_params, dtype=float)

    # Dense qubit Hamiltonian matrix.
    h_mat = np.asarray(qml.matrix(H, wire_order=wire_order), dtype=complex)
    dim = h_mat.shape[0]

    # Build |HF> statevector in big-endian computational-basis ordering.
    hf_vec = np.zeros(dim, dtype=complex)
    hf_idx = 0
    for occ in hf_state:
        hf_idx = (hf_idx << 1) | int(occ)
    hf_vec[hf_idx] = 1.0

    # Dense Jordan-Wigner creation/annihilation matrices.
    I2 = np.eye(2, dtype=complex)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    a1 = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    adag1 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)

    def kron_all(ops):
        out = ops[0]
        for op in ops[1:]:
            out = np.kron(out, op)
        return out

    a_ops = []
    adag_ops = []
    for p in range(qubits):
        ops_z = [Z] * p
        ops_tail = [I2] * (qubits - p - 1)
        a_ops.append(kron_all(ops_z + [a1] + ops_tail))
        adag_ops.append(kron_all(ops_z + [adag1] + ops_tail))

    # Build explicit T-terms, same excitation ordering as vqe.py.
    single_terms = [adag_ops[a] @ a_ops[i] for (i, a) in singles]
    double_terms = [adag_ops[a] @ adag_ops[b] @ a_ops[j] @ a_ops[i] for (i, j, a, b) in doubles]

    def energy_exact(params):
        params = np.asarray(params, dtype=float)
        t_op = np.zeros((dim, dim), dtype=complex)
        for coeff, op in zip(params[:n_s], single_terms):
            t_op = t_op + coeff * op
        for coeff, op in zip(params[n_s:], double_terms):
            t_op = t_op + coeff * op

        # Non-Trotterized unitary: U = exp(T - T^\dagger)
        k_op = t_op - t_op.conj().T
        psi = expm(k_op) @ hf_vec
        return float(np.real(np.vdot(psi, h_mat @ psi)))

    hf_e = energy_exact(np.zeros(n_params, dtype=float))
    print("HF energy:", hf_e)
    if shots is not None:
        print("Note: `shots` ignored (exact dense statevector expectation value).")

    if n_params == 0:
        params = np.zeros(0, dtype=float)
        energy = hf_e
    else:
        result = minimize(
            energy_exact,
            params0,
            method=opt_method,
            options={"maxiter": int(max_iter)},
        )
        params = np.asarray(result.x, dtype=float)
        energy = float(result.fun)
        print("Optimizer:", opt_method, "| success:", bool(getattr(result, "success", False)))
        if hasattr(result, "message"):
            print("Message:", result.message)

    print("\nOptimal parameters:\n", list(params))
    print("UCCSD energy = ", energy)

    # Print amplitudes exactly in the simple vqe.py style.
    print("\nPrinting amplitudes")
    print("Operator\tAmplitude")
    print("++++++++++++++++++++++++++++++")

    t1 = params[:n_s]
    t2 = params[n_s:]

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
            f.write("\n")
            f.write("Operator\tAmplitude\n")
            f.write("++++++++++++++++++++++++++++++\n")
            f.write("\n".join(amp_lines))
            f.write("\n")

    return params
