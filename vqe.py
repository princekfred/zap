"""VQE helpers (PennyLane)."""

def gs_exact(
    symbols,
    geometry,
    active_electrons,
    active_orbitals,
    charge,
    method="pyscf",
    basis="sto-3g",
    shots=None,
    max_iter=100,
    amplitudes_outfile=None,
):
    """Run a UCCSD-VQE ground-state calculation and print results.
    # his 
    Returns the optimized parameter vector.
    """
    try:
        import pennylane as qml
        from pennylane import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'pennylane'. Install with:\n"
            "  python -m pip install pennylane pennylane-lightning pyscf"
        ) from exc

    # PennyLane's qchem helpers expect NumPy semantics (e.g., `.flatten()`).
    try:
        geometry = np.array(geometry, dtype=float, requires_grad=False)
    except TypeError:
        geometry = np.array(geometry, dtype=float)

    # Build the electronic Hamiltonian
    try:
        H, qubits = qml.qchem.molecular_hamiltonian(
            symbols,
            geometry,
            basis=basis,
            method=method,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            charge=charge,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Failed to build the molecular Hamiltonian. For `method=\"pyscf\"`, install PySCF:\n"
            "  python -m pip install pyscf"
        ) from exc

    hf_state = qml.qchem.hf_state(active_electrons, qubits)

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

    print("HF energy:", hf_energy())

    # Excitations and wires
    singles, doubles = qml.qchem.excitations(active_electrons, qubits)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

    params = np.zeros(len(doubles) + len(singles), dtype=float)

    # Device
    dev = make_device(shots_=shots)

    @qml.qnode(dev, interface="autograd", diff_method="best")
    def circuit(params, wires, s_wires, d_wires, hf_state):
        qml.UCCSD(params, wires, s_wires, d_wires, hf_state)
        return qml.expval(H)

    if max_iter < 1:
        raise ValueError("`max_iter` must be >= 1")

    optimizer = qml.GradientDescentOptimizer(stepsize=0.5)
    energy = None
    for _ in range(max_iter):
        params, energy = optimizer.step_and_cost(
            circuit,
            params,
            wires=range(qubits),
            s_wires=s_wires,
            d_wires=d_wires,
            hf_state=hf_state,
        )

    print("\nOptimal parameters:\n", list(params))
    print("Energy minimum = ", energy)

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
            f.write("Operator\tAmplitude\n")
            f.write("++++++++++++++++++++++++++++++\n")
            f.write("\n".join(amp_lines))
            f.write("\n")

    return params
