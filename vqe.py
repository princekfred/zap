"""VQE helpers (PennyLane)."""

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
):
    """Run a UCCSD-VQE ground-state calculation and print results.

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

    hf_e = hf_energy()
    print("HF energy:", hf_e)

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

    optimizer = qml.GradientDescentOptimizer(stepsize=0.1)
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
            f.write(f"Energy minimum: {float(energy)}\n")
            f.write("\n")
            f.write("Operator\tAmplitude\n")
            f.write("++++++++++++++++++++++++++++++\n")
            f.write("\n".join(amp_lines))
            f.write("\n")

    return params
