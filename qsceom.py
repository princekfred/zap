"""QSC-EOM helpers (PennyLane)."""

import exc


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
    """Run exact QSC-EOM from optimized VQE parameters."""
    try:
        import pennylane as qml
        from pennylane import numpy as np
    except ModuleNotFoundError as exc_mod:
        raise ModuleNotFoundError(
            "Missing dependency 'pennylane'. Install with:\n"
            "  python -m pip install pennylane pennylane-lightning pyscf"
        ) from exc_mod

    try:
        geometry = np.array(geometry, dtype=float, requires_grad=False)
    except TypeError:
        geometry = np.array(geometry, dtype=float)

    H, qubits = qml.qchem.molecular_hamiltonian(
        symbols,
        geometry,
        basis=basis,
        method=method,
        unit=unit,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        charge=charge,
    )
    singles, doubles = qml.qchem.excitations(active_electrons, qubits)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
    wires = range(qubits)

    list1 = exc.inite(active_electrons, qubits)
    null_state = np.zeros(qubits, dtype=int)

    dev_kwargs = {"wires": qubits}
    if shots and int(shots) > 0:
        dev_kwargs["shots"] = int(shots)

    try:
        dev = qml.device("lightning.qubit", **dev_kwargs)
    except Exception:
        dev = qml.device("default.qubit", **dev_kwargs)

    @qml.qnode(dev)
    def circuit_d(params_, occ, wires_, s_wires_, d_wires_, hf_state):
        for w in occ:
            qml.X(wires=w)
        qml.UCCSD(params_, wires_, s_wires_, d_wires_, hf_state)
        return qml.expval(H)

    @qml.qnode(dev)
    def circuit_od(params_, occ1, occ2, wires_, s_wires_, d_wires_, hf_state):
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

        qml.UCCSD(params_, wires_, s_wires_, d_wires_, hf_state)
        return qml.expval(H)

    M = np.zeros((len(list1), len(list1)), dtype=float)
    for i in range(len(list1)):
        M[i, i] = circuit_d(params, list1[i], wires, s_wires, d_wires, null_state)

    for i in range(len(list1)):
        for j in range(len(list1)):
            if i == j:
                continue
            mtmp = circuit_od(params, list1[i], list1[j], wires, s_wires, d_wires, null_state)
            M[i, j] = mtmp - M[i, i] / 2.0 - M[j, j] / 2.0

    eig, evec = np.linalg.eig(M)
    idx = np.argsort(eig)
    eig = eig[idx]
    evec = evec[:, idx]

    if state_idx < 0 or state_idx >= len(eig):
        raise ValueError(
            f"`state_idx` must be between 0 and {len(eig) - 1}, got {state_idx}."
        )

    hf_state = np.array(range(active_electrons))
    vector = evec[:, state_idx]

    if r1r2_outfile:
        lines = ["R1/R2", "Excitations | Coefficients"]
        for det_idx, coeff in enumerate(vector):
            det = list1[det_idx]
            holes = [hf for hf in hf_state if hf not in det]
            particles = [virt for virt in det if virt not in hf_state]
            labels = [f"{p}^ {h}" for p, h in zip(particles, holes)]
            label = "; ".join(labels) if labels else "reference"
            lines.append(f"{label}\t| {coeff.item()}")
        with open(r1r2_outfile, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            f.write("\n")

    print("QSC-EOM eigenvalues:\n", eig)
    return eig
