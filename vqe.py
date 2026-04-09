"""VQE helpers (PennyLane)."""


def _single_spatial_pair_key(single_excitation):
    """Key single excitations by (hole_spatial, particle_spatial) indices."""
    i, a = single_excitation
    return int(i) // 2, int(a) // 2


def _build_spin_paired_single_map(singles):
    """Map each single-excitation index to a shared spin-paired parameter index."""
    single_to_free = {}
    ordered_keys = []
    for single in singles:
        key = _single_spatial_pair_key(single)
        if key not in single_to_free:
            single_to_free[key] = len(ordered_keys)
            ordered_keys.append(key)
    return [single_to_free[_single_spatial_pair_key(single)] for single in singles], ordered_keys


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
    enforce_spin_paired_t1=True,
):
    """Run a UCCSD-VQE ground-state calculation and print results.

    Returns the optimized parameter vector.

    When ``enforce_spin_paired_t1=True``, single excitations that differ only by
    alpha/beta spin channel (for example ``6^ 0`` and ``7^ 1``) share one
    variational parameter exactly.
    """
    try:
        import pennylane as qml
        from pennylane import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'pennylane'. Install with:\n"
            "  python -m pip install pennylane pennylane-lightning pyscf"
        ) from exc
    try:
        from scipy.optimize import minimize
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'scipy'. Install with:\n"
            "  python -m pip install scipy"
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

    if enforce_spin_paired_t1 and len(singles) > 0:
        single_to_free, ordered_single_keys = _build_spin_paired_single_map(singles)
        n_single_free = len(ordered_single_keys)
        print(
            "Enforcing spin-paired T1 amplitudes:",
            f"{len(singles)} -> {n_single_free} independent singles.",
        )
    else:
        n_single_free = len(singles)
        single_to_free = list(range(n_single_free))

    x0 = np.zeros(n_single_free + len(doubles), dtype=float)

    # Device
    dev = make_device(shots_=shots)

    @qml.qnode(dev)
    def circuit(params, wires, s_wires, d_wires, hf_state):
        qml.UCCSD(params, wires, s_wires, d_wires, hf_state)
        return qml.expval(H)

    if max_iter < 1:
        raise ValueError("`max_iter` must be >= 1")

    def expand_params(x):
        x = np.asarray(x, dtype=float)
        if enforce_spin_paired_t1 and len(singles) > 0:
            t1_full = np.zeros(len(singles), dtype=float)
            for single_idx, free_idx in enumerate(single_to_free):
                t1_full[single_idx] = x[free_idx]
            t2_full = np.asarray(x[n_single_free:], dtype=float)
            return np.concatenate((t1_full, t2_full), axis=0)
        return x

    def objective(x):
        full_params = expand_params(x)
        value = circuit(
            full_params,
            wires=range(qubits),
            s_wires=s_wires,
            d_wires=d_wires,
            hf_state=hf_state,
        )
        return float(value)

    step_counter = {"n": 0}

    def callback(xk):
        step_counter["n"] += 1
        if step_counter["n"] % 10 == 0:
            e = objective(xk)
            print(f"VQE iter {step_counter['n']}: energy = {float(e):.12f}")

    if int(x0.size) == 0:
        params = expand_params(x0)
        energy = float(hf_e)
        print("Optimizer:", "BFGS", "| success:", True)
        print("Message:", "No variational parameters; returning HF state energy.")
    else:
        result = minimize(
            objective,
            np.asarray(x0, dtype=float),
            method="BFGS",
            tol=1e-8,
            callback=callback,
            options={"maxiter": int(max_iter), "gtol": 1e-8},
        )

        params = expand_params(result.x)
        energy = float(result.fun)
        print("Optimizer:", "BFGS", "| success:", bool(getattr(result, "success", False)))
        if hasattr(result, "message"):
            print("Message:", result.message)

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
            f.write("\n")
            f.write("Operator\tAmplitude\n")
            f.write("++++++++++++++++++++++++++++++\n")
            f.write("\n".join(amp_lines))
            f.write("\n")

    return params
