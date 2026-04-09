"""Exact (non‑Trotterized) UCCSD VQE via dense matrix exponentiation.

PennyLane's built-in ``qml.UCCSD`` template is a *disentangled* (first‑order
Trotterized) product of exponentials. This module instead applies the *combined*
UCCSD generator exactly:

    U(params) = exp(T(params) - T(params)†)

where ``T`` contains the (spin-conserving) single and double excitation terms
returned by ``qml.qchem.excitations``.

Implementation notes:
  - This forms dense 2**n × 2**n matrices, so it only works for small systems.
  - ``shots`` is accepted for API compatibility but ignored (statevector energy).
"""


def _wire0_is_msb(qml, wire_order):
    """Infer computational-basis bit ordering for the given ``wire_order``."""
    if len(wire_order) <= 1:
        return True

    import numpy as np

    x0 = np.asarray(
        qml.matrix(qml.PauliX(wire_order[0]), wire_order=wire_order), dtype=complex
    )
    dim = x0.shape[0]

    # If wire_order[0] is MSB, X maps |0...0> -> |1 0...0>, index 2^(n-1).
    idx_msb = 1 << (len(wire_order) - 1)
    if idx_msb < dim and abs(x0[idx_msb, 0] - 1) < 1e-12:
        return True

    # If wire_order[0] is LSB, X maps |0...0> -> |0...01>, index 1.
    if dim > 1 and abs(x0[1, 0] - 1) < 1e-12:
        return False

    # Fallback: PennyLane uses big-endian ordering in most contexts.
    return True


def _creation_annihilation_mats(*, n_modes, wire_order, wire0_is_msb):
    """Dense Jordan–Wigner creation/annihilation matrices for each mode."""
    import numpy as np

    dim = 1 << n_modes

    wire_to_pos = {w: idx for idx, w in enumerate(wire_order)}

    def bitpos(mode):
        pos = wire_to_pos[mode]
        return (n_modes - 1 - pos) if wire0_is_msb else pos

    # parity_mask[p] includes bits for modes 0..p-1 (JW Z-string).
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


def _normalize_opt_method_name(opt_method):
    """Normalize common optimizer aliases to scipy names."""
    name = str(opt_method).strip()
    token = name.upper().replace("-", "").replace("_", "").replace(" ", "")
    aliases = {
        "BFGS": "BFGS",
        "LBGS": "BFGS",
        "LBFGS": "BFGS",
        "LBFGSB": "BFGS",
        "LBGFS": "BFGS",
        "GRADIENTDESCENT": "BFGS",
        "GD": "BFGS",
        "GRADIENT": "BFGS",
    }
    return aliases.get(token, name)


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


def _print_amplitudes(
    *,
    singles,
    doubles,
    params,
    active_electrons=None,
    active_orbitals=None,
    hf_energy=None,
    energy_min=None,
    amplitudes_outfile=None,
):
    """Print parameters in the expected `a^ b^ i j` / `a^ i` notation and order."""
    import numpy as np

    print("\nPrinting amplitudes")
    print("Operator\tAmplitude")
    print("++++++++++++++++++++++++++++++")

    n_s = len(singles)
    t1 = np.asarray(params[:n_s], dtype=float)
    t2 = np.asarray(params[n_s:], dtype=float)

    lines = []
    for (i, j, a, b), amp in zip(doubles, t2):
        a2, b2, s_ab = _canon_pair(a, b)
        i2, j2, s_ij = _canon_pair(i, j)
        line = f"{a2}^ {b2}^ {i2} {j2} \t| {s_ab * s_ij * amp}"
        lines.append(line)
        print(line)

    for (i, a), amp in zip(singles, t1):
        line = f"{a}^ {i} \t| {amp}"
        lines.append(line)
        print(line)

    if amplitudes_outfile:
        with open(amplitudes_outfile, "w", encoding="utf-8") as f:
            if active_electrons is not None:
                f.write(f"Active electrons: {int(active_electrons)}\n")
            if active_orbitals is not None:
                f.write(f"Active orbitals: {int(active_orbitals)}\n")
            if hf_energy is not None:
                f.write(f"HF energy: {float(hf_energy)}\n")
            if energy_min is not None:
                f.write(f"UCCSD energy: {float(energy_min)}\n")
            if (
                active_electrons is not None
                or active_orbitals is not None
                or hf_energy is not None
                or energy_min is not None
            ):
                f.write("\n")
            f.write("Operator\tAmplitude\n")
            f.write("++++++++++++++++++++++++++++++\n")
            f.write("\n".join(lines))
            f.write("\n")


def gs_exact(
    symbols,
    geometry,
    active_electrons,
    active_orbitals,
    charge,
    shots=None,
    max_iter=500,
    opt_method="BFGS",
    method="pyscf",
    basis=None,
    unit=None,
    amplitudes_outfile=None,
    hamiltonian=None,
    qubits=None,
    enforce_spin_paired_t1=True,
):
    """Optimize a non‑Trotterized UCCSD ansatz using dense matrices.

    Returns the optimized parameter vector in PennyLane excitation ordering:
    singles first, then doubles.

    When ``enforce_spin_paired_t1=True``, single excitations that differ only by
    alpha/beta spin channel (for example ``6^ 0`` and ``7^ 1``) share one
    variational parameter exactly.
    """
    import time

    try:
        import pennylane as qml
        from pennylane import numpy as pnp
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'pennylane'. Install with:\n"
            "  python -m pip install pennylane pennylane-lightning pyscf"
        ) from exc

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'numpy'. Install with:\n" "  python -m pip install numpy"
        ) from exc

    try:
        from scipy.linalg import expm
        from scipy.optimize import minimize
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'scipy'. Install with:\n" "  python -m pip install scipy"
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
    n_qubits = int(qubits)

    # Keep this conversion for compatibility with existing callers and downstream helpers.
    try:
        geometry = pnp.array(geometry, dtype=float, requires_grad=False)
    except TypeError:
        geometry = pnp.array(geometry, dtype=float)
    wire_order = list(range(n_qubits))
    hf_state = qml.qchem.hf_state(active_electrons, n_qubits)

    h_mat = np.asarray(qml.matrix(H, wire_order=wire_order), dtype=complex)

    singles, doubles = qml.qchem.excitations(active_electrons, n_qubits)

    wire0_is_msb = _wire0_is_msb(qml, wire_order)
    dim = 1 << n_qubits

    hf_vec = np.zeros(dim, dtype=complex)
    occ_wires = np.nonzero(np.asarray(hf_state, dtype=int))[0]
    hf_idx = 0
    for w in occ_wires:
        bit = (n_qubits - 1 - int(w)) if wire0_is_msb else int(w)
        hf_idx |= 1 << bit
    hf_vec[hf_idx] = 1.0

    a_ops, adag_ops = _creation_annihilation_mats(
        n_modes=n_qubits, wire_order=wire_order, wire0_is_msb=wire0_is_msb
    )

    # Build anti-Hermitian generators K_k so U(params)=exp(sum_k params[k] * K_k).
    k_single = []
    for (i, a) in singles:
        k_single.append(adag_ops[a] @ a_ops[i] - adag_ops[i] @ a_ops[a])
    k_double = []
    for (i, j, a, b) in doubles:
        t = adag_ops[a] @ adag_ops[b] @ a_ops[j] @ a_ops[i]
        k_double.append(t - t.conj().T)

    if enforce_spin_paired_t1 and len(k_single) > 0:
        single_to_free, ordered_single_keys = _build_spin_paired_single_map(singles)
        n_single_free = len(ordered_single_keys)
        k_single_tied = [np.zeros((dim, dim), dtype=complex) for _ in range(n_single_free)]
        for single_idx, free_idx in enumerate(single_to_free):
            k_single_tied[free_idx] = k_single_tied[free_idx] + k_single[single_idx]
        k_single_stack = np.stack(k_single_tied, axis=0)
        print(
            "Enforcing spin-paired T1 amplitudes:",
            f"{len(k_single)} -> {n_single_free} independent singles.",
        )
    else:
        n_single_free = len(k_single)
        single_to_free = list(range(n_single_free))
        k_single_stack = (
            np.stack(k_single, axis=0)
            if len(k_single) > 0
            else np.zeros((0, dim, dim), dtype=complex)
        )

    k_double_stack = (
        np.stack(k_double, axis=0)
        if len(k_double) > 0
        else np.zeros((0, dim, dim), dtype=complex)
    )

    k_stack = np.concatenate((k_single_stack, k_double_stack), axis=0)
    x0 = np.zeros(int(n_single_free + len(k_double)), dtype=float)

    try:
        from scipy.sparse.linalg import expm_multiply
    except Exception:
        expm_multiply = None

    def energy(x):
        k_total = np.tensordot(x, k_stack, axes=(0, 0))
        if expm_multiply is not None:
            psi = expm_multiply(k_total, hf_vec)
        else:
            psi = expm(k_total) @ hf_vec
        return float(np.real(np.vdot(psi, h_mat @ psi)))

    hf_e = energy(x0)
    print("HF energy:", hf_e)
    if shots is not None:
        print("Note: `shots` ignored (statevector expectation value).")

    t0 = time.time()
    requested_method = _normalize_opt_method_name(opt_method)
    method = "BFGS"
    if str(requested_method).upper() != "BFGS":
        print(f"Requested optimizer `{opt_method}` overridden to BFGS (tol=1e-8).")

    if int(x0.size) == 0:
        x_opt = np.asarray(x0, dtype=float)
        energy = float(hf_e)
        success = True
        message = "No variational parameters; returning HF state energy."
    else:
        res = minimize(
            energy,
            x0,
            method=method,
            tol=1e-8,
            options={"maxiter": int(max_iter), "gtol": 1e-8},
        )
        x_opt = np.asarray(res.x, dtype=float)
        energy = float(res.fun)
        success = bool(getattr(res, "success", False))
        message = str(getattr(res, "message", ""))

    # Expand tied optimization variables back to full singles+doubles ordering.
    if enforce_spin_paired_t1 and len(k_single) > 0:
        t1_full = np.zeros(len(singles), dtype=float)
        for single_idx, free_idx in enumerate(single_to_free):
            t1_full[single_idx] = x_opt[free_idx]
        t2_full = np.asarray(x_opt[n_single_free:], dtype=float)
        params = np.concatenate((t1_full, t2_full), axis=0)
    else:
        params = np.asarray(x_opt, dtype=float)

    elapsed = time.time() - t0

    print(f"Optimization time: {elapsed:.2f}s")
    print("Optimizer:", method, "| success:", success)
    if message:
        print("Message:", message)

    print("\nOptimal parameters:\n", list(params))
    print("UCCSD energy = ", energy)

    _print_amplitudes(
        singles=singles,
        doubles=doubles,
        params=params,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        hf_energy=hf_e,
        energy_min=energy,
        amplitudes_outfile=amplitudes_outfile,
    )

    return params
