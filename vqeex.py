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


def gs_exact(
    symbols,
    geometry,
    active_electrons,
    active_orbitals,
    charge,
    method="pyscf",
    shots=None,
    max_iter=100,
    opt_method="BFGS",
):
    """Optimize a non‑Trotterized UCCSD ansatz using dense matrices.

    Returns the optimized parameter vector in PennyLane excitation ordering:
    singles first, then doubles.
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

    # PennyLane's qchem helpers expect NumPy semantics (e.g., `.flatten()`).
    try:
        geometry = pnp.array(geometry, dtype=float, requires_grad=False)
    except TypeError:
        geometry = pnp.array(geometry, dtype=float)

    H, n_qubits = qml.qchem.molecular_hamiltonian(
        symbols,
        geometry,
        basis="sto-3g",
        method=method,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        charge=charge,
    )

    wire_order = list(range(n_qubits))
    hf_state = qml.qchem.hf_state(active_electrons, n_qubits)

    h_mat = np.asarray(qml.matrix(H, wire_order=wire_order), dtype=complex)

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def _hf_statevector():
        qml.BasisState(hf_state, wires=wire_order)
        return qml.state()

    hf_vec = np.asarray(_hf_statevector(), dtype=complex)

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

    wire0_is_msb = _wire0_is_msb(qml, wire_order)
    a_ops, adag_ops = _creation_annihilation_mats(
        n_modes=n_qubits, wire_order=wire_order, wire0_is_msb=wire0_is_msb
    )

    # Build anti-Hermitian generators K_k so U(params)=exp(sum_k params[k] * K_k).
    k_mats = []
    for (i, a) in singles:
        k_mats.append(adag_ops[a] @ a_ops[i] - adag_ops[i] @ a_ops[a])
    for (i, j, a, b) in doubles:
        t = adag_ops[a] @ adag_ops[b] @ a_ops[j] @ a_ops[i]
        k_mats.append(t - t.conj().T)

    k_stack = np.stack(k_mats, axis=0)
    x0 = np.zeros(k_stack.shape[0], dtype=float)

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
    res = minimize(energy, x0, method=opt_method, options={"maxiter": int(max_iter)})
    elapsed = time.time() - t0

    params = res.x
    e_min = res.fun

    print(f"Optimization time: {elapsed:.2f}s")
    print("Optimizer:", opt_method, "| success:", bool(getattr(res, "success", False)))
    if hasattr(res, "message"):
        print("Message:", res.message)

    print("\nOptimal parameters:\n", list(params))
    print("Energy minimum = ", e_min)

    # Print amplitudes in the same convention/order as Vqe.py (doubles then singles).
    print("\nPrinting amplitudes")
    print("Operator\tAmplitude")
    print("++++++++++++++++++++++++++++++")

    n_s = len(singles)
    t1 = params[:n_s]
    t2 = params[n_s:]

    for (i, j, a, b), amp in zip(doubles, t2):
        print(f"{a}^ {b}^ {i} {j} \t| {amp}")
    for (i, a), amp in zip(singles, t1):
        print(f"{a}^ {i} \t| {amp}")
    
    return params
