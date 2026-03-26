from pathlib import Path
import sys

import numpy as np
import pennylane as qml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import casci
import qsceom
import symm


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "CH+2re" / "CH+ Trotterized"
AMP_FILE = OUTPUT_DIR / "t1_t2.txt"
RVEC_DIR = OUTPUT_DIR / "r_vectors"


def _canon_pair(p, q):
    p = int(p)
    q = int(q)
    sign = 1.0
    if (p % 2) != (q % 2):
        if p % 2 == 1:
            p, q = q, p
            sign *= -1.0
    else:
        if p > q:
            p, q = q, p
            sign *= -1.0
    return p, q, sign


def _load_params_from_t1_t2(cfg, qubits):
    singles, doubles = qml.qchem.excitations(cfg["active_electrons"], qubits)
    expected = len(doubles) + len(singles)

    amp_values = []
    for raw in AMP_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or "|" not in line:
            continue
        left, right = line.split("|", 1)
        if "^" not in left:
            continue
        try:
            amp_values.append(float(right.strip()))
        except ValueError:
            continue

    if len(amp_values) < expected:
        raise ValueError(
            f"Not enough amplitudes in {AMP_FILE}. Found {len(amp_values)}, expected {expected}."
        )
    amp_values = amp_values[:expected]

    # File order is doubles first (canonicalized sign), then singles.
    t2 = np.zeros(len(doubles), dtype=float)
    for idx, (i, j, a, b) in enumerate(doubles):
        _, _, s_ab = _canon_pair(a, b)
        _, _, s_ij = _canon_pair(i, j)
        t2[idx] = amp_values[idx] / (s_ab * s_ij)

    t1_start = len(doubles)
    t1 = np.asarray(amp_values[t1_start : t1_start + len(singles)], dtype=float)
    return np.concatenate([t1, t2])


def _write_r_vector_file(path, vector, det_list, active_electrons):
    hf_state = list(range(int(active_electrons)))
    lines = ["R1/R2", "Excitations | Coefficients"]
    for det_idx, coeff in enumerate(vector):
        det = det_list[det_idx]
        holes = [hf for hf in hf_state if hf not in det]
        particles = [virt for virt in det if virt not in hf_state]
        labels = [f"{p}^ {h}" for p, h in zip(particles, holes)]
        label = "; ".join(labels) if labels else "reference"
        try:
            cval = coeff.item()
        except Exception:
            cval = coeff
        lines.append(f"{label}\t| {cval}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if not AMP_FILE.exists():
        raise FileNotFoundError(f"Missing amplitude file: {AMP_FILE}")

    cfg = {
        "symbols": ["C", "H"],
        "geometry": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2 * 2.13713]], dtype=float),
        "active_electrons": 6,
        "active_orbitals": 6,
        "charge": 1,
        "basis": "6-31g",
        "unit": "bohr",
    }

    hamiltonian, qubits, _ = casci.build_casci_hamiltonian_from_problem(
        cfg,
        n_excited=0,
        casci_output_path=None,
    )
    qubits = int(qubits)
    params = _load_params_from_t1_t2(cfg, qubits)

    eigvals, eigvecs, det_list = qsceom.ee_exact(
        cfg["symbols"],
        cfg["geometry"],
        cfg["active_electrons"],
        cfg["active_orbitals"],
        cfg["charge"],
        params,
        shots=0,
        method="pyscf",
        basis=cfg["basis"],
        unit=cfg["unit"],
        state_idx=1,
        r1r2_outfile=None,
        hamiltonian=hamiltonian,
        qubits=qubits,
        return_eigvecs=True,
        symmetry_roots=0,
    )

    RVEC_DIR.mkdir(parents=True, exist_ok=True)
    for stale in RVEC_DIR.glob("r1_r2_soln*"):
        if stale.is_file():
            stale.unlink()

    sort_order = sorted(range(int(len(eigvals))), key=lambda idx: float(eigvals[idx]))
    n_roots = min(25, len(sort_order), int(eigvecs.shape[1]))
    dominant_by_root = {}
    try:
        root_symm = symm.summarize_lowest_qsceom_roots_r1r2(
            eigvals,
            eigvecs,
            det_list,
            symbols=cfg["symbols"],
            geometry=cfg["geometry"],
            active_electrons=cfg["active_electrons"],
            active_orbitals=cfg["active_orbitals"],
            charge=cfg["charge"],
            basis=cfg["basis"],
            unit=cfg["unit"],
            point_group="C2v",
            n_roots=n_roots,
        )
        dominant_by_root = {
            int(entry["root_index"]): str(entry["dominant_irrep"])
            for entry in root_symm.get("roots", [])
        }
    except Exception as exc:
        print("Root-wise symmetry table skipped:", exc)

    eigmap_lines = ["soln_idx\teigenvalue_hartree\tdominant_irrep"]
    print("Lowest 25 eigenvalue/irrep table (ascending):")
    print("soln_idx\teigenvalue_hartree\tdominant_irrep")
    for soln_idx, root_idx in enumerate(sort_order[:n_roots]):
        out_path = RVEC_DIR / f"r1_r2_soln{soln_idx}"
        _write_r_vector_file(out_path, eigvecs[:, root_idx], det_list, cfg["active_electrons"])
        energy = float(eigvals[root_idx])
        dominant = dominant_by_root.get(int(root_idx), "unknown")
        eigmap_lines.append(f"{soln_idx}\t{energy:.12f}\t{dominant}")
        print(f"{soln_idx}\t{energy:.12f}\t{dominant}")

    (RVEC_DIR / "eigenvalue_map.txt").write_text("\n".join(eigmap_lines) + "\n", encoding="utf-8")

    print(f"WROTE_DIR={RVEC_DIR}")
    print(f"ROOTS_WRITTEN={n_roots}")
    print(f"LOWEST_E={float(eigvals[0])}")
    print(f"HIGHEST_WRITTEN_E={float(eigvals[n_roots - 1])}")


if __name__ == "__main__":
    main()
