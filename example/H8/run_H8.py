"""Single entrypoint for SCF + exact VQE + QSC-EOM workflow for H8."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import SCF
import qsceom
import vqeex


def _as_array(coords):
    try:
        from pennylane import numpy as pnp

        try:
            return pnp.array(coords, dtype=float, requires_grad=False)
        except TypeError:
            return pnp.array(coords, dtype=float)
    except ModuleNotFoundError:
        try:
            import numpy as np
        except ModuleNotFoundError:
            return coords
        return np.array(coords, dtype=float)


def _default_problem():
    r = 1.88973
    symbols = ["H", "H", "H", "H", "H", "H", "H", "H"]
    coords = [
        [0.0, 0.0, 0.0 * r],
        [0.0, 0.0, 2 * r],
        [0.0, 0.0, 4 * r],
        [0.0, 0.0, 6 * r],
        [0.0, 0.0, 8 * r],
        [0.0, 0.0, 10 * r],
        [0.0, 0.0, 12 * r],
        [0.0, 0.0, 14 * r],
    ]
    return {
        "symbols": symbols,
        "geometry": _as_array(coords),
        # Keep a small active space for exact dense-matrix VQE tractability.
        "active_electrons": 8,
        "active_orbitals": 8,
        "charge": 0,
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run exact chemistry pipeline for H8 from a single script."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=["all", "scf", "vqe", "qsceom"],
        help="all: SCF + VQE + QSC-EOM, scf: SCF only, vqe: VQE only, qsceom: VQE+QSC-EOM",
    )
  
    parser.add_argument(
        "--method",
        default="pyscf",
        choices=["pyscf", "dhf"],
        help="Backend used by PennyLane molecular Hamiltonian builders.",
    )
    parser.add_argument(
        "--max-iter",
        default=500,
        type=int,
        help="Maximum optimizer iterations for exact VQE.",
    )
    parser.add_argument(
        "--shots",
        default=0,
        type=int,
        help="QSC-EOM shots (kept for compatibility; exact routine ignores nonzero).",
    )
    parser.add_argument(
        "--state-idx",
        default=1,
        type=int,
        help="QSC-EOM eigenvector index used for R1/R2 output.",
    )
    parser.add_argument(
        "--skip-files",
        action="store_true",
        help="Do not write fock/two-electron/R1R2 output files.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg = _default_problem()

    run_scf = args.mode in {"all", "scf"}
    run_vqe = args.mode in {"all", "vqe", "qsceom"}
    run_qsceom = args.mode in {"all", "qsceom"}

    if run_scf:
        print("\n[1/3] Running SCF...")
        scf_result = SCF.run_scf(
            cfg["symbols"],
            cfg["geometry"],
            charge=cfg["charge"],
            unit="bohr",
            basis="sto-3g",
            fock_output=None if args.skip_files else "fock.txt",
            two_e_output=None if args.skip_files else "two_elec.txt",
        )
        print(
            "SCF completed. Orbitals:",
            scf_result["n_spatial_orbitals"],
            "| converged:",
            scf_result["converged"],
        )

    params = None
    if run_vqe:
        print("\n[2/3] Running exact VQE...")
        params = vqeex.gs_exact(
            cfg["symbols"],
            cfg["geometry"],
            cfg["active_electrons"],
            cfg["active_orbitals"],
            cfg["charge"],
            method=args.method,
            max_iter=args.max_iter,
            basis="sto-3g",
            amplitudes_outfile=None if args.skip_files else "t1_t2.txt",
        )
        print("Returned parameter vector length:", len(params))

    if run_qsceom:
        if params is None:
            raise RuntimeError("QSC-EOM requested without optimized parameters.")

        print("\n[3/3] Running QSC-EOM...")
        qsceom.ee_exact(
            cfg["symbols"],
            cfg["geometry"],
            cfg["active_electrons"],
            cfg["active_orbitals"],
            cfg["charge"],
            params,
            shots=args.shots,
            method=args.method,
            state_idx=args.state_idx,
            basis="sto-3g",
            r1r2_outfile=None if args.skip_files else "out_r1_r2.txt",
        )
    


if __name__ == "__main__":
    main()
