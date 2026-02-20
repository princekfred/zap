"""Single entrypoint for CH+ SCF + exact VQE + QSC-EOM workflow."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import SCF
import qsceom
import vqe

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "CH+_trotterized"


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
    symbols = ["C", "H"]
    coords = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.12],
    ]
    return {
        "symbols": symbols,
        "geometry": _as_array(coords),
        "active_electrons": 4,
        "active_orbitals": 4,
        "charge": 1,
        "basis": "6-31g",
        "unit": "angstrom",
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run CH+ chemistry pipeline from a single script."
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
        help="Do not write fock/two-electron/R1R2/QSC-EOM-energy output files.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg = _default_problem()

    run_scf = args.mode in {"all", "scf"}
    run_vqe = args.mode in {"all", "vqe", "qsceom"}
    run_qsceom = args.mode in {"all", "qsceom"}

    if not args.skip_files:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fock_file = None if args.skip_files else str(OUTPUT_DIR / "fock.txt")
    two_e_file = None if args.skip_files else str(OUTPUT_DIR / "two_elec.txt")
    amp_file = None if args.skip_files else str(OUTPUT_DIR / "t1_t2.txt")
    r1r2_file = None if args.skip_files else str(OUTPUT_DIR / "out_r1_r2.txt")
    qscex_ene_file = None if args.skip_files else str(OUTPUT_DIR / "qscex_ene")

    if run_scf:
        print("\n[1/3] Running SCF...")
        scf_result = SCF.run_scf(
            cfg["symbols"],
            cfg["geometry"],
            charge=cfg["charge"],
            basis=cfg["basis"],
            unit=cfg["unit"],
            active_electrons=cfg["active_electrons"],
            active_orbitals=cfg["active_orbitals"],
            fock_output=fock_file,
            two_e_output=two_e_file,
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
        params = vqe.gs_exact(
            cfg["symbols"],
            cfg["geometry"],
            cfg["active_electrons"],
            cfg["active_orbitals"],
            cfg["charge"],
            method=args.method,
            basis=cfg["basis"],
            unit=cfg["unit"],
            max_iter=args.max_iter,
            amplitudes_outfile=amp_file,
        )
        print("Returned parameter vector length:", len(params))

    if run_qsceom:
        if params is None:
            raise RuntimeError("QSC-EOM requested without optimized parameters.")

        print("\n[3/3] Running QSC-EOM...")
        eig = qsceom.ee_exact(
            cfg["symbols"],
            cfg["geometry"],
            cfg["active_electrons"],
            cfg["active_orbitals"],
            cfg["charge"],
            params,
            shots=args.shots,
            method=args.method,
            basis=cfg["basis"],
            unit=cfg["unit"],
            state_idx=args.state_idx,
            r1r2_outfile=r1r2_file,
        )
        print("QSC-EOM energies (Hartree):", eig)

        if qscex_ene_file:
            with open(qscex_ene_file, "w", encoding="utf-8") as f:
                for value in eig:
                    f.write(f"{float(value)}\n")


if __name__ == "__main__":
    main()
