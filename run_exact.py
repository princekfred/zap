"""Project CLI entrypoint for SCF + exact VQE + QSC-EOM workflows."""

from __future__ import annotations

import argparse
from typing import Any

import SCF
import qsceom
import vqe

BOHR_PER_ANGSTROM = 1.88973


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


def _problem(system: str) -> dict[str, Any]:
    r = BOHR_PER_ANGSTROM
    systems = {
        "h4": {
            "symbols": ["H", "H", "H", "H"],
            "geometry": _as_array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 3.0 * r],
                    [0.0, 0.0, 6.0 * r],
                    [0.0, 0.0, 9.0 * r],
                ]
            ),
            "active_electrons": 4,
            "active_orbitals": 4,
            "charge": 0,
            "basis": "sto-3g",
        },
        "h8": {
            "symbols": ["H", "H", "H", "H", "H", "H", "H", "H"],
            "geometry": _as_array([[0.0, 0.0, 2 * i * r] for i in range(8)]),
            "active_electrons": 8,
            "active_orbitals": 8,
            "charge": 0,
            "basis": "sto-3g",
        },
        "n2": {
            "symbols": ["N", "N"],
            "geometry": _as_array(
                [
                    [0.0, 0.0, 0.40 * r],
                    [0.0, 0.0, -0.5488 * r],
                ]
            ),
            "active_electrons": 6,
            "active_orbitals": 6,
            "charge": 0,
            "basis": "sto-6g",
        },
        "ch-": {
            "symbols": ["C", "H"],
            "geometry": _as_array(
                [
                    [0.0, 0.0, 0.5 * r],
                    [0.0, 0.0, -0.5 * r],
                ]
            ),
            "active_electrons": 4,
            "active_orbitals": 4,
            "charge": -1,
            "basis": "sto-3g",
        },
    }
    return systems[system]


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run exact chemistry workflow from a single CLI."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=["all", "scf", "vqe", "qsceom"],
        help="all: SCF + VQE + QSC-EOM, scf: SCF only, vqe: VQE only, qsceom: VQE+QSC-EOM",
    )
    parser.add_argument(
        "--system",
        default="h4",
        choices=["h4", "h8", "n2", "ch-"],
        help="Molecule/system preset.",
    )
    parser.add_argument(
        "--method",
        default="pyscf",
        choices=["pyscf", "dhf"],
        help="Backend for PennyLane molecular Hamiltonian builders.",
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
        help="QSC-EOM shots (0 means analytic expectation values).",
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
        help="Do not write fock/two-electron/t1_t2/R1R2 output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _problem(args.system)

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
            basis=cfg["basis"],
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
        params = vqe.gs_exact(
            cfg["symbols"],
            cfg["geometry"],
            cfg["active_electrons"],
            cfg["active_orbitals"],
            cfg["charge"],
            method=args.method,
            basis=cfg["basis"],
            max_iter=args.max_iter,
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
            basis=cfg["basis"],
            state_idx=args.state_idx,
            r1r2_outfile=None if args.skip_files else "out_r1_r2.txt",
        )


if __name__ == "__main__":
    main()
