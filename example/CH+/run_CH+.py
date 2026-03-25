
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import SCF
import casci
import qsceom
import symm
import vqe

OUTPUT_DIR = PROJECT_ROOT / "outputs" /"CH+2re"/ "CH+ Trotterized"
HARTREE_TO_EV = 27.211386245988


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
    re = 2.13713
    symbols = ["C", "H"]
    coords = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2*re],
    ]
    return {
        "symbols": symbols,
        "geometry": _as_array(coords),
        "active_electrons": 6,
        "active_orbitals": 6,
        "charge": 1,
        "basis": "6-31g",
        "unit": "bohr",
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run CH+ workflow from a single script."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=["all", "scf", "vqe", "qsceom"],
        help="all: SCF + vqe + QSC-EOM, scf: SCF only, vqe: VQE only, qsceom: VQE+QSC-EOM",
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
        help="Shots used by QSC-EOM.",
    )
    parser.add_argument(
        "--state-idx",
        default=1,
        type=int,
        help="QSC-EOM eigenvector index used for R1/R2 output and gap reporting.",
    )
    parser.add_argument(
        "--skip-files",
        action="store_true",
        help="Do not write fock/two-electron/R1R2/energy output files.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg = _default_problem()

    run_scf = args.mode in {"all", "scf"}
    run_vqe = args.mode in {"all", "vqe", "qsceom"}
    run_qsceom = args.mode in {"all", "qsceom"}
    target_idx = int(args.state_idx)

    if not args.skip_files:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fock_file = None if args.skip_files else str(OUTPUT_DIR / "fock.txt")
    two_e_file = None if args.skip_files else str(OUTPUT_DIR / "two_elec.txt")
    amp_file = None if args.skip_files else str(OUTPUT_DIR / "t1_t2.txt")
    r1r2_file = None if args.skip_files else str(OUTPUT_DIR / "r1_r2.txt")
    qscex_ene_file = None if args.skip_files else str(OUTPUT_DIR / "qsceom_energy")
    casci_file = None if args.skip_files else str(OUTPUT_DIR / "CASCI_output.txt")
    qsc_symm_file = None if args.skip_files else str(OUTPUT_DIR / "qsceom_symmetry.txt")
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
            count_space="active",
            fock_output=fock_file,
            two_e_output=two_e_file,
        )
        print(
            "SCF completed. Orbitals:",
            scf_result["reported_n_spatial_orbitals"],
            "| converged:",
            scf_result["converged"],
        )

    shared_hamiltonian = None
    shared_qubits = None
    casci_energies = None
    if run_vqe:
        shared_hamiltonian, shared_qubits, casci_energies = (
            casci.build_casci_hamiltonian_from_problem(
                cfg,
                n_excited=None,  # request full CASCI root list
                casci_output_path=casci_file,
            )
        )
        print(
            "Loaded CASCI/FCI Hamiltonian from casci.py with",
            shared_qubits,
            "qubits.",
        )
        print("CASCI excited states generated:", max(len(casci_energies) - 1, 0))

    params = None
    if run_vqe:
        print("\n[2/3] Running VQE...")
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
            hamiltonian=shared_hamiltonian,
            qubits=shared_qubits,
        )
        print("Returned parameter vector length:", len(params))

    if run_qsceom:
        if params is None:
            raise RuntimeError("QSC-EOM requested without optimized parameters.")

        print("\n[3/3] Running QSC-EOM...")
        eig, eigvec, det_list = qsceom.ee_exact(
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
            state_idx=target_idx,
            r1r2_outfile=r1r2_file,
            hamiltonian=shared_hamiltonian,
            qubits=shared_qubits,
            return_vector=True,
        )
        print("QSC-EOM energies (Hartree):", eig)

        print("Using QSC-EOM excited state index:", target_idx)

        if target_idx < 0 or target_idx >= len(eig):
            raise ValueError(
                f"Resolved target state index {target_idx} is out of range 0..{len(eig)-1}."
            )

        ground_energy = float(eig[0])
        excited_energy = float(eig[target_idx])
        gap_h = excited_energy - ground_energy
        gap_ev = gap_h * HARTREE_TO_EV

        ground_ref = None
        excited_ref = None
        ground_err_h = None
        ground_err_ev = None
        excited_err_h = None
        excited_err_ev = None

        if casci_energies is not None and len(casci_energies) > 0:
            ground_ref = float(casci_energies[0])
            ground_err_h = ground_energy - ground_ref
            ground_err_ev = ground_err_h * HARTREE_TO_EV

        if casci_energies is not None and len(casci_energies) > target_idx:
            excited_ref = float(casci_energies[target_idx])

        if excited_ref is not None:
            excited_err_h = excited_energy - excited_ref
            excited_err_ev = excited_err_h * HARTREE_TO_EV

        print(
            f"CH+ state[{target_idx}] energy difference (Excited - Ground):",
            f"{gap_h:.12f} Hartree = {gap_ev:.6f} eV",
        )

        if ground_err_h is not None:
            print(
                "Ground-state error (QSC-EOM - CASCI):",
                f"{ground_err_h:.12f} Hartree = {ground_err_ev:.6f} eV",
            )

        if excited_err_h is not None:
            print(
                f"state[{target_idx}] excited-state error (QSC-EOM - CASCI):",
                f"{excited_err_h:.12f} Hartree = {excited_err_ev:.6f} eV",
            )

        # Symmetry decomposition of the selected QSC-EOM eigenvector R.
        try:
            symm_info = symm.analyze_qsceom_eigenvector(
                eigvec,
                det_list,
                symbols=cfg["symbols"],
                geometry=cfg["geometry"],
                active_electrons=cfg["active_electrons"],
                active_orbitals=cfg["active_orbitals"],
                charge=cfg["charge"],
                basis=cfg["basis"],
                unit=cfg["unit"],
                point_group="C2v",
            )
            symm.print_sym_info_(symm_info["weights_by_irrep"], symm_info["groupname"])

            if qsc_symm_file:
                symm_report = symm.format_sym_info_(
                    symm_info["weights_by_irrep"], symm_info["groupname"]
                )
                with open(qsc_symm_file, "w", encoding="utf-8") as f:
                    f.write(symm_report)
        except Exception as exc:
            print("Symmetry analysis skipped:", exc)

        if qscex_ene_file:
            with open(qscex_ene_file, "w", encoding="utf-8") as f:
                for value in eig:
                    f.write(f"{float(value)}\n")

        if casci_energies is not None and len(casci_energies) > target_idx:
            casci_gap_h = float(casci_energies[target_idx] - casci_energies[0])
            casci_gap_ev = casci_gap_h * HARTREE_TO_EV
            print(
                "CASCI energy difference at the same index:",
                f"{casci_gap_h:.12f} Hartree = {casci_gap_ev:.6f} eV",
            )


if __name__ == "__main__":
    main()
