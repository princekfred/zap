"""Single entrypoint for CH+ paper workflow (VQE + QSC-EOM).

Paper setup:
- State: 2^1 Delta of CH+
- Geometry: 2 * Re = 2 * 2.13713 bohr = 4.27426 bohr
- Basis: 6-31G
Reference: J. Chem. Theory Comput. 2024, 20, 9032-9040
DOI: 10.1021/acs.jctc.4c01071
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import SCF
import casci
import qsceom
import vqe

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "CH+ Trotterized"
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


def _reference_2_1_delta_energy(cfg, nroots_per_component=12, tol=1e-6, singlet_s2_tol=1e-3):
    """Return CH+ 2^1Delta reference from C2v (A1 + A2) singlet subspace.

    In C2v, each singlet Delta manifold appears as a near-degenerate A1/A2 pair.
    The second such pair is the paper target 2^1Delta.
    """
    try:
        import numpy as np
        from pyscf import fci, gto, mcscf, scf
    except Exception:
        return None

    mol = gto.Mole()
    coords = [
        [cfg["symbols"][i], [float(x) for x in cfg["geometry"][i]]]
        for i in range(len(cfg["symbols"]))
    ]
    mol.atom = coords
    mol.basis = cfg["basis"]
    mol.charge = int(cfg["charge"])
    mol.unit = cfg["unit"]
    mol.symmetry = "C2v"
    mol.build()

    mf = scf.RHF(mol)
    mf.max_cycle = 200
    mf.conv_tol = 1e-9
    mf.kernel()
    if not mf.converged:
        return None

    ncas = int(cfg["active_orbitals"])
    nelecas = int(cfg["active_electrons"])

    def _roots_for(sym, nroots):
        mc = mcscf.CASCI(mf, ncas, nelecas)
        solver = fci.direct_spin0_symm.FCI(mol)
        solver.wfnsym = sym
        solver.nroots = int(nroots)
        mc.fcisolver = solver
        mc.kernel()
        energies = np.atleast_1d(mc.e_tot).astype(float)
        ci_roots = mc.ci if isinstance(mc.ci, (list, tuple)) else [mc.ci]
        entries = []
        for root_idx, energy in enumerate(energies):
            s2_val = float("nan")
            if root_idx < len(ci_roots):
                try:
                    s2_val = float(mc.fcisolver.spin_square(ci_roots[root_idx], ncas, nelecas)[0])
                except Exception:
                    pass
            entries.append(
                {
                    "root_index": int(root_idx),
                    "energy": float(energy),
                    "s2": s2_val,
                }
            )
        entries.sort(key=lambda item: item["energy"])
        return entries

    roots_by_irrep = {
        "A1": _roots_for("A1", nroots_per_component),
        "A2": _roots_for("A2", nroots_per_component),
        "B1": _roots_for("B1", nroots_per_component),
        "B2": _roots_for("B2", nroots_per_component),
    }
    a1_roots = [
        entry for entry in roots_by_irrep["A1"] if abs(float(entry["s2"])) <= singlet_s2_tol
    ]
    a2_roots = [
        entry for entry in roots_by_irrep["A2"] if abs(float(entry["s2"])) <= singlet_s2_tol
    ]
    if not a1_roots or not a2_roots:
        return None

    # Build one-to-one A1/A2 near-degenerate pairs for Delta manifolds.
    used_a2 = set()
    delta_pairs = []
    a2_energies = np.array([item["energy"] for item in a2_roots], dtype=float)
    for a1_idx, a1_entry in enumerate(a1_roots):
        e_a1 = float(a1_entry["energy"])
        a2_idx = int(np.argmin(np.abs(a2_energies - e_a1)))
        if a2_idx in used_a2:
            continue
        a2_entry = a2_roots[a2_idx]
        e_a2 = float(a2_entry["energy"])
        if abs(e_a1 - e_a2) > tol:
            continue
        used_a2.add(a2_idx)
        pair_energy = 0.5 * (e_a1 + e_a2)
        delta_pairs.append(
            {
                "pair_energy": float(pair_energy),
                "a1_root_index": int(a1_entry["root_index"]),
                "a2_root_index": int(a2_entry["root_index"]),
                "a1_energy": e_a1,
                "a2_energy": e_a2,
            }
        )

    delta_pairs.sort(key=lambda item: item["pair_energy"])
    for idx, pair in enumerate(delta_pairs, start=1):
        pair["delta_order"] = idx
        pair["delta_label"] = f"{idx}^1Delta"

    if len(delta_pairs) < 2:
        return None

    return {
        "converged_scf_energy": float(mf.e_tot),
        "roots_by_irrep": {
            key: [dict(item) for item in values]
            for key, values in roots_by_irrep.items()
        },
        "delta_pairs": delta_pairs,
        "first_1Delta_energy": float(delta_pairs[0]["pair_energy"]),
        "second_1Delta_energy": float(delta_pairs[1]["pair_energy"]),  # 2^1Delta
        "subspace": "A1+A2 (C2v)",
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run CH+ paper workflow from a single script."
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
        default=None,
        type=int,
        help=(
            "Optional QSC-EOM eigenvector index. If omitted, the script auto-matches "
            "the paper 2^1Delta state using symmetry-resolved CASCI energies."
        ),
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
    target_idx = None if args.state_idx is None else int(args.state_idx)

    if not args.skip_files:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fock_file = None if args.skip_files else str(OUTPUT_DIR / "fock.txt")
    two_e_file = None if args.skip_files else str(OUTPUT_DIR / "two_elec.txt")
    amp_file = None if args.skip_files else str(OUTPUT_DIR / "t1_t2.txt")
    r1r2_file = None if args.skip_files else str(OUTPUT_DIR / "out_r1_r2.txt")
    qscex_ene_file = None if args.skip_files else str(OUTPUT_DIR / "qsceom_ene")
    casci_file = None if args.skip_files else str(OUTPUT_DIR / "CASCI_output.txt")
    gap_file = None if args.skip_files else str(OUTPUT_DIR / "energy_gap_2_1Delta.txt")
    label_file = None if args.skip_files else str(OUTPUT_DIR / "state_labels_c2v.txt")

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
    ref_delta = None
    if run_vqe:
        n_excited_for_casci = max(target_idx, 1) if target_idx is not None else 40
        shared_hamiltonian, shared_qubits, casci_energies = (
            casci.build_casci_hamiltonian_from_problem(
                cfg,
                n_excited=n_excited_for_casci,
                casci_output_path=casci_file,
            )
        )
        print(
            "Loaded CASCI/FCI Hamiltonian from casci.py with",
            shared_qubits,
            "qubits.",
        )
        print("CASCI excited states generated:", max(len(casci_energies) - 1, 0))
        ref_delta = _reference_2_1_delta_energy(cfg)
        if ref_delta is not None:
            print(
                "Symmetry CASCI references:",
                f"1^1Delta={ref_delta['first_1Delta_energy']:.12f} Ha,",
                f"2^1Delta={ref_delta['second_1Delta_energy']:.12f} Ha",
            )
            print("C2v A1+A2 Delta manifolds identified from near-degenerate pairs:")
            for pair in ref_delta["delta_pairs"]:
                print(
                    f"  {pair['delta_label']}: E={pair['pair_energy']:.12f} Ha "
                    f"(A1 root {pair['a1_root_index']}, A2 root {pair['a2_root_index']})"
                )
            if label_file:
                with open(label_file, "w", encoding="utf-8") as f:
                    f.write("point_group\tC2v\n")
                    f.write("target_subspace\tA1+A2\n")
                    f.write(
                        f"converged_scf_energy_hartree\t"
                        f"{ref_delta['converged_scf_energy']:.12f}\n"
                    )
                    f.write("delta_manifolds\n")
                    for pair in ref_delta["delta_pairs"]:
                        f.write(
                            f"{pair['delta_label']}\tE={pair['pair_energy']:.12f}\t"
                            f"A1_root={pair['a1_root_index']}\t"
                            f"A2_root={pair['a2_root_index']}\n"
                        )
                    f.write("roots_by_irrep\n")
                    for irrep in ("A1", "A2", "B1", "B2"):
                        for entry in ref_delta["roots_by_irrep"][irrep]:
                            f.write(
                                f"{irrep}[{entry['root_index']}]\t"
                                f"E={entry['energy']:.12f}\tS2={entry['s2']:.8f}\n"
                            )

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

        # First pass to collect eigenvalues. In auto-target mode, do not write R1/R2
        # until the 2^1Delta root is resolved.
        first_idx = 0 if target_idx is None else target_idx
        first_r1r2_file = None if target_idx is None else r1r2_file
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
            state_idx=first_idx,
            r1r2_outfile=first_r1r2_file,
            hamiltonian=shared_hamiltonian,
            qubits=shared_qubits,
        )
        print("QSC-EOM energies (Hartree):", eig)

        if target_idx is None:
            if ref_delta is None:
                raise RuntimeError(
                    "Could not identify paper 2^1Delta reference from symmetry CASCI. "
                    "Please pass --state-idx explicitly."
                )
            target_pair = ref_delta["delta_pairs"][1]
            ref_2_1_delta = float(target_pair["pair_energy"])
            target_idx = int((abs(eig - ref_2_1_delta)).argmin())
            print(
                "2^1Delta to QSC-EOM root:",
                f"state_idx={target_idx}",
                f"state_energy={float(eig[target_idx]):.12f} Ha",
                f"reference={ref_2_1_delta:.12f} Ha",
                f"(A1 root {target_pair['a1_root_index']} + A2 root {target_pair['a2_root_index']})",
            )

        # Ensure QSC-EOM is explicitly run with the resolved target state.
        if target_idx != first_idx:
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
                state_idx=target_idx,
                r1r2_outfile=r1r2_file,
                hamiltonian=shared_hamiltonian,
                qubits=shared_qubits,
            )

        print("Using QSC-EOM excited state index:", target_idx)

        if target_idx < 0 or target_idx >= len(eig):
            raise ValueError(
                f"Resolved target state index {target_idx} is out of range 0..{len(eig)-1}."
            )

        ground_energy = float(eig[0])
        excited_energy = float(eig[target_idx])
        gap_h = excited_energy - ground_energy
        gap_ev = gap_h * HARTREE_TO_EV

        print(
            "CH+ 2^1Delta energy difference (Excited - Ground):",
            f"{gap_h:.12f} Hartree = {gap_ev:.6f} eV",
        )

        if ref_delta is not None:
            ref_gap_h = float(ref_delta["second_1Delta_energy"]) - float(casci_energies[0])
            ref_gap_ev = ref_gap_h * HARTREE_TO_EV
            err_ev = gap_ev - ref_gap_ev
            print(
                "Paper-style excitation-energy error (QSC-EOM - CASCI ref):",
                f"{err_ev:.6f} eV",
            )

        if qscex_ene_file:
            with open(qscex_ene_file, "w", encoding="utf-8") as f:
                for value in eig:
                    f.write(f"{float(value)}\n")

        if gap_file:
            with open(gap_file, "w", encoding="utf-8") as f:
                f.write("state_label\t2^1Delta\n")
                f.write(f"state_idx\t{target_idx}\n")
                f.write(f"ground_energy_hartree\t{ground_energy:.12f}\n")
                f.write(f"excited_energy_hartree\t{excited_energy:.12f}\n")
                f.write(f"energy_difference_hartree\t{gap_h:.12f}\n")
                f.write(f"energy_difference_eV\t{gap_ev:.8f}\n")
                if ref_delta is not None:
                    ref_gap_h = float(ref_delta["second_1Delta_energy"]) - float(
                        casci_energies[0]
                    )
                    ref_gap_ev = ref_gap_h * HARTREE_TO_EV
                    f.write(
                        f"reference_2_1Delta_energy_hartree\t"
                        f"{float(ref_delta['second_1Delta_energy']):.12f}\n"
                    )
                    f.write(f"reference_gap_hartree\t{ref_gap_h:.12f}\n")
                    f.write(f"reference_gap_eV\t{ref_gap_ev:.8f}\n")
                    f.write(f"error_vs_reference_eV\t{(gap_ev - ref_gap_ev):.8f}\n")

        if casci_energies is not None and len(casci_energies) > target_idx:
            casci_gap_h = float(casci_energies[target_idx] - casci_energies[0])
            casci_gap_ev = casci_gap_h * HARTREE_TO_EV
            print(
                "CASCI energy difference at the same index:",
                f"{casci_gap_h:.12f} Hartree = {casci_gap_ev:.6f} eV",
            )


if __name__ == "__main__":
    main()
