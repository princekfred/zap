
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import SCF
import casci
import qsceom_exact
import symm
import vqee

OUTPUT_DIR = PROJECT_ROOT / "outputs" /"CH+1.5re"/ "CH+ Full operator"
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
        [0.0, 0.0, 1.5*re],
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
    The second such pair is the target 2^1Delta.
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
        default=None,
        type=int,
        help=(
            "Optional QSC-EOM eigenvector index. If omitted, the script auto-matches "
            "the 2^1Delta state using symmetry-resolved CASCI energies."
        ),
    )
    parser.add_argument(
        "--skip-files",
        action="store_true",
        help="Do not write fock/two-electron/R1R2/energy output files.",
    )
    return parser.parse_args()


def _write_r1r2_like_file(path, vector, det_list, active_electrons):
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

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


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
    r1r2_file = None if args.skip_files else str(OUTPUT_DIR / "r1_r2.txt")
    r_vectors_dir = None if args.skip_files else (OUTPUT_DIR / "r_vectors")
    qscex_ene_file = None if args.skip_files else str(OUTPUT_DIR / "qsceom_energy")
    casci_file = None if args.skip_files else str(OUTPUT_DIR / "CASCI_output.txt")
    gap_file = None if args.skip_files else str(OUTPUT_DIR / "energy_gap_2_1Delta.txt")

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

    params = None
    if run_vqe:
        print("\n[2/3] Running VQE...")
        params = vqee.gs_exact(
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
        # until the target root is resolved.
        first_idx = 0 if target_idx is None else target_idx
        print("\n[3/3] Running QSC-EOM...")
        eig, eigvecs, det_list = qsceom_exact.ee_exact(
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
            r1r2_outfile=None,
            hamiltonian=shared_hamiltonian,
            qubits=shared_qubits,
            return_eigvecs=True,
        )
        print("QSC-EOM energies (Hartree):", eig)

        if target_idx is None:
            if ref_delta is None:
                raise RuntimeError(
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

        print("Using QSC-EOM excited state index:", target_idx)

        if target_idx < 0 or target_idx >= len(eig):
            raise ValueError(
                f"Resolved target state index {target_idx} is out of range 0..{len(eig)-1}."
            )
        eigvec = eigvecs[:, target_idx]

        if r1r2_file:
            _write_r1r2_like_file(
                r1r2_file,
                eigvec,
                det_list,
                cfg["active_electrons"],
            )

        sort_order = sorted(range(int(len(eig))), key=lambda idx: float(eig[idx]))
        n_roots = min(25, len(sort_order), int(eigvecs.shape[1]))
        dominant_by_root = {}
        try:
            root_symm = symm.summarize_lowest_qsceom_roots_r1r2(
                eig,
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

        lowest25_rows = []
        for soln_idx, root_idx in enumerate(sort_order[:n_roots]):
            energy = float(eig[root_idx])
            dominant = dominant_by_root.get(int(root_idx), "unknown")
            lowest25_rows.append((soln_idx, root_idx, energy, dominant))

        print("Lowest 25 eigenvalue/irrep table (ascending):")
        print("soln_idx\teigenvalue_hartree\tdominant_irrep")
        for soln_idx, _root_idx, energy, dominant in lowest25_rows:
            print(f"{soln_idx}\t{energy:.12f}\t{dominant}")

        if r_vectors_dir is not None:
            r_vectors_dir.mkdir(parents=True, exist_ok=True)
            for soln_idx, root_idx, energy, dominant in lowest25_rows:
                out_path = r_vectors_dir / f"r1_r2_soln{soln_idx}"
                _write_r1r2_like_file(
                    out_path,
                    eigvecs[:, root_idx],
                    det_list,
                    cfg["active_electrons"],
                )
            print(f"Wrote {n_roots} R-vector files to: {r_vectors_dir}")

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

        if ref_delta is not None:
            excited_ref = float(ref_delta["second_1Delta_energy"])
        elif casci_energies is not None and len(casci_energies) > target_idx:
            excited_ref = float(casci_energies[target_idx])

        if excited_ref is not None:
            excited_err_h = excited_energy - excited_ref
            excited_err_ev = excited_err_h * HARTREE_TO_EV

        print(
            "CH+ 2^1Delta energy difference (Excited - Ground):",
            f"{gap_h:.12f} Hartree = {gap_ev:.6f} eV",
        )

        if ground_err_h is not None:
            print(
                "Ground-state error (QSC-EOM - CASCI):",
                f"{ground_err_h:.12f} Hartree = {ground_err_ev:.6f} eV",
            )

        if excited_err_h is not None:
            print(
                "2^1Delta excited-state error (QSC-EOM - CASCI):",
                f"{excited_err_h:.12f} Hartree = {excited_err_ev:.6f} eV",
            )

        if ref_delta is not None and ground_ref is not None:
            ref_gap_h = float(ref_delta["second_1Delta_energy"]) - float(casci_energies[0])
            ref_gap_ev = ref_gap_h * HARTREE_TO_EV
            err_ev = gap_ev - ref_gap_ev
            print(
                "excitation-energy error (QSC-EOM - CASCI ref):",
                f"{err_ev:.6f} eV",
            )

        # Symmetry decomposition of the selected QSC-EOM eigenvector R.
        try:
            symm_info = symm.analyze_qsceom_eigenvector_r1r2(
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
        except Exception as exc:
            print("Symmetry analysis skipped:", exc)

        if qscex_ene_file:
            with open(qscex_ene_file, "w", encoding="utf-8") as f:
                f.write("soln_idx\teigenvalue_hartree\tdominant_irrep\n")
                for soln_idx, _root_idx, energy, dominant in lowest25_rows:
                    f.write(f"{soln_idx}\t{energy:.12f}\t{dominant}\n")

        if gap_file:
            with open(gap_file, "w", encoding="utf-8") as f:
                f.write("state_label\t2^1Delta\n")
                f.write(f"state_idx\t{target_idx}\n")
                f.write(f"ground_energy_hartree\t{ground_energy:.12f}\n")
                f.write(f"excited_energy_hartree\t{excited_energy:.12f}\n")
                f.write(f"energy_difference_hartree\t{gap_h:.12f}\n")
                f.write(f"energy_difference_eV\t{gap_ev:.8f}\n")
                if ground_ref is not None:
                    f.write(f"reference_ground_energy_hartree\t{ground_ref:.12f}\n")
                    f.write(f"ground_state_error_hartree\t{ground_err_h:.12f}\n")
                    f.write(f"ground_state_error_eV\t{ground_err_ev:.8f}\n")
                if excited_ref is not None:
                    f.write(f"reference_excited_energy_hartree\t{excited_ref:.12f}\n")
                    f.write(f"excited_state_error_hartree\t{excited_err_h:.12f}\n")
                    f.write(f"excited_state_error_eV\t{excited_err_ev:.8f}\n")

        if casci_energies is not None and len(casci_energies) > target_idx:
            casci_gap_h = float(casci_energies[target_idx] - casci_energies[0])
            casci_gap_ev = casci_gap_h * HARTREE_TO_EV
            print(
                "CASCI energy difference at the same index:",
                f"{casci_gap_h:.12f} Hartree = {casci_gap_ev:.6f} eV",
            )


if __name__ == "__main__":
    main()
