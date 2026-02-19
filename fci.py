"""FCI helpers for molecular geometries using PySCF."""

from __future__ import annotations

import argparse


def active_space_fci_values(
    symbols,
    geometry,
    *,
    charge=0,
    basis="sto-3g",
    unit="bohr",
    active_electrons=None,
    active_orbitals=None,
    nroots=1,
    max_cycle=200,
    conv_tol=1e-9,
):
    """Compute active-space CASCI/FCI total energies for a molecular geometry.

    Returns a 1D NumPy array of total energies (Hartree), one per requested root.
    """
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'numpy'. Install with:\n  python -m pip install numpy"
        ) from exc

    try:
        from pyscf import gto, mcscf, scf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'pyscf'. Install with:\n  python -m pip install pyscf"
        ) from exc

    coords = np.asarray(geometry, dtype=float)

    mol = gto.Mole()
    mol.atom = [[symbols[i], coords[i].tolist()] for i in range(len(symbols))]
    mol.basis = basis
    mol.charge = int(charge)
    mol.unit = unit
    mol.build()

    mf = scf.RHF(mol)
    mf.max_cycle = int(max_cycle)
    mf.conv_tol = float(conv_tol)
    mf.kernel()

    if not mf.converged:
        raise RuntimeError("RHF failed to converge; cannot continue to FCI.")

    if active_electrons is None:
        active_electrons = int(mol.nelectron)
    if active_orbitals is None:
        active_orbitals = int(mol.nao_nr())

    if int(active_electrons) < 1:
        raise ValueError("`active_electrons` must be >= 1.")
    if int(active_orbitals) < 1:
        raise ValueError("`active_orbitals` must be >= 1.")

    casci = mcscf.CASCI(mf, int(active_orbitals), int(active_electrons))
    casci.fcisolver.nroots = int(max(1, nroots))
    energies = casci.kernel()[0]
    return np.array(energies, dtype=float).reshape(-1)


def _parse_symbols(text):
    symbols = [item.strip() for item in text.split(",") if item.strip()]
    if not symbols:
        raise ValueError("No symbols provided.")
    return symbols


def _parse_geometry(text):
    coords = []
    for atom in text.split(";"):
        atom = atom.strip()
        if not atom:
            continue
        xyz = [float(v.strip()) for v in atom.split(",")]
        if len(xyz) != 3:
            raise ValueError("Each geometry entry must have 3 comma-separated values.")
        coords.append(xyz)
    if not coords:
        raise ValueError("No geometry coordinates provided.")
    return coords


def _parse_args():
    parser = argparse.ArgumentParser(description="Compute active-space FCI energies.")
    parser.add_argument(
        "--symbols",
        required=True,
        help='Comma-separated symbols, e.g. "H,F".',
    )
    parser.add_argument(
        "--geometry",
        required=True,
        help='Semicolon-separated xyz triples',
    )
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--unit", default="angstrom", choices=["bohr", "angstrom"])
    parser.add_argument("--active-electrons", type=int, default=None)
    parser.add_argument("--active-orbitals", type=int, default=None)
    parser.add_argument("--nroots", type=int, default=1)
    parser.add_argument("--outfile", default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    symbols = _parse_symbols(args.symbols)
    geometry = _parse_geometry(args.geometry)
    if len(symbols) != len(geometry):
        raise ValueError("Number of symbols and geometry rows must match.")

    energies = active_space_fci_values(
        symbols,
        geometry,
        charge=args.charge,
        basis=args.basis,
        unit=args.unit,
        active_electrons=args.active_electrons,
        active_orbitals=args.active_orbitals,
        nroots=args.nroots,
    )
    print("FCI energies (Hartree):", energies)

    if args.outfile:
        with open(args.outfile, "w", encoding="utf-8") as f:
            for value in energies:
                f.write(f"{float(value)}\n")


if __name__ == "__main__":
    main()
