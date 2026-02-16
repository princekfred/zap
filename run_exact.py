import vqeex
import sys


def main():
    try:
        from pennylane import numpy as pnp
    except ModuleNotFoundError:
        pnp = None

    r = 1.88973
    symbols = ["H", "H", "H", "H"]
    coords = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 3.0 * r],
        [0.0, 0.0, 6.0 * r],
        [0.0, 0.0, 9.0 * r],
    ]
    geometry = (
        pnp.array(coords, dtype=float, requires_grad=False) if pnp is not None else coords
    )

    active_electrons = 4
    active_orbitals = 4
    charge = 0

    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "vqe"
    extra = [a.lower() for a in sys.argv[2:]]
    if mode in {"pyscf", "dhf"}:
        extra = [mode, *extra]
        mode = "vqe"
    method = "dhf" if "dhf" in extra else "pyscf"

    if mode == "qsceom":
        import qsceom

        params = vqeex.gs_exact(
            symbols,
            geometry,
            active_electrons,
            active_orbitals,
            charge,
            max_iter=100,
            method=method,
        )
        out = qsceom.ee_exact(
            symbols,
            geometry,
            active_electrons,
            active_orbitals,
            charge,
            params,
            method=method,
        )
        print("\nGround energy:", out["ground_energy"])
        print("Lowest excitation energies:", list(out["excitation_energies"][:10]))
    else:
        params = vqeex.gs_exact(
            symbols,
            geometry,
            active_electrons,
            active_orbitals,
            charge,
            max_iter=500,
            method=method,
        )
    print("\nReturned parameter vector length:", len(params))


if __name__ == "__main__":
    main()
