# Exact UCCSD + QSC-EOM (PennyLane/PySCF)

This repository contains a small quantum chemistry workflow for:

- SCF orbital analysis with PySCF
- exact (non-Trotterized) UCCSD VQE
- QSC-EOM excited-state analysis with `R1/R2` coefficient output

The main entrypoint is `run_exact.py`.

## Repository Layout

- `run_exact.py`: single CLI entrypoint for the full workflow
- `vqeex.py`: exact UCCSD VQE optimizer (dense matrix exponential)
- `qsceom.py`: QSC-EOM matrix build and `R1/R2` export
- `SCF.py`: RHF + orbital/two-electron integral report generation
- `exc.py`: determinant list generator used by QSC-EOM
- `fun.py`: optional compatibility import helper

## Requirements

- Python 3.10+ recommended
- Dependencies listed in `requirements.txt`

Install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional editable install with console script:

```bash
python -m pip install -e .
run-exact --help
```

## Quick Start

Run full pipeline (SCF + VQE + QSC-EOM):

```bash
python run_exact.py
```

Run selected stages:

```bash
python run_exact.py scf
python run_exact.py vqe
python run_exact.py qsceom
```

Useful options:

```bash
python run_exact.py qsceom --state-idx 1
python run_exact.py all --max-iter 700
python run_exact.py all --skip-files
python run_exact.py all --method pyscf
```

## Outputs

By default, the workflow writes:

- `fock.txt`: SCF orbital summary
- `two_elec.txt`: antisymmetrized two-electron MO spin-orbital integrals
- `out_r1_r2.txt`: QSC-EOM `R1/R2` excitation coefficients
- `t1_t2.txt`: optimized UCCSD amplitudes (`t2` then `t1`)

These files are generated artifacts and are ignored by Git (`.gitignore`).

## Notes and Limitations

- `vqeex.py` uses dense matrices (`2**n x 2**n`), so it is only practical for
  small active spaces.
- `shots` is accepted in CLI for compatibility, but QSC-EOM currently uses
  exact statevector evaluation.
- Numerical degeneracies can change eigenvector basis choices; `qsceom.py`
  includes deterministic phase/symmetry handling for stable `R1/R2` output.

## License

This project is released under the MIT License. See `LICENSE`.

## CI

A GitHub Actions workflow is included at `.github/workflows/ci.yml` for
dependency install, syntax checks, and a CLI smoke test.
