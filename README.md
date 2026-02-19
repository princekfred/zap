# UCCSD + QSC-EOM (PennyLane/PySCF)

Reference implementation of:
- RHF orbital analysis with PySCF
- UCCSD VQE ground-state optimization with PennyLane
- QSC-EOM excited-state analysis with `R1/R2` coefficient export

## Repository Layout

- `run_exact.py`: canonical CLI entrypoint
- `SCF.py`: RHF + orbital/two-electron integral report generation
- `vqe.py`: UCCSD VQE optimizer
- `qsceom.py`: QSC-EOM matrix build + `R1/R2` export
- `exc.py`: determinant/excitation configuration generator
- `example/*/run_*.py`: per-system runnable examples (H4, H8, N2, CH-)

## Requirements

- Python 3.10+
- Dependencies from `requirements.txt`

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional editable install:

```bash
python -m pip install -e .
run-exact --help
```

## Quick Start

Run the full H4 pipeline:

```bash
python run_exact.py
```

Run individual stages:

```bash
python run_exact.py scf
python run_exact.py vqe
python run_exact.py qsceom
```

Choose preset systems:

```bash
python run_exact.py all --system h4
python run_exact.py all --system h8
python run_exact.py all --system n2
python run_exact.py all --system ch-
```

Other common options:

```bash
python run_exact.py all --max-iter 700
python run_exact.py qsceom --state-idx 1
python run_exact.py all --skip-files
python run_exact.py all --method pyscf
```

## Outputs

By default, the workflow writes:
- `fock.txt`: SCF orbital summary
- `two_elec.txt`: antisymmetrized two-electron spin-orbital integrals
- `t1_t2.txt`: optimized UCCSD amplitudes
- `out_r1_r2.txt`: QSC-EOM `R1/R2` excitation coefficients

## Development

```bash
python -m py_compile run_exact.py SCF.py vqe.py qsceom.py exc.py fun.py
python run_exact.py --help
```

## Community and GitHub Standards

- Contributions: `CONTRIBUTING.md`
- Code of conduct: `CODE_OF_CONDUCT.md`
- Security policy: `SECURITY.md`
- Pull request template: `.github/pull_request_template.md`
- Issue templates: `.github/ISSUE_TEMPLATE/`

## License

MIT. See `LICENSE`.
