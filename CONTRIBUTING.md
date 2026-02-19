# Contributing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Development Notes

- Keep `run_exact.py` as the single user-facing entrypoint.
- Preserve excitation label format in `out_r1_r2.txt`:
  - singles: `p^ h`
  - doubles: `p^ h; p^ h`
- Generated outputs (`fock.txt`, `two_elec.txt`, `out_r1_r2.txt`) should not be committed.
- Follow `CODE_OF_CONDUCT.md` in all interactions.

## Basic Checks

```bash
python -m py_compile run_exact.py SCF.py vqe.py qsceom.py exc.py fun.py
python run_exact.py --help
ruff check .
```

## Pull Requests

- Keep changes focused and well-scoped.
- Include a short summary of scientific/algorithmic impact.
- Add or update documentation when behavior changes.
- Use the PR checklist in `.github/pull_request_template.md`.
