"""Execute all notebook code cells locally with an existing Python environment.

Only Kaggle filesystem roots and the pip-install cell are replaced. Each
notebook runs in a fresh subprocess; notebook files/outputs are not modified.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def execute(path, bundle_dir, output_dir):
    notebook = json.loads(path.read_text())
    namespace = {"__name__": "__main__"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        source = "".join(source) if isinstance(source, list) else source
        compile(source, f"{path.name}:cell-{index}", "exec")
        if "'pip', 'install'" in source:
            print("Skipping pip installation: using the existing validation environment", flush=True)
            continue
        source = source.replace("INPUT_ROOT = Path('/kaggle/input')", f"INPUT_ROOT = Path({str(bundle_dir)!r})")
        source = source.replace("WORK_ROOT = Path('/kaggle/working')", f"WORK_ROOT = Path({str(output_dir)!r})")
        exec(compile(source, f"{path.name}:cell-{index}", "exec"), namespace)
    assert namespace["cfg"]["mode"] == "smoke"
    assert namespace["result"]["test"]["protocol"] == "adrec_upstream"
    assert len(namespace["recommended_ids"]) == 10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", default="work/kaggle_upload")
    parser.add_argument("--output-dir", default="work/kaggle_notebook_checks")
    parser.add_argument("--execute", type=Path)
    args = parser.parse_args()
    bundle, output = Path(args.bundle_dir).resolve(), Path(args.output_dir).resolve()
    if args.execute:
        execute(args.execute.resolve(), bundle, output)
        return
    if output.exists():
        raise FileExistsError("Choose a fresh check output directory")
    output.mkdir(parents=True)
    checks = []
    for notebook in sorted((ROOT / "notebooks").glob("*_kaggle.ipynb")):
        started = time.monotonic()
        log = output / (notebook.stem + ".log")
        with log.open("w") as handle:
            completed = subprocess.run([sys.executable, "-m", "scripts.check_kaggle_notebooks",
                "--execute", str(notebook), "--bundle-dir", str(bundle),
                "--output-dir", str(output / notebook.stem)], cwd=ROOT,
                stdout=handle, stderr=subprocess.STDOUT)
        checks.append({"notebook": notebook.name, "passed": completed.returncode == 0,
                       "seconds": round(time.monotonic() - started, 2)})
        print(checks[-1], flush=True)
        if completed.returncode:
            print(log.read_text()[-8000:])
            raise RuntimeError(f"Notebook failed; see {log}")
    (output / "summary.json").write_text(json.dumps({"environment": "local CPU",
        "pip_install_tested": False, "kaggle_gpu_tested": False, "checks": checks}, indent=2) + "\n")


if __name__ == "__main__":
    main()
