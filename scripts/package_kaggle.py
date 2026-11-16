"""Make an explicit, checksummed source bundle for an unpushed local branch."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

from src.data.load import load_dataset
from src.data.catalog import dataset_spec

ROOT = Path(__file__).resolve().parents[1]


def package(output_dir, dataset=None, dataset_name="baby"):
    spec = dataset_spec(dataset_name)
    if dataset:
        load_dataset(dataset, spec["sha256"])
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for folder in ("src", "configs", "third_party", "notebooks"):
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and ".ipynb_checkpoints" not in path.parts:
                if path.suffix in {".py", ".json", ".yaml", ".txt", ".md", ".ipynb"} or path.name == "LICENSE":
                    files.append(path)
    files += [ROOT / "requirements-shared.txt", ROOT / "docs/KAGGLE_TRAINING.md",
              ROOT / "docs/MULTI_DATASET_TRAINING.md"]
    manifest = {
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "note": "Working-copy snapshot, including local changes; hashes identify the actual code.",
        "files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},
    }
    archive = output / "cmpe257_kaggle_code.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(files):
            bundle.write(path, path.relative_to(ROOT).as_posix())
        bundle.writestr("bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")
    if dataset:
        destination = output / "dataset.pkl"
        if Path(dataset).resolve() != destination:
            shutil.copyfile(dataset, destination)
    print(f"Code: {archive}\nSHA256: {hashlib.sha256(archive.read_bytes()).hexdigest()}")
    if dataset:
        print(f"Data: {output / 'dataset.pkl'}")
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="work/kaggle_upload")
    parser.add_argument("--dataset", help="Optional local dataset.pkl to copy beside the source ZIP")
    parser.add_argument("--dataset-name", default="baby", choices=["baby", "beauty", "ml-100k"])
    args = parser.parse_args()
    package(args.output_dir, args.dataset, args.dataset_name)
