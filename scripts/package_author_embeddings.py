"""Package the Baby author-embedding experiment without modifying training code."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from scripts.package_kaggle import ROOT, package
from src.data.load import load_dataset


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package_inputs(checkpoint, dataset, output_dir):
    import torch

    checkpoint, dataset = Path(checkpoint).resolve(), Path(dataset).resolve()
    spec_path = ROOT / 'configs/author_baby_embeddings.json'
    spec = json.loads(spec_path.read_text())
    raw = checkpoint.read_bytes()
    blob = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
    if (hashlib.sha256(raw).hexdigest() != spec['sha256']
            or blob != spec['git_blob_sha1'] or len(raw) != spec['size_bytes']):
        raise ValueError('Checkpoint must be the exact pinned official Baby pretrain.pth')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    weight = state[spec['state_key']]
    if (list(weight.shape) != spec['embedding_shape'] or str(weight.dtype) != spec['dtype']
            or not torch.isfinite(weight).all()):
        raise ValueError('Invalid pretrained item embedding tensor')
    load_dataset(dataset, spec['dataset_sha256'])
    baselines = {
        'project_warmup5': ROOT / 'results/adrec/baby',
        'project_warmup0': ROOT / 'results/adrec_warmup_ablation/baby',
    }
    completion = json.loads((baselines['project_warmup0'] / 'completion.json').read_text())
    if not completion['complete'] or completion['completed_seeds'] != list(range(42, 47)):
        raise ValueError('Five completed historical no-warm-up seeds are required')
    files = {'dataset.pkl': dataset, 'authors/baby/pretrain.pth': checkpoint,
             'author_checkpoint.json': spec_path}
    for label, folder in baselines.items():
        for seed in range(42, 47):
            for suffix in ('config.json', 'results.json', 'environment.json', 'checkpoints.json'):
                if not (folder / f'adrec_seed{seed}_{suffix}').is_file():
                    raise FileNotFoundError(folder / f'adrec_seed{seed}_{suffix}')
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix in {'.json', '.csv', '.txt', '.md'}:
                files[f'baseline/{label}/{path.name}'] = path
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / 'adrec_baby_author_embeddings_inputs.zip'
    temporary = output / 'adrec_baby_author_embeddings_inputs.zip.tmp'
    with tempfile.TemporaryDirectory(prefix='adrec_author_source_') as scratch:
        code = package(scratch)
        with zipfile.ZipFile(code) as source:
            code_hash = hashlib.sha256(source.read('bundle_manifest.json')).hexdigest()
        manifest = {'experiment': 'adrec_baby_author_embeddings', 'dataset': 'baby',
                    'seeds': list(range(42, 47)), 'warmup_epochs': [5, 0],
                    'dataset_sha256': spec['dataset_sha256'], 'code_manifest_sha256': code_hash,
                    'files': {name: sha256(path) for name, path in files.items()}}
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as bundle:
            bundle.write(code, 'cmpe257_kaggle_code.zip')
            for name, path in files.items():
                bundle.write(path, name)
            bundle.writestr('author_inputs_manifest.json', json.dumps(manifest, indent=2) + '\n')
        with zipfile.ZipFile(temporary) as bundle:
            if bundle.testzip() is not None:
                raise ValueError('ZIP integrity check failed')
            for name, expected in manifest['files'].items():
                if hashlib.sha256(bundle.read(name)).hexdigest() != expected:
                    raise ValueError(f'Archive hash mismatch: {name}')
    temporary.replace(archive)
    notebook = ROOT / 'notebooks/adrec_author_embeddings_kaggle.ipynb'
    shutil.copyfile(notebook, output / notebook.name)
    print(f'Upload: {archive}\nSize: {archive.stat().st_size / 1024 / 1024:.1f} MB\nSHA256: {sha256(archive)}')
    return archive


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='checkpoints/author_embeddings/baby/pretrain.pth')
    parser.add_argument('--dataset', default='data/baby/dataset.pkl')
    parser.add_argument('--output-dir', default='work/author_embeddings_upload')
    args = parser.parse_args()
    package_inputs(args.checkpoint, args.dataset, args.output_dir)
