"""Create a deterministic portable source+weights archive without source DICOM."""
from __future__ import annotations
import argparse
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dxaqc.bundle import sha256, verify_bundle

DOCS = ['PROJECT_DOCUMENTATION.md', 'RELEASE.md', 'DEPLOYMENT.md', 'TRAINING.md',
        'DEMO.md', 'FINAL_VALIDATION.md', 'STUDY_LEVEL.md', 'ROBUSTNESS.md',
        'METHODS_AND_MODELS.md', 'EXPERIMENT_RESULTS.md', 'RESEARCH_SOURCES.md',
        'THIRD_PARTY.md', 'EXPERIMENTS.md', 'ACCEPTANCE.json']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'models/release')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/dist/dxaqc-release.tar')
    args = parser.parse_args()
    manifest = verify_bundle(args.model_dir)
    sources = {name: ROOT / name for name in ['README.md','pyproject.toml','uv.lock','Dockerfile','.dockerignore','requirements-runtime.txt']}
    for directory in ('src', 'scripts', 'tests'):
        for path in sorted((ROOT / directory).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.sh') and '__pycache__' not in path.parts:
                sources[path.relative_to(ROOT).as_posix()] = path
    for name in DOCS:
        path = ROOT / 'docs' / name
        if path.exists():
            sources['docs/' + name] = path
    # The TЗ contains task illustrations, not local training data.
    sources['docs/ТЗ-ДепЗдрав.pdf'] = ROOT / 'docs/ТЗ-ДепЗдрав.pdf'
    for name in ['manifest.json', *manifest['files']]:
        sources['models/release/' + name] = args.model_dir / name
    metadata = {'model_id': manifest['model_id'], 'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'files': {name: sha256(path) for name,path in sources.items()},
                'includes_training_data': False, 'model_manifest_sha256': sha256(args.model_dir/'manifest.json')}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(args.output, 'w') as archive:
        for name,path in sorted(sources.items()):
            info=archive.gettarinfo(str(path),arcname='dxaqc/'+name)
            info.uid=info.gid=info.mtime=0
            info.uname=info.gname=''
            with path.open('rb') as stream: archive.addfile(info,stream)
        data=(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n').encode()
        info=tarfile.TarInfo('dxaqc/BUILD_INFO.json');info.size=len(data);info.mode=0o644
        archive.addfile(info,io.BytesIO(data))
    digest=sha256(args.output)
    args.output.with_suffix(args.output.suffix+'.sha256').write_text(digest+'  '+args.output.name+'\n')
    print(json.dumps({'archive':str(args.output),'bytes':args.output.stat().st_size,'sha256':digest,'files':len(sources)+1}))

if __name__=='__main__': main()
