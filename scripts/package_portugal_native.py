"""Bundle the current Portugal data catalogue, sources and construction evidence."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    docs=ROOT/'docs/portugal_rec_native';results=ROOT/'results/portugal_rec_native'
    manifest=json.loads((results/'manifest.json').read_text())
    report=json.loads((results/'two_year_integrity.json').read_text())
    assert not report['failed'] and manifest['years']==[2023,2024]
    roots=[docs,results,ROOT/'data/portugal_reference_2023',ROOT/'data/portugal_reference_2024',*sorted((ROOT/'data/datasets').glob('rec_pt_2023_*_v2'))]
    files=[p for root in roots for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ('package_checksums.sha256','package_manifest.json')]
    files += [ROOT/name for name in ['data/portugal_native_source_v23.tar.gz','scripts/build_portugal_native_suite.py','scripts/portugal_rec_common.py','scripts/portugal_site_calendar.py','scripts/prepare_portugal_reference.py','scripts/prepare_portugal_reference_2024.py','scripts/portugal_profiles_2024.py','scripts/extend_portugal_2024.py','scripts/package_portugal_native.py','scripts/requirements-portugal-generator.txt','scripts/audit/verify_portugal_two_year_inputs.py','scripts/audit/verify_portugal_core30_sessions.py']]
    files += [ROOT/'scripts/portugal_lifecycle_stories.py', ROOT/'scripts/audit/verify_portugal_forecasts_stories.py']
    files=sorted(set(files));checks=docs/'package_checksums.sha256';checks.write_text(''.join(f'{digest(p)}  {p.relative_to(ROOT)}\n' for p in files))
    archive=ROOT/'docs/portugal_rec_native.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in files+[checks]:z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==len(files)+1
        assert not any(x.startswith('citylearn/') or '/annual/' in x for x in z.namelist())
    info={'version':manifest['version'],'years':[2023,2024],'records':70176,'configurations':9,'families':4,'package':str(archive.relative_to(ROOT)),'sha256':digest(archive),'bytes':archive.stat().st_size,'files':len(files)+1,'simulator_core':manifest['simulator_commit'],'frozen_2023_commit':manifest['frozen_2023_commit'],'scope':'Data, public sources, atlas, offline reconstruction and static input checks; no annual/controller evaluation.'}
    (docs/'package_manifest.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
if __name__=='__main__':main()
