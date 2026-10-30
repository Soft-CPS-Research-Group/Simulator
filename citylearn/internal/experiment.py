"""Portable evidence accompanying KPI exports; no credentials or remote calls."""
import hashlib
import json
import subprocess
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def experiment_manifest(env):
    from citylearn import __version__
    from citylearn.agents.baseline import BusinessAsUsualAgent
    schema = env.schema
    root = Path(schema.get('root_directory') or '.').resolve()
    files = {}
    missing = []
    def collect(value):
        if isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
        elif isinstance(value, str) and Path(value).suffix.lower() in {'.csv', '.parquet', '.json'}:
            path = Path(value)
            path = path if path.is_absolute() else root/path
            if path.is_file():
                files[value] = _sha256(path)
            else:
                missing.append(value)
    collect(schema)
    repo = Path(__file__).resolve().parents[2]
    revision, dirty = None, None
    try:
        if not (repo/'.git').exists():
            raise OSError('Installed package has no source checkout')
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, stderr=subprocess.DEVNULL,
                                            text=True, timeout=5).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                            cwd=repo, stderr=subprocess.DEVNULL, text=True, timeout=5).strip())
    except (OSError, subprocess.SubprocessError):
        pass
    payload = json.dumps(schema, sort_keys=True, default=str).encode()
    metadata = env.get_metadata()
    tracker = env.episode_tracker
    members, district = env._provenance_service.summary()
    return dict(simulator_version=__version__, git_commit=revision, tracked_worktree_dirty=dirty,
                business_as_usual_definition=BusinessAsUsualAgent.DEFINITION_VERSION,
                schema_sha256=hashlib.sha256(payload).hexdigest(), effective_schema=schema,
                data_files_sha256=files, unresolved_file_references=sorted(set(missing)),
                random_seed=env.random_seed, seconds_per_time_step=env.seconds_per_time_step,
                episode_start_time_step=tracker.episode_start_time_step,
                episode_end_time_step=tracker.episode_end_time_step,
                completed_intervals=env._provenance_service.steps,
                community_market=metadata['community_market'], energy_accounting=metadata['energy_accounting'],
                deferrable_service_audit=metadata['deferrable_service_audit'],
                provenance_summary=dict(buildings=members, district=district),
                source_destinations=env._provenance_service.source_destinations())
