"""Exercise the installed wheel, including the final KPI provenance manifest.

Run with ``python -I`` so the source tree cannot hide missing packaged modules.
"""

import argparse
import json
from pathlib import Path

import citylearn
from citylearn.citylearn import CityLearnEnv
from citylearn.internal.ev_departure_audit import DEFINITION_VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_root = Path(__file__).resolve().parents[2]
    installed_module = Path(citylearn.__file__).resolve()
    assert source_root not in installed_module.parents, installed_module

    env = CityLearnEnv(
        str(args.schema.resolve()),
        episode_time_steps=4,
        central_agent=False,
        render_directory=args.output.resolve(),
        export_kpis_on_episode_end=False,
        offline=True,
        random_seed=0,
    )
    try:
        env.reset()
        while not (env.terminated or env.truncated):
            env.step([[0.0] * len(names) for names in env.action_names])
        env.export_final_kpis(filepath="installed_kpis.csv")
        output = Path(env.new_folder_path)
        assert (output / "installed_kpis.csv").is_file()
        manifest = json.loads((output / "installed_kpis.manifest.json").read_text())
        assert isinstance(manifest, dict) and manifest
        assert DEFINITION_VERSION in json.dumps(manifest)
        assert 'district_ev_departure_audit_target_coverage_ratio' in (output / 'installed_kpis.csv').read_text()
        service = json.loads((output / 'installed_kpis.deferrable_service.json').read_text())
        assert service['definition_version'] == 'deferrable-service-audit-v1'
        assert manifest['deferrable_service_audit']['definition_version'] == service['definition_version']
        assert 'district_deferrable_appliance_audit_offered_count' in (output / 'installed_kpis.csv').read_text()
        assert list(output.glob("exported_data_business_as_usual_ep*.csv"))
        print(f"Installed {citylearn.__version__}: KPI, BAU and manifest export passed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
