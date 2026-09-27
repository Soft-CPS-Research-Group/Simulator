"""Render current member histories from executable schemas, without runtime imports."""
from pathlib import Path
import argparse
import json
import re
import shutil
import pandas as pd

LABELS = {
    'core_4': 'Core-4', 'core_15': 'Core-15',
    'core_30_normal': 'Core-30 Normal', 'core_30_topology': 'Core-30 Topology',
    'core_30_safety': 'Core-30 Safety', 'core_30_robustness': 'Core-30 Robustness',
    'core_30_combined': 'Core-30 Combined', 'premium_100_normal': 'Premium Normal',
    'premium_100_allin': 'Premium All-in',
}
INTRO = ('Histórias do inventário sintético de 2023–2024. As transições são geradas '
         'diretamente dos schemas em evaluation/, por configuração; datas e horas '
         'em Europe/Lisbon. A disponibilidade dos equipamentos segue os eventos. '
         'As saídas são definitivas, com todos os ativos do membro; não há atividade '
         'privada posterior nem repetição de entradas em 2024. Os limites que o motor '
         'original consegue impor estão descritos no README da documentação.\n')


def lifecycle(root):
    variants = {}
    for path in sorted((root / 'evaluation').glob('*.json')):
        schema = json.loads(path.read_text())
        start = pd.Timestamp(schema['start_date'], tz='UTC')
        events = []
        omitted = []
        for event in sorted(schema.get('topology_events', []), key=lambda e: e['time_step']):
            if event.get('purpose') == 'native_schema_template_only':
                omitted.append(event['id'])
                continue
            stamp = start + pd.Timedelta(seconds=event['time_step'] * schema['seconds_per_time_step'])
            events.append({**event, 'timestamp_utc': stamp.isoformat(),
                           'timestamp_local': stamp.tz_convert('Europe/Lisbon').isoformat()})
        variants[path.stem] = {
            'schema': str(path.relative_to(root)),
            'initial_membership': {mid: b.get('include', True) for mid, b in schema['buildings'].items()},
            'events': events, 'excluded_template_registration': omitted,
        }
    return {'source': 'evaluation/*.json: start_date + time_step * seconds_per_time_step',
            'timezone': 'Europe/Lisbon', 'years': [2023, 2024], 'variants': variants}


def sentence(event):
    at = event['timestamp_local'].replace('T', ' ')
    op = event['operation']
    if op == 'add_member':
        return f'entra pela primeira vez na REC em {at}'
    if op == 'remove_member':
        return f'sai definitivamente da REC, com todos os seus ativos, em {at}'
    aid = event['target_asset_id']
    if op == 'remove_asset':
        text = f'retira definitivamente {aid} em {at}'
        if event.get('target_asset_type') == 'deferrable_appliance':
            text += ', cancelando os ciclos pendentes ou interrompidos desse equipamento'
        return text
    if op == 'add_asset':
        text = f'instala {aid} em {at}'
        if event.get('purpose') == 'replacement_in':
            text += ', equipamento substituto com identidade própria'
        soc = event.get('overrides', {}).get('initial_soc')
        if soc is not None:
            text += f' e SOC inicial de {100*soc:g}%'
        return text
    raise ValueError(f'Unsupported lifecycle operation: {op}')


def narrative(catalogue, mid):
    groups = {}
    for variant, data in catalogue['variants'].items():
        events = [e for e in data['events'] if e['target_member_id'] == mid]
        initial = data['initial_membership'][mid]
        key = json.dumps([initial, events], sort_keys=True)
        groups.setdefault(key, {'names': [], 'initial': initial, 'events': events})['names'].append(LABELS.get(variant, variant))
    phrases = []
    for group in groups.values():
        states = ['presente no início de 2023' if group['initial'] else 'ainda não integra a REC no início de 2023']
        states.extend(sentence(e) for e in group['events'])
        if group['events']:
            states.append('em 2024 mantém o estado resultante, sem novas transições de topologia')
        else:
            states.append('sem transições de membros ou ativos em 2023–2024')
        phrases.append(' / '.join(group['names']) + ': ' + '; '.join(states) + '.')
    return 'Histórico por configuração: ' + ' '.join(phrases)


def base_story(text):
    # These are dedicated trailing lifecycle sections, never family narrative.
    text = re.split(r' (?:Nas variantes dinâmicas|No cenário dinâmico corrente:|Histórico por configuração:)', text)[0]
    return text.replace(' Personagens e instalações são fictícias.', '').strip()


def render(members, title='Atlas de membros fictícios'):
    return '# ' + title + '\n\n' + INTRO + ''.join(
        f'\n## {m.member_id}\n\n{m.story}\n' for m in members.itertuples())


def synchronize(root):
    catalogue = lifecycle(root)
    members = pd.read_parquet(root / 'catalogs/members.parquet')
    # Preserve the previously corrected residential narrative (e.g. school children)
    # from the native atlas when available, then replace only its lifecycle section.
    previous = root / 'evaluation/MEMBER_STORIES.md'
    stories = dict(re.findall(r'## (Member_\d+)\n\n(.*?)(?=\n## |\Z)', previous.read_text(), re.S)) if previous.exists() else {}
    for i, member in members.iterrows():
        history = narrative(catalogue, member.member_id)
        original = stories.get(member.member_id, member.story).strip()
        members.at[i, 'story'] = base_story(original) + ' ' + history + ' Personagens e instalações são fictícias.'
        # Uniform field across all four catalogues, explicitly scoped by variant.
        members.at[i, 'dynamic_membership_story'] = history
    members.to_parquet(root / 'catalogs/members.parquet', index=False, compression='zstd')
    text = render(members)
    (root / 'MEMBER_STORIES.md').write_text(text)
    if previous.exists():
        previous.write_text(text)
    plan = root / 'catalogs/lifecycle_plan.json'
    if plan.exists():
        old = json.loads(plan.read_text())
        if old.get('source') != catalogue['source']:
            archive = root / 'sources/2023_original_lifecycle_plan.json'
            archive.parent.mkdir(exist_ok=True, parents=True)
            if not archive.exists():
                shutil.copy2(plan, archive)
    plan.write_text(json.dumps(catalogue, indent=2, ensure_ascii=False) + '\n')


def atlas(repo):
    root = repo / 'data/datasets/rec_pt_2023_premium_100_v2'
    members = pd.read_parquet(root / 'catalogs/members.parquet')
    target = repo / 'docs/portugal_rec_native/Atlas_Membros_Premium.md'
    target.parent.mkdir(exist_ok=True, parents=True)
    target.write_text(render(members, 'Atlas das famílias e instalações · Premium · motor original'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    # Dataset synchronization is part of extend_portugal_2024, before checksums.
    atlas(args.repo)
