"""Static lifecycle consistency, forecast boundary and variability evidence.

No controller runs and no CityLearn imports. Optional previous tree verifies
that only narratives, metadata and final price forecast cells changed.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from portugal_lifecycle_stories import lifecycle, narrative, render, atlas

TOTAL = 70176
UTC = pd.date_range('2023-01-01', periods=TOTAL, freq='15min', tz='UTC')
HORIZONS = (4, 24, 96)
LABELS = {'outdoor_dry_bulb_temperature': ('Temperatura', '°C', 1),
          'outdoor_relative_humidity': ('Humidade', 'p.p.', 1),
          'direct_solar_irradiance': ('Irradiância direta — diurna', 'W/m²', 1),
          'diffuse_solar_irradiance': ('Irradiância difusa — diurna', 'W/m²', 1),
          'electricity_pricing': ('Preço', 'cêntimos/kWh', 100)}


def metrics(actual, predicted):
    error = predicted - actual
    return dict(n=len(error), mae=float(np.abs(error).mean()),
                rmse=float(np.sqrt(np.mean(error**2))), bias=float(error.mean()),
                p95_absolute_error=float(np.quantile(np.abs(error), .95)),
                predicted_std=float(np.std(predicted)), actual_std=float(np.std(actual)),
                nmae_mean_pct=float(np.abs(error).mean()/max(np.mean(actual), 1e-10)*100))


def main(repo, reference, previous=None):
    checks = []
    def check(name, ok):
        checks.append({'name': name, 'passed': bool(ok)})
    roots = sorted((repo/'data/datasets').glob('rec_pt_2023_*_v2'))
    atlas(repo)
    for root in roots:
        cat = lifecycle(root)
        stored = json.loads((root/'catalogs/lifecycle_plan.json').read_text())
        check(root.name+': catalogue derived from all executable variants', stored == cat)
        members = pd.read_parquet(root/'catalogs/members.parquet')
        for m in members.itertuples():
            text = narrative(cat, m.member_id)
            check(root.name+': dated story '+m.member_id,
                  m.dynamic_membership_story == text and m.story.endswith(text+' Personagens e instalações são fictícias.'))
        expected = render(members)
        check(root.name+': root stories synchronized', (root/'MEMBER_STORIES.md').read_text() == expected)
        if (root/'evaluation/MEMBER_STORIES.md').exists():
            check(root.name+': evaluation stories synchronized', (root/'evaluation/MEMBER_STORIES.md').read_text() == expected)
        if 'core_30' in root.name:
            check('Core-30 requested exit correction', '2023-11-14 22:00:00+00:00' in members.iloc[0].story and '2023-10-18' not in members.iloc[0].story)
        if 'premium' in root.name:
            check('Premium requested exit correction', '2023-08-14 13:00:00+01:00' in members.iloc[0].story and '2023-07-20' not in members.iloc[0].story)
            check('Atlas is the same narrative as catalogue',
                  (repo/'docs/portugal_rec_native/Atlas_Membros_Premium.md').read_text() == render(members, 'Atlas das famílias e instalações · Premium · motor original'))
        if previous:
            oldroot = previous/'data/datasets'/root.name
            oldprice = pd.read_parquet(oldroot/'shared/pricing_btn_bi.parquet')
            price = pd.read_parquet(root/'shared/pricing_btn_bi.parquet')
            check(root.name+': all actual prices unchanged', price.electricity_pricing.equals(oldprice.electricity_pricing))
            for k, h in enumerate(HORIZONS, 1):
                col = f'electricity_pricing_predicted_{k}'
                check(root.name+': earlier forecasts unchanged '+str(h), price[col][:-h].equals(oldprice[col][:-h]))
            # All other time series / event / physical catalogues must be identical.
            allowed = {'catalogs/members.parquet', 'shared/pricing_btn_bi.parquet'}
            unchanged = True
            for path in root.rglob('*.parquet'):
                rel = str(path.relative_to(root))
                if rel not in allowed:
                    unchanged &= path.read_bytes() == (oldroot/rel).read_bytes()
            check(root.name+': all other data files byte-identical', unchanged)
            for path in (root/'evaluation').glob('*.json'):
                new = json.loads(path.read_text()); old = json.loads((oldroot/'evaluation'/path.name).read_text())
                for s in (new,old):
                    s.pop('dataset_contract_version',None);s['experimental_design'].pop('version',None)
                check(root.name+': executable schema unchanged '+path.stem, new == old)

    root = next(p for p in roots if 'micro_4' in p.name)
    weather = pd.read_parquet(root/'shared/weather.parquet')
    check('2023 humidity is the documented synthetic proxy', np.allclose(weather.outdoor_relative_humidity[:35040],np.clip(78-1.7*(weather.outdoor_dry_bulb_temperature[:35040]-14),30,95),atol=1e-5))
    price = pd.read_parquet(root/'shared/pricing_btn_bi.parquet')
    support = pd.read_parquet(reference/'forecast_support/price_components.parquet').set_index('timestamp_utc')
    check('support is exactly one external day', support.index.equals(pd.date_range('2025-01-01',periods=96,freq='15min',tz='UTC')))
    check('simulation period still two years', len(price)==TOTAL and len(weather)==TOTAL)
    c = pd.concat([pd.read_parquet(reference/'price_components_btn_bi.parquet').set_index('timestamp_utc'),support])
    evidence = pd.read_parquet(reference/'forecast_support/boundary_forecasts.parquet')
    check('124 final forecast targets recorded', len(evidence)==sum(HORIZONS))
    for k,h in enumerate(HORIZONS,1):
        rows = evidence[evidence.horizon_steps==h]
        issue = pd.DatetimeIndex(rows.issue_utc); target = issue+pd.Timedelta(minutes=15*h)
        madrid = target.tz_convert('Europe/Madrid')
        publication = (madrid.tz_localize(None).normalize()-pd.Timedelta(days=1)+pd.Timedelta(hours=14)).tz_localize('Europe/Madrid').tz_convert('UTC')
        source = pd.DatetimeIndex(np.where(issue>=publication, target, target-pd.Timedelta(days=1)))
        expected = ((c.loc[source,'omie_eur_kwh'].to_numpy()+.014+.0016574)*1.16+c.loc[target,'access_energy_eur_kwh'].to_numpy()+.001).astype('float32')
        check('unclipped targets '+str(h), pd.DatetimeIndex(rows.target_utc).equals(target) and target[-1]==UTC[-1]+pd.Timedelta(minutes=15*h))
        check('publication-aware boundary prices '+str(h), np.array_equal(price[f'electricity_pricing_predicted_{k}'].to_numpy()[-h:],expected))
        check('no unpublished OMIE leakage '+str(h), np.all((source<=issue)|(publication<=issue)))
        check('boundary audit source dates '+str(h), pd.DatetimeIndex(rows.omie_source_utc).equals(source))
        for col in ('outdoor_dry_bulb_temperature','outdoor_relative_humidity'):
            check('weather has full final horizon '+col+' '+str(h), np.array_equal(weather[f'{col}_predicted_{k}'].to_numpy()[-h:],weather[col].to_numpy()[np.arange(TOTAL-h,TOTAL)+h-96]))

    rows = []
    for key,(label,unit,scale) in LABELS.items():
        frame = price if key=='electricity_pricing' else weather
        for k,h in enumerate(HORIZONS,1):
            issue = np.arange(96,TOTAL-96)  # same issue sample across horizons; warm-up excluded
            target = issue+h
            actual = frame[key].to_numpy()[target]*scale
            pred = frame[f'{key}_predicted_{k}'].to_numpy()[issue]*scale
            daylight = np.ones(len(issue),dtype=bool)
            if 'solar' in key:
                daylight = (weather.direct_solar_irradiance.to_numpy()[target]+weather.diffuse_solar_irradiance.to_numpy()[target])>1
            for year in (2023,2024):
                select = (UTC[issue].year==year)&daylight
                item = dict(signal=key,label=label,unit=unit,year=year,horizon_hours=h/4,**metrics(actual[select],pred[select]))
                if key=='electricity_pricing':
                    madrid=UTC[target].tz_convert('Europe/Madrid')
                    pub=(madrid.tz_localize(None).normalize()-pd.Timedelta(days=1)+pd.Timedelta(hours=14)).tz_localize('Europe/Madrid').tz_convert('UTC')
                    available=UTC[issue]>=pub
                    item['omie_published_pct']=float(100*available[select].mean())
                    item['fallback_mae']=float(np.abs(actual-pred)[select&~available].mean()) if (select&~available).any() else None
                rows.append(item)
            if key in ('direct_solar_irradiance','electricity_pricing'):
                for month in range(1,13):
                    select=(UTC[issue].year==2024)&(UTC[issue].month==month)&daylight
                    rows.append(dict(signal=key,label=label,unit=unit,year=2024,month=month,horizon_hours=h/4,**metrics(actual[select],pred[select])))
    # Exogenous community aggregates only: no EV, BESS or controlled-machine policy.
    aggregates=[]
    for root in roots:
        frames=[pd.read_parquet(p,columns=['non_shiftable_load','solar_generation']) for p in sorted((root/'timeseries').glob('Member_*.parquet'))]
        total=sum(f for f in frames)*4  # interval energy -> kW
        for key in total:
            a=total[key].to_numpy()
            for year in (2023,2024):
                target=np.flatnonzero((UTC.year==year)&(np.arange(TOTAL)>=192)&(np.arange(TOTAL)<TOTAL-96))
                if key=='solar_generation':target=target[a[target]>0]
                aggregates.append(dict(family=root.name,signal=key,year=year,unit='kW',**metrics(a[target],a[target-96])))
    perturbations=[]
    manifest=json.loads((repo/'results/portugal_rec_native/manifest.json').read_text())
    for variant in manifest['variants']:
        schema=repo/variant['schema'];s=json.loads(schema.read_text())
        for phase,path in [('development',s.get('robustness',{}).get('events_file')),('confirmation',s['experimental_design'].get('native_confirmation_robustness_file'))]:
            if not path:continue
            events=pd.read_parquet(schema.parent.parent/path);f=events[events.module=='forecast']
            slots=set(t for e in f.itertuples() for t in range(e.start_time_step,e.end_time_step+1))
            perturbations.append(dict(variant=variant['variant'],phase=phase,rows=len(f),unique_campaign_hours=len(slots)/4,
                elapsed_time_pct=len(slots)/TOTAL*100,features=sorted(f.target_feature.unique()),
                values=sorted(f.value.dropna().unique()),modes=sorted(f['mode'].unique())))
    failed=[c['name'] for c in checks if not c['passed']]
    report=dict(version=manifest['version'],scope='Static forecast input errors; no controller or runtime forecast evaluation; errors exclude health overlays.',
        checks=checks,failed=failed,forecast_metrics=rows,exogenous_persistence=aggregates,forecast_fault_campaigns=perturbations,
        previous_revision_checked=previous is not None,
        native_boundary_limitation='Original engine clips target before persistence for derived load/PV at episode end; file padding is removed by the loader. No engine changes made.',
        uncertainty_model='Deterministic daily persistence, not calibrated probabilistic forecasts or archived numerical weather predictions. Shared weather and price across families; no independent member forecast-skill distribution.')
    out=repo/'results/portugal_rec_native/forecasts_and_stories.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    document(repo,report,price,reference)
    print(json.dumps(dict(checks=len(checks),failed=failed,metrics=[r for r in rows if r['year']==2024 and 'month' not in r]),ensure_ascii=False,indent=2))
    return not failed


def document(repo,report,price,reference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    docs=repo/'docs/portugal_rec_native';figures=docs/'figures';figures.mkdir(exist_ok=True)
    rows=report['forecast_metrics'];annual=[r for r in rows if 'month' not in r]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax,key,title in zip(axes,['electricity_pricing','direct_solar_irradiance'],['Preço · erro absoluto médio (cênt./kWh)','Irradiância direta diurna · erro médio (W/m²)']):
        for h,color in [(1,'#246b8e'),(6,'#c27727'),(24,'#793d78')]:
            q=[r for r in rows if r['signal']==key and r.get('month') and r['horizon_hours']==h]
            ax.plot([r['month'] for r in q],[r['mae'] for r in q],marker='o',ms=3,label=f'{h} h',color=color)
        ax.set(title=title,xlabel='Mês de emissão · 2024',xticks=range(1,13));ax.grid(alpha=.15);ax.legend()
    fig.savefig(figures/'forecast_errors_2024.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3.4),layout='constrained')
    support=pd.read_parquet(reference/'forecast_support/price_components.parquet').set_index('timestamp_utc')
    truth=pd.Series(price.electricity_pricing.to_numpy(),index=UTC)
    truth=pd.concat([truth,support.electricity_pricing])
    q=truth.loc['2024-12-31':'2025-01-01']*100
    ax.plot(q.index,q.values,label='Preço realizado / apoio',color='#172f42',lw=1.5)
    for hour,color in [(12,'#c27727'),(13,'#246b8e')]:
        issue=pd.Timestamp(f'2024-12-31 {hour}:00',tz='UTC');i=UTC.get_loc(issue)
        ax.scatter([issue+pd.Timedelta(hours=24)],[100*price.iloc[i].electricity_pricing_predicted_3],s=45,color=color,label=f'Previsão +24 h emitida às {hour}h')
    ax.axvline(pd.Timestamp('2025-01-01',tz='UTC'),color='#a34840',ls='--',lw=1,label='Fim do período simulado')
    ax.set(ylabel='cêntimos/kWh',title='A previsão mantém o horizonte ao atravessar o fim de 2024');ax.legend(loc='upper right',fontsize=8);ax.grid(alpha=.15)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %Hh'));fig.savefig(figures/'forecast_boundary_2024.png',dpi=170);plt.close(fig)
    table=['| Sinal | Ano | MAE a 1 h | MAE a 6 h | MAE a 24 h | Unidade |','|---|---:|---:|---:|---:|---|']
    for key,(label,unit,_) in LABELS.items():
        for year in (2023,2024):
            q=[r for r in annual if r['signal']==key and r['year']==year]
            table.append(f'| {label} | {year} | '+ ' | '.join(f"{r['mae']:.3f}" for r in q)+f' | {unit} |')
    agg=['| Base | Carga 2023 | Carga 2024 | PV diurno 2023 | PV diurno 2024 |','|---|---:|---:|---:|---:|']
    for family in sorted({r['family'] for r in report['exogenous_persistence']}):
        q=[next(r for r in report['exogenous_persistence'] if r['family']==family and r['signal']==s and r['year']==y) for s in ('non_shiftable_load','solar_generation') for y in (2023,2024)]
        name=family.replace('rec_pt_2023_','').replace('_v2','')
        agg.append(f'| {name} | '+' | '.join(f"{r['nmae_mean_pct']:.1f}%" for r in q)+' |')
    text='''# Histórias, fronteira de 2024 e variabilidade das previsões

## Correções concluídas nos dados

O atlas, os ficheiros MEMBER_STORIES e os campos narrativos dos quatro catálogos passaram a usar os eventos dos schemas. O catálogo lifecycle_plan.json é agora derivado por configuração, com estado inicial e datas calculadas a partir do índice temporal. O desenho antigo fica apenas em sources/2023_original_lifecycle_plan.json. O registo interno de um carregador-modelo não é apresentado como remoção física. As transições de 2023 mantêm o seu efeito em 2024.

No Core-30 Topology/Combined, Member_001 sai em **14/11/2023 às 22h de Lisboa**. No Premium All-in sai em **14/08/2023 às 13h de Lisboa**. As variantes estáticas não recebem essas saídas. As datas incluem igualmente instalação, substituição e retirada de ativos. Não se alteraram eventos, inventário, perfis físicos ou pedidos EV.

Acrescentaram-se **96 amostras de apoio de 01/01/2025**, exclusivamente para os destinos das previsões de preço. As séries simuladas continuam com **70 176 registos**, terminando em 31/12/2024. Foram corrigidas as últimas 4, 24 e 96 posições das previsões de 1, 6 e 24 horas: 124 células por base. As horas de destino avançam para janeiro, em vez de ficarem presas a 31 de dezembro às 23h45. A meteorologia já tinha horizontes completos por persistência e não foi alterada.

O OMIE só é usado após a hora conservadora de publicação: 14h de Madrid, 13h de Lisboa. Antes disso usa-se o mesmo instante do dia anterior. Os destinos de janeiro usam TAR de 0,0149/0,0830 €/kWh e TSE de 0,0016574 €/kWh, conhecidos em dezembro. Mantêm-se as demais componentes do contrato de referência. Isto não reconstrói uma nova oferta comercial em 2025.

Fontes arquivadas em data/portugal_reference_2024/forecast_support: [ERSE — tarifas 2025, p. 131](https://www.erse.pt/media/xczdo4as/tep-se-2025.pdf), [anúncio de 16/12/2024](https://www.erse.pt/media/jbyluyci/comunicado_tarifas-ee-2025.pdf), [TSE, anúncio de 20/12/2024](https://www.erse.pt/comunicacao/destaques/erse-aprova-diretivas-de-reparticao-do-financiamento-dos-custos-com-a-tarifa-social-de-eletricidade-de-2025/). O ficheiro OMIE de 2 de janeiro só é necessário porque a primeira hora de Madrid corresponde às 23h UTC do dia 1.

![Fronteira temporal](figures/forecast_boundary_2024.png)

## O que significa «variabilidade» nesta edição

Há variação real das séries e erros de previsão resultantes de diferenças entre dias. Não existem previsões meteorológicas operacionais arquivadas, distribuições probabilísticas, intervalos de confiança ou níveis de qualidade sorteados por membro. Consumo e PV derivados no motor e meteorologia nos ficheiros seguem persistência diária; os preços combinam informação de mercado publicada com persistência quando ainda não foi publicada. Não se acrescentou ruído artificial às previsões nesta revisão.

Na persistência diária, a estimativa do mesmo destino é a observação desse instante 24 horas antes, independentemente de a emissão ocorrer a 1, 6 ou 24 horas. Por isso os erros médios por horizonte são quase iguais. As pequenas diferenças decorrem das amostras de destino e da máscara diurna. **A dificuldade de previsão não cresce de forma realista com o horizonte.** Trata-se de uma referência causal simples, não de um modelo meteorológico com atualização contínua.

O preço e a meteorologia são comuns a todas as famílias, coerentemente com a região e tarifa comuns. A heterogeneidade de carga e PV vem dos perfis e equipamentos; não existe heterogeneidade de qualidade de previsão por membro. A resolução de 15 minutos da meteorologia resulta de interpolação horária, pelo que não representa toda a variabilidade sub-horária de nuvens ou sensores.

Há uma diferença de origem na humidade: em 2023 é a aproximação sintética clip(78 − 1,7 × (temperatura − 14), 30, 95); em 2024 é ERA5. A subida do seu erro entre anos não pode ser interpretada apenas como mudança meteorológica. Preservou-se a edição física de 2023, como acordado. Para estudar especificamente previsão de humidade, essa diferença metodológica teria de ser tratada; não foi escondida nem corrigida com ruído arbitrário.

## Erros medidos nos ficheiros

MAE é o erro absoluto médio contra a amostra futura efetiva. Emissões após o primeiro dia e antes do último dia; mesma amostra de emissão para todos os horizontes; ano da emissão. Nos canais solares só se incluem destinos com irradiância total observada superior a 1 W/m². Não se contam as noites como acertos fáceis. Não são resultados de um controlador nem uma avaliação dos overlays de health.

'''+ '\n'.join(table)+'''

![Variação mensal dos erros](figures/forecast_errors_2024.png)

Os preços a 1 e 6 horas beneficiam de o OMIE já estar publicado. A 24 horas há períodos sem informação publicada, com maior erro. Em 2023 o ajuste MIBEL definitivo é estimado causalmente; em abril de 2024 a previsão não conhece ainda a TSE retroativa. Esses efeitos explicam erros mesmo quando o OMIE é conhecido. O ficheiro JSON inclui erro quadrático, viés, percentil 95, desvio-padrão, fração publicada e erro do período sem publicação.

## Consumo e PV: referência física sem executar controladores

Erro absoluto da persistência diária, dividido pela média da série, nos agregados das quatro bases estáticas. O PV é contado apenas quando existe produção. A carga inclui o consumo não deslocável dos ficheiros; exclui carregamento EV, baterias e decisões sobre máquinas controladas. Não é uma medição das observações derivadas do motor, que também podem conter histórico de máquinas e efeitos da topologia. Todos os horizontes desta regra recorrem ao mesmo valor anterior para um dado destino.

'''+ '\n'.join(agg)+'''

## Falhas e ruído continuam separados

Core-30 Robustness, Core-30 Combined e Premium All-in incluem um viés de preço de +0,025 €/kWh durante uma campanha de seis horas em cada ano, replicado nos canais relevantes. Por configuração, as 22 linhas de desenvolvimento representam canais e anos, não 22 acontecimentos independentes: são **12 horas distintas em 731 dias**, cerca de 0,068% do período. A confirmação reservada usa −0,035 €/kWh, oito horas em cada ano (16 horas no total). Isto testa falhas pontuais; não representa erro de previsão persistente durante todo o ano.

O ruído de medição continua separado, nas variantes configuradas: erros gaussianos nos canais escolhidos de potência e SOC, em campanhas sazonais. Não se transforma automaticamente num modelo de incerteza de previsão. Os ficheiros de verdade física foram preservados. Esta revisão inspeciona a configuração dos eventos, não certifica a sua execução no motor.

## Limite que não pode ser corrigido só nos datasets

O motor original limita internamente o índice de destino das previsões **derivadas de consumo e PV** ao último registo carregado, antes de procurar o dia anterior. O carregador de dados corta os ficheiros à janela de simulação; acrescentar linhas físicas de apoio não resolve esse caminho. Consequentemente, dentro das últimas 24 horas de um episódio continuam a existir canais derivados com horizonte encurtado. O mesmo problema pode ocorrer no fim de episódios mais curtos. Os preços armazenados, incluindo os usados pelo caminho nativo publication_aware, e a meteorologia têm os seus destinos completos.

Não foi alterado o simulador, nem desligados canais para esconder o problema. **O fecho desta revisão aplica-se às histórias e às previsões fornecidas pelos dados; não certifica a fronteira das previsões de consumo/PV calculadas pelo motor.** Resolver esse comportamento em execução exigiria uma decisão posterior sobre o motor ou sobre a interface do controlador. Não foi feita tal alteração.

Para uma futura investigação centrada na qualidade das previsões, falta um modelo validado com atualização por horizonte, correlações temporais e entre membros, além de qualidade diferenciada por sinal. Para os dados atuais fica documentada e quantificada a referência simples realmente existente.
'''
    text += f"\nVerificação desta revisão: {len(report['checks'])} verificações, {sum(not x['passed'] for x in report['checks'])} falhas. Evidência: results/portugal_rec_native/forecasts_and_stories.json.\n"
    (docs/'Previsoes_e_historias.md').write_text(text)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=Path(__file__).resolve().parents[2]/'data/portugal_reference_2024')
    p.add_argument('--previous',type=Path)
    a=p.parse_args();raise SystemExit(0 if main(a.repo,a.reference,a.previous) else 1)
