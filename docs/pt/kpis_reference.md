# KPIs Reference

O simulador tem duas APIs de KPIs:

| API | Estado | Formato | Uso recomendado |
|---|---|---|---|
| `evaluate()` | legacy | DataFrame com nomes historicos | Compatibilidade com workflows antigos. |
| `evaluate_v2()` | principal | DataFrame com nomes estruturados | Produção, dashboards, comparacao cientifica. |

Para a arvore completa de nomes v2, ver tambem [`KPI_V2_TREE.md`](../KPI_V2_TREE.md).

## Contrato de Unidades

| Unidade no nome | Significado |
|---|---|
| `eur` | Custo monetario. Pode representar outra currency se o dataset usar outra moeda. |
| `kwh` | Energia acumulada. |
| `kgco2` | Emissoes acumuladas. |
| `kw` | Potencia de pico/media. |
| `ratio` | Razao adimensional. |
| `percent` | Percentagem. |
| `count` | Contagem de eventos/ciclos. |
| `hours` | Horas. |
| `c` | Temperatura em Celsius. |

## Convencao KPI v2

```text
level_family_subfamily_metric_variant_unit
```

| Parte | Exemplos |
|---|---|
| `level` | `building`, `district`. |
| `family` | `cost`, `energy_grid`, `emissions`, `solar_self_consumption`, `ev`, `battery`, `electrical_service_phase`, `equity`, `comfort_resilience`, `deferrable_appliance`. |
| `subfamily` | `total`, `daily_average`, `ratio_to_baseline`, `shape_quality`, `service`. |
| `metric` | `import`, `export`, `charge`, `completed_cycles`. |
| `variant` | `control`, `baseline`, `delta`, `total`, `average`, `l1`. |
| `unit` | `eur`, `kwh`, `ratio`, etc. |

Exemplos:

| KPI | Significado |
|---|---|
| `district_cost_total_control_eur` | Custo total do controlo. |
| `building_energy_grid_total_import_control_kwh` | Import total do building. |
| `district_energy_grid_shape_quality_ramping_average_to_baseline_ratio` | Ramping medio relativo ao baseline. |
| `district_solar_self_consumption_ratio_self_consumption_ratio` | Self-consumption solar do distrito. |
| `building_deferrable_appliance_service_completed_cycles_count` | Ciclos deferiveis completados. |

## Equacoes Base

Para uma serie net `net_t`, onde import e positivo e export e negativo:

```text
total_import_kwh = sum(max(net_t, 0))
total_export_kwh = sum(max(-net_t, 0))
total_net_exchange_kwh = sum(net_t)
daily_average_x = total_x / simulated_days
delta_x = control_x - baseline_x
ratio_to_baseline_x = control_x / baseline_x
self_consumption = (solar_generation_total - export_total) / solar_generation_total
```

Safe division devolve `None`/placeholder quando o denominador nao e fisicamente valido.

## Familias Principais

| Familia | Nivel | O que mede |
|---|---|---|
| `cost` | building, district | Custo control/baseline/delta/ratio. |
| `energy_grid` | building, district | Import, export, net exchange e shape quality. |
| `emissions` | building, district | Emissoes control/baseline/delta/ratio. |
| `solar_self_consumption` | building, district | Geracao, export e self-consumption. |
| `ev` | building, district | Departures, success rate, deficits, charge e V2G. |
| `battery` | building, district | Charge, discharge, throughput, cycles, capacity fade. |
| `electrical_service_phase` | building, district | Violacoes, imbalance e peaks por fase. |
| `equity` | building, district | Beneficios relativos e distribuicao. |
| `comfort_resilience` | building, district | Discomfort e eventos de outage/resilience. |
| `deferrable_appliance` | building, district | Ciclos completed/missed, service level e energia servida. |

## KPIs EV

| KPI conceptual | Unidade | O que mede |
|---|---:|---|
| departure count | count | Numero de departures observadas. |
| departure met count | count | Departures em que SOC requerido foi cumprido. |
| departure minimum acceptable count | count | Departures em que o SOC e pelo menos `target_soc - ev_departure_service_tolerance`. |
| departure within tolerance count | count | Departures dentro da tolerancia simetrica configurada. |
| departure success ratio | ratio | `met / departures`. |
| departure minimum acceptable ratio | ratio | `minimum_acceptable / departures`. Este e o KPI principal de conforto/servico EV. |
| departure within tolerance ratio | ratio | `within_symmetric_tolerance / departures`. |
| departure SOC deficit mean | ratio | Deficit medio nao-negativo de SOC por departure. |
| departure shortfall beyond tolerance mean | ratio | Deficit medio abaixo de `target_soc - ev_departure_service_tolerance`. |
| departure SOC surplus mean | ratio | Excesso medio nao-negativo de SOC por departure. |
| departure SOC absolute error mean | ratio | Erro absoluto medio de SOC face ao target pedido. |
| departure tolerance | ratio | Tolerancia de servico configurada para o minimo aceitavel no departure. |
| charge total | kWh | Energia carregada em EVs. |
| V2G export total | kWh | Energia exportada por EVs. |

Semantica das tolerancias EV no departure:

- `ev_departure_success_rate` mede cumprimento estrito: `actual_soc >= target_soc`.
- `ev_departure_min_acceptable_rate` mede servico minimo: `actual_soc >= target_soc - ev_departure_service_tolerance`.
- `ev_departure_within_tolerance_rate` mede proximidade simetrica ao target: `abs(actual_soc - target_soc) <= ev_departure_within_tolerance`.
- As duas tolerancias fazem default para `0.05`.

## KPIs BESS

| KPI conceptual | Unidade | O que mede |
|---|---:|---|
| charge total | kWh | Energia carregada na BESS. |
| discharge total | kWh | Energia descarregada. |
| throughput total | kWh | Charge + discharge absolutos. |
| equivalent full cycles | count | Throughput relativo a capacidade. |
| capacity fade | ratio | Degradacao relativa da capacidade. |

## KPIs Deferrable Appliances

| KPI conceptual | Unidade | O que mede |
|---|---:|---|
| completed cycles | count | Ciclos terminados com sucesso. |
| missed cycles | count | Ciclos que passaram a janela sem start valido. |
| service level | ratio | `completed / (completed + missed)`. |
| served energy | kWh | Energia dos ciclos completados. |
| unserved energy | kWh | Energia dos ciclos falhados. |
| average start delay | hours | Media de `start - earliest_start`. |

## KPIs Electrical Service e Fases

| KPI conceptual | Unidade | O que mede |
|---|---:|---|
| violation total | kWh | Energia acima dos limites de servico/fase. |
| violation time step count | count | Numero de steps com violacao. |
| phase imbalance average | ratio | Desequilibrio medio entre fases. |
| phase import peak L1/L2/L3 | kW | Pico de import por fase. |
| phase export peak L1/L2/L3 | kW | Pico de export por fase. |

## KPIs Community Market

Condicionais a `community_market.enabled=true`.

| KPI conceptual | Nivel | Unidade | O que mede |
|---|---|---:|---|
| local traded total | district | kWh | Energia local transacionada. |
| local traded daily average | district | kWh/day | Media diaria local. |
| community import share | district | ratio | Share da demanda servida localmente. |
| settled cost | building/district | eur | Custo apos settlement local. |
| counterfactual cost | building/district | eur | Custo sem mercado local. |
| market savings | building/district | eur | Diferenca entre counterfactual e settled. |

## Export

| Modo | Como obter |
|---|---|
| Em memoria | `env.evaluate_v2()` |
| Export no fim | `CityLearnEnv(..., export_kpis_on_episode_end=True)` |
| Com render | `render_mode="end"` ou `render_mode="during"` |

Para releases novas, qualquer KPI novo deve ser registado em [`releases.md`](releases.md) e, se fizer parte do contrato v2, tambem em [`KPI_V2_TREE.md`](../KPI_V2_TREE.md).
