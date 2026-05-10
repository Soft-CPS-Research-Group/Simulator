# Releases

Este ficheiro e o changelog operacional do fork. A partir de agora, cada mini bump deve explicar o que mudou, o impacto para utilizadores, impacto para datasets, testes corridos e risco de compatibilidade.

Responsavel default por releases deste fork: [@calofonseca](https://github.com/calofonseca).

## Politica de Versao

| Tipo | Quando usar | Exemplo |
|---|---|---|
| Patch | Fixes e features aditivas compativeis. | `0.4.1 -> 0.4.2`. |
| Minor | Mudanca de contrato ou nova capacidade grande. | `0.3.x -> 0.4.0`. |
| Major | Breaking changes amplos ou desalinhamento com API anterior. | `0.x -> 1.0`. |

## Checklist por Release

1. Atualizar `citylearn/__init__.py`.
2. Atualizar este ficheiro.
3. Atualizar README se houver novas features relevantes.
4. Atualizar docs especificas quando muda schema/action/observation/KPI.
5. Correr suite relevante:
   - `.venv/bin/pytest -q`
   - `.venv/bin/python scripts/audit/audit_entity_contract.py --strict`
   - `.venv/bin/python scripts/audit/audit_physics.py`
6. Validar pelo menos um smoke run com dataset representativo.
7. Criar commit/tag.
8. Publicar pacote quando aplicavel.

## Template

```markdown
## vX.Y.Z - YYYY-MM-DD

Release owner: [@calofonseca](https://github.com/calofonseca).

### Summary
- ...

### Added
- ...

### Changed
- ...

### Fixed
- ...

### Dataset/Schema Impact
- ...

### Compatibility
- ...

### Validation
- `...`: pass

### Migration Notes
- ...
```

## v0.4.3 - Contrato de Documentacao e Preparacao de Release

Release owner: [@calofonseca](https://github.com/calofonseca).

### Summary

Release focada em tornar o simulador mais facil de adotar, auditar e integrar, transformando o README num portal de documentacao e adicionando documentacao bilingue.

### Added

| Area | Mudancas |
|---|---|
| README | Mapa de documentacao em ingles por default com links para portugues. |
| Como correr simulacoes | Instalacao, quickstarts, CLI, parametros de `CityLearnEnv`, render/export e validacao. |
| Schema | Referencia de buildings, devices, modos PV, EVs, chargers, deferrables, topology e community market. |
| Interfaces | Contratos flat/entity, entity specs, tabelas, edges e dynamic topology. |
| Acoes | Nomes, ranges, conversoes fisicas e payloads entity. |
| Observacoes | Dicionario com unidades, sentinels, bundles, tabelas entity e edges. |
| Datasets | Contrato CSV/Parquet, conversao de dados reais, 15s e formato de deferrables. |
| KPIs | Explicacao v1/v2, unidades, equacoes e familias EV/BESS/deferrable/community/fases. |
| Features | Inventario de capacidades do simulador. |
| Releases | Checklist, politica de versao e template de changelog. |
| Developer guide | Testes, auditorias, checks de performance e arquitetura interna. |
| Publishing guide | Workflow PyPI e checks locais de build. |
| Docs PT | Mirror completo em portugues em `docs/pt/`. |

### Compatibility

Nao introduz breaking changes de runtime. A release mantem as mudancas aditivas do simulador documentadas em `v0.4.2`.

### Validation

| Check | Resultado |
|---|---|
| Links Markdown locais | Pass. |
| `git diff --check` | Pass. |
| Smoke simulation flat | Pass. |
| Smoke simulation entity | Pass. |
| Smoke simulation 15s Parquet | Pass. |

## v0.4.2 - Expansao Aditiva de Observacoes Fisicas

Release owner: [@calofonseca](https://github.com/calofonseca).

### Summary

Release aditiva focada em melhorar a informacao fisica disponivel para algoritmos sem remover observacoes antigas.

### Added

| Area | Mudancas |
|---|---|
| Chargers | `min_charging_power_kw`, `min_discharging_power_kw`, `charger_efficiency_ratio`. |
| Chargers core | `charge_efficiency_at_max_ratio`, `discharge_efficiency_at_max_ratio`, incoming EV required/departure/hours/ratio. |
| Storage | `min_charge_power_kw`, `min_discharge_power_kw`, `efficiency_ratio`, `round_trip_efficiency_ratio`. |
| Storage core | `current_efficiency_ratio`, `degraded_capacity_kwh`, `soc_min_ratio`. |
| Storage phases | `phase_connection_L1/L2/L3`. |
| Deferrables | `must_run`, average/peak/load factor, peak offset, remaining duration/power, current step power. |
| Building core | `charging_total_service_power_kw`, `charging_phase_L1/L2/L3_power_kw`. |
| PV | `generation_capacity_factor_ratio`. |

### Compatibility

Compatibilidade aditiva. Observacoes antigas e duplicacoes EV/charger foram mantidas.

### Validation

| Comando | Resultado |
|---|---|
| Focused tests de entity/bundles/deferrables/parquet | Pass. |
| Broader simulator tests | Pass. |
| Full suite | Pass. |
| `audit_entity_contract.py --strict` | Pass. |
| `audit_physics.py` | Pass. |

## v0.4.0 - Deferrable Appliances Normalizados

Release owner: [@calofonseca](https://github.com/calofonseca).

### Summary

Introduziu o modelo oficial de `deferrable_appliances` com catalogo de ciclos e schedule de flexibilidade, substituindo o formato antigo de washing machines.

### Added

| Area | Mudancas |
|---|---|
| Schema | `deferrable_appliances` com `cycle_profiles_file` e `flexibility_schedule_file`. |
| Modelo | `DeferrableAppliance` com ciclo pendente/running/missed/completed. |
| Actions | `deferrable_appliance_{id}` no flat e `start` na entity table. |
| Observations | Pending, running, can_start, deadlines, slack, priority, energia restante. |
| KPIs | Completed/missed cycles, service level, served/unserved energy, delay medio. |
| Dynamic topology | Add/remove `deferrable_appliance`. |

### Compatibility

Breaking change de schema: `washing_machines` deixou de ser o formato oficial.

## v0.3.2 - Performance, 15s e Parquet

Release owner: [@calofonseca](https://github.com/calofonseca).

### Summary

Consolidou suporte sub-horario e melhorou performance para datasets grandes, especialmente 15 segundos.

### Added/Changed

| Area | Mudancas |
|---|---|
| Loader | Leitura por janela em CSV e Parquet. |
| Cache | Partilha weather/pricing/carbon quando o ficheiro e igual e sem noise. |
| Parquet | Datasets 15s podem usar `.parquet`. |
| Sub-hourly | Fixtures/testes para 15s, 1min, 5min, 15min e 1h. |
| PV | Modo `absolute` para datasets reais em `kWh/step`. |
| Unit contract | Documentacao formal de kWh/step, kW, preco e emissoes. |

## Pre-v0.3 - EVs e Washing Machines Iniciais

Fork owner: [@calofonseca](https://github.com/calofonseca).

### Summary

Antes da consolidacao dos deferrables normalizados, o fork introduziu EVs/chargers e washing machines iniciais no simulador.

### Added

| Area | Mudancas |
|---|---|
| EVs | EV definitions, charger schedules, connected/incoming EV observations. |
| Chargers | Acoes de charging/discharging e schedules por charger. |
| Washing machines | Primeira versao de appliances flexiveis. |
| Algorithms integration | Preparacao para RBC/MADDPG e repos externos. |
