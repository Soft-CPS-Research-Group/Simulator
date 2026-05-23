# Schema Reference

Esta pagina documenta o contrato do `schema.json`. O schema e a fonte oficial para construir buildings, dispositivos, EVs, appliances, modos de interface, topology dynamic, bundles e mercado.

## Regras Gerais

| Regra | Contrato |
|---|---|
| Caminhos | Sao relativos a `root_directory`, exceto quando sao absolutos. |
| Ficheiros tabulares | CSV e Parquet sao aceites de forma intercambiavel quando as colunas sao equivalentes. |
| Energia | Series energeticas sao `kWh/step`. |
| Potencia e limites | Potencias e limites sao `kW`. |
| Precos | Precos sao por `kWh`. |
| Emissoes | Carbon intensity e `kgCO2/kWh`. |
| Timesteps | Campos de janela, deadline e topology usam indices globais da simulacao. |
| Legacy washing machines | `washing_machines` e `washing_machine_energy_simulation` nao sao o formato oficial. Usar `deferrable_appliances`. |

## Top Level

| Chave | Tipo | Obrigatoria | O que faz |
|---|---:|---:|---|
| `root_directory` | string | sim | Pasta base dos ficheiros do dataset. |
| `random_seed` | int/null | nao | Semente global. |
| `central_agent` | bool | sim | Controlador unico ou multi-agent por building. |
| `seconds_per_time_step` | number | sim | Duracao fisica de cada step. |
| `simulation_start_time_step` | int | sim | Primeiro timestep global. |
| `simulation_end_time_step` | int | sim | Ultimo timestep global. |
| `episode_time_steps` | int/list/null | nao | Tamanho ou janelas de episodio. |
| `rolling_episode_split` | bool/null | nao | Episodios sequenciais. |
| `random_episode_split` | bool/null | nao | Episodios aleatorios. |
| `reward_function` | object/string | sim | Classe do reward e kwargs. |
| `observations` | object | sim | Catalogo flat de observacoes e flags. |
| `actions` | object | sim | Catalogo flat de acoes e flags. |
| `buildings` | object | sim | Definicao de cada building. |
| `agent` | object/string | nao | Configuracao de agente para CLI. |
| `electric_vehicles_def` | object | se houver EVs | Catalogo de EVs. |
| `interface` | `flat`/`entity` | nao | Modo de I/O do ambiente. |
| `observation_bundles` | object | nao | Bundles adicionais da entity interface. |
| `topology_mode` | `static`/`dynamic` | nao | Ativa eventos dinamicos. |
| `topology_events` | list | se dynamic | Eventos add/remove. |
| `community_market` | object | nao | Mercado local e KPIs associados. |
| `ev_departure_within_tolerance` | float | nao | Tolerancia simetrica de accuracy do SOC no departure, default `0.05`. |
| `ev_departure_service_tolerance` | float | nao | Tolerancia inferior de servico EV minimo no departure, default `0.05`. |
| `render_mode` | string | nao | `none`, `during`, `end`. |
| `export_kpis_on_episode_end` | bool | nao | Export de KPIs no fim do episodio. |
| `start_date` | string | nao | Data base de render/export. |
| `debug_timing` | bool | nao | Logs de tempo runtime. |
| `check_observation_limits` | bool | nao | Valida bounds de observacoes. |
| `physics_invariant_checks` | bool | nao | Checks de invariantes fisicos. |
| `metrics_log_interval` | int | nao | Periodicidade de logs. |

## `observations`

Cada entrada tem o formato:

```json
"hour": {
  "active": true,
  "shared_in_central_agent": true
}
```

| Campo | Tipo | Default | O que faz |
|---|---:|---:|---|
| `active` | bool | `false` | Liga a observacao no flat mode. |
| `shared_in_central_agent` | bool | `false` | Se `central_agent=true`, a observacao aparece uma vez no vetor central quando e comum. |

Observacoes especiais com prefixo `electric_vehicle_` sao expandidas por charger. Observacoes com prefixo `deferrable_appliance_` sao expandidas por appliance.

## `actions`

Cada entrada tem o formato:

```json
"electrical_storage": {
  "active": true
}
```

| Campo | Tipo | Default | O que faz |
|---|---:|---:|---|
| `active` | bool | `false` | Liga a acao. |

A acao `electric_vehicle_storage` e expandida para `electric_vehicle_storage_{charger_id}`. A acao `deferrable_appliance` e expandida para `deferrable_appliance_{appliance_id}`.

## `observation_bundles`

Usado apenas na entity interface.

```json
"observation_bundles": {
  "entity_core_electrical": true,
  "entity_community_operational": true,
  "entity_forecasts_existing": false,
  "entity_forecasts_derived": false,
  "entity_action_feedback": false,
  "entity_temporal_derived": true
}
```

| Bundle | Default | O que adiciona |
|---|---:|---|
| `entity_base` | sempre ativo | Features estaticas/essenciais de charger, storage, phases e deferrables. |
| `entity_core_electrical` | `false` | Potencias, energia por step, PV, BESS, EV, eficiencia e metricas eletricas por building. |
| `entity_community_operational` | `false` | Agregados district/community, headrooms, counts e topology version. |
| `entity_forecasts_existing` | `false` | Forecasts ja existentes no dataset. |
| `entity_forecasts_derived` | `false` | Agregados futuros perfeitos do dataset e grid em buckets de 15 min. |
| `entity_temporal_derived` | `false` | Lags, medias curtas e calendario sin/cos. |
| `entity_action_feedback` | `false` | Acao pedida, limitada e aplicada, mais flags de clipping. |

## `buildings`

```json
"buildings": {
  "Building_1": {
    "include": true,
    "type": "citylearn.building.Building",
    "energy_simulation": "Building_1.csv",
    "weather": "weather.parquet",
    "pricing": "pricing.csv",
    "carbon_intensity": "carbon.csv"
  }
}
```

| Chave | Tipo | Obrigatoria | O que faz |
|---|---:|---:|---|
| `include` | bool | nao | Inclui building inicialmente. Em topology dynamic pode comecar inativo. |
| `type` | string | nao | Classe do building. Ex.: `citylearn.building.Building`, `LSTMDynamicsBuilding`. |
| `energy_simulation` | string | sim | Ficheiro principal de cargas, calendario e PV input. |
| `weather` | string/object | sim | Ficheiro meteorologico. |
| `pricing` | string/object | sim | Ficheiro de precos. |
| `carbon_intensity` | string/object | nao | Ficheiro de emissoes. |
| `inactive_observations` | list | nao | Desliga observacoes neste building. |
| `inactive_actions` | list | nao | Desliga acoes neste building. |
| `cooling_device` | object | sim em datasets HVAC | Dispositivo de cooling. |
| `heating_device` | object | sim em datasets HVAC | Dispositivo de heating. |
| `dhw_device` | object | sim em datasets HVAC | Dispositivo de hot water. |
| `cooling_storage` | object | sim em datasets HVAC | Storage termico cooling. |
| `heating_storage` | object | sim em datasets HVAC | Storage termico heating. |
| `dhw_storage` | object | sim em datasets HVAC | Storage termico DHW. |
| `electrical_storage` | object | nao | BESS do building. |
| `pv` | object | nao | PV do building. |
| `chargers` | object | nao | Chargers associados ao building. |
| `deferrable_appliances` | object | nao | Appliances deferiveis normalizados. |
| `charging_constraints` | object | nao | Limites de charging por building/fase. |
| `electrical_service` | object | nao | Limites de import/export por fase e total. |
| `equity_group` | string | nao | Grupo usado nos KPIs de equity. |
| `dynamics` | object | nao | Modelo dinamico para indoor temperature. |
| `occupant` | object | nao | Modelo occupant interaction. |
| `power_outage` | object | nao | Config de outage. |
| `set_point_hold_time_steps` | int | nao | Hold time para occupant interaction. |

## Secoes de Dispositivo

Padrao comum:

```json
"electrical_storage": {
  "type": "citylearn.energy_model.Battery",
  "attributes": {
    "capacity": 50.0,
    "nominal_power": 25.0,
    "depth_of_discharge": 0.9
  },
  "autosize": false,
  "autosize_attributes": {}
}
```

| Campo | Tipo | O que faz |
|---|---:|---|
| `type` | string | Classe Python a instanciar. |
| `attributes` | object | Kwargs do construtor. |
| `autosize` | bool | Se `true`, chama autosizer do building quando existir. |
| `autosize_attributes` | object | Parametros do autosizer. |

### Classes comuns

| Classe | Atributos principais |
|---|---|
| `citylearn.energy_model.HeatPump` | `nominal_power`, `efficiency`, `target_heating_temperature`, `target_cooling_temperature`. |
| `citylearn.energy_model.ElectricHeater` | `nominal_power`, `efficiency`. |
| `citylearn.energy_model.StorageTank` | `capacity`, `efficiency`, `loss_coefficient`, `initial_soc`, `max_output_power`, `max_input_power`. |
| `citylearn.energy_model.Battery` | `capacity`, `nominal_power`, `capacity_loss_coefficient`, `power_efficiency_curve`, `capacity_power_curve`, `depth_of_discharge`. |
| `citylearn.energy_model.PV` | `nominal_power`, `generation_mode`. |

## PV

```json
"pv": {
  "type": "citylearn.energy_model.PV",
  "attributes": {
    "nominal_power": 120.0,
    "generation_mode": "absolute"
  }
}
```

| `generation_mode` | Aliases | Interpreta `energy_simulation.solar_generation` como | Formula |
|---|---|---|---|
| `per_kw` | `profile`, `w_per_kw` | Perfil por 1 kW instalado em `W/kW`. | `generation_kwh = nominal_power * value / 1000`. |
| `absolute` | `absolute_kwh`, `kwh`, `kwh_step`, `energy` | Geracao absoluta ja em `kWh/step`. | `generation_kwh = value`. |

Usa `absolute` quando o dataset vem de medicao real ou de conversor externo ja em energia por step. Usa `per_kw` quando o dataset contem perfil normalizado por capacidade instalada.

## Chargers

```json
"chargers": {
  "AC001": {
    "type": "citylearn.electric_vehicle_charger.Charger",
    "charger_simulation": "chargers/AC001.parquet",
    "attributes": {
      "max_charging_power": 7.4,
      "min_charging_power": 1.4,
      "max_discharging_power": 0.0,
      "min_discharging_power": 0.0,
      "efficiency": 0.95,
      "phase_connection": "L1"
    }
  }
}
```

| Campo | Unidade | O que faz |
|---|---:|---|
| `charger_simulation` | path | Schedule temporal do charger. |
| `max_charging_power` | kW | Limite superior de carga. |
| `min_charging_power` | kW | Minimo tecnico de carga. |
| `max_discharging_power` | kW | Limite V2G. |
| `min_discharging_power` | kW | Minimo tecnico V2G. |
| `efficiency` | ratio | Eficiencia fixa. |
| `charge_efficiency_curve` | ratio vs potencia normalizada | Curva opcional de eficiencia de carga. |
| `discharge_efficiency_curve` | ratio vs potencia normalizada | Curva opcional de eficiencia de descarga. |
| `phase_connection` | `L1`, `L2`, `L3`, `all_phases` | Fase eletrica do charger. |

## EV Definitions

```json
"electric_vehicles_def": {
  "EV_1": {
    "type": "citylearn.electric_vehicle.ElectricVehicle",
    "include": true,
    "battery": {
      "attributes": {
        "capacity": 60.0,
        "nominal_power": 50.0,
        "initial_soc": 0.4,
        "depth_of_discharge": 0.9
      }
    }
  }
}
```

| Campo | O que faz |
|---|---|
| `type` | Classe EV. |
| `include` | Inclui EV no pool inicial. |
| `battery.attributes` | Kwargs de `Battery`. |

O standby loss das baterias EV fica isolado dos defaults de storage estacionario:

- `battery.attributes.loss_coefficient` ausente ou `null` usa default `0.0` em EVs.
- Valores EV explicitos de `loss_coefficient` sao interpretados como ratios de perda horaria.
- Todas as implementacoes de `StorageDevice`, incluindo `Battery` estacionaria e `StorageTank`, convertem valores horarios de `loss_coefficient` para perda efetiva por step com `loss_coefficient * seconds_per_time_step / 3600`.
- Os ranges default de parametros de storage estacionario ficam inalterados.

## Deferrable Appliances

```json
"deferrable_appliances": {
  "washer_1": {
    "type": "citylearn.energy_model.DeferrableAppliance",
    "cycle_profiles_file": "deferrables/washer_profiles.csv",
    "flexibility_schedule_file": "deferrables/washer_schedule.csv",
    "attributes": {
      "trigger_threshold": 0.5
    }
  }
}
```

| Campo | Obrigatorio | O que faz |
|---|---:|---|
| `type` | nao | Classe appliance. Default: `citylearn.energy_model.DeferrableAppliance`. |
| `cycle_profiles_file` | sim | Catalogo de perfis fisicos dos ciclos. |
| `flexibility_schedule_file` | sim | Pedidos/janelas de flexibilidade. |
| `attributes.trigger_threshold` | nao | Threshold de start (default `0.5`; `action > threshold` e interpretado como ON). |

## Dynamic Topology

`topology_mode="dynamic"` requer `interface="entity"`.

```json
"topology_events": [
  {
    "id": "add_pv_b2_t100",
    "time_step": 100,
    "operation": "add_asset",
    "target_member_id": "Building_2",
    "target_asset_type": "pv",
    "target_asset_id": "pv_1",
    "source_member_id": "Building_1",
    "source_asset_id": "pv_1",
    "overrides": {"nominal_power": 80.0}
  }
]
```

| Campo | Tipo | O que faz |
|---|---:|---|
| `id` | string | ID do evento. |
| `time_step` | int | Timestep global onde o evento e aplicado. |
| `operation` | enum | `add_member`, `remove_member`, `add_asset`, `remove_asset`. |
| `target_member_id` | string | Building alvo. |
| `target_asset_type` | enum | `charger`, `deferrable_appliance`, `pv`, `electrical_storage`. |
| `target_asset_id` | string | Asset alvo. |
| `source_member_id` | string | Building fonte para clones. |
| `source_asset_id` | string | Asset fonte para clones. |
| `overrides` | object | Atributos a aplicar depois do clone. |

Remover `deferrable_appliance` cancela ciclos pendentes/running e limpa consumo futuro. Remover PV cria PV zero. Remover storage cria BESS zero.

## Community Market

```json
"community_market": {
  "enabled": true,
  "local_price_ratio_to_grid_import": 0.8,
  "grid_export_price": 0.0,
  "import_member_weights": {
    "Building_1": 1.0,
    "Building_2": 2.0
  },
  "kpis": {
    "community_local_traded_enabled": true,
    "community_self_consumption_enabled": true
  }
}
```

| Campo | Default | O que faz |
|---|---:|---|
| `enabled` | `false` | Ativa settlement local. |
| `local_price_ratio_to_grid_import` | `0.8` | Preco local relativo ao preco de import grid. |
| `intra_community_sell_ratio` | alias | Nome legacy para o mesmo ratio. |
| `grid_export_price` | `0.0` | Preco para export grid. Pode ser escalar/serie conforme configuracao. |
| `import_member_weights` | `{}` | Pesos para repartir energia local entre importadores. |
| `kpis.community_local_traded_enabled` | `true` | Ativa KPI v2 de energia local transacionada. |
| `kpis.community_self_consumption_enabled` | `true` | Ativa KPI v2 de import share local. |

## Combinacoes Importantes

| Caso | Configuracao |
|---|---|
| Treino flat classico | `interface` omitido ou `flat`, `topology_mode` omitido ou `static`. |
| Entity para GNN/Transformer | `interface="entity"`, bundles conforme necessidade. |
| Topologia dinamica | `interface="entity"`, `topology_mode="dynamic"`, `topology_events`. |
| Datasets reais PV absoluto | `pv.attributes.generation_mode="absolute"`. |
| 15 segundos | `seconds_per_time_step=15` e ficheiros ja nessa cadencia. |
| Parquet pesado | Trocar paths `.csv` por `.parquet` mantendo colunas. |
