# Contrato de unidades de dados

Este documento fixa o contrato de unidades esperado pelo simulador. O objetivo é
evitar ambiguidades quando se usam datasets horários, sub-horários ou dados reais
que chegam originalmente em potência.

## Regra base

O motor físico do CityLearn contabiliza energia por passo de simulação.

- Colunas energéticas de séries temporais: `kWh/step`.
- Potências nominais, limites, contratos e ações físicas: `kW`.
- Capacidades de armazenamento: `kWh`.
- Estado de carga (`soc`): fração em `[0, 1]`.
- Preços de energia: moeda por `kWh`.
- Intensidade carbónica: massa de CO2 por `kWh`.

O `seconds_per_time_step` do schema deve representar a cadência física do dataset.
Se o dataset e o schema tiverem resoluções diferentes, o simulador aplica a
conversão de compatibilidade existente e imprime um aviso explícito
`[CityLearn][unit-conversion]`. Esse modo é suportado, mas o caminho preferido é
gerar o dataset já na mesma resolução do schema.

## Séries temporais principais

| Campo | Unidade | Notas |
| --- | --- | --- |
| `energy_simulation.non_shiftable_load` | `kWh/step` | Energia consumida no passo. |
| `energy_simulation.cooling_demand` | `kWh/step` | Demanda térmica no passo. |
| `energy_simulation.heating_demand` | `kWh/step` | Demanda térmica no passo. |
| `energy_simulation.dhw_demand` | `kWh/step` | Demanda térmica no passo. |
| `energy_simulation.solar_generation` | perfil PV por kW instalado ou `kWh/step` | Depende de `pv.attributes.generation_mode`; ver secção de PV. |
| `pricing.electricity_pricing` | moeda/`kWh` | Não é escalado pelo tamanho do passo. |
| `carbon_intensity.carbon_intensity` | CO2/`kWh` | Não é escalado pelo tamanho do passo. |
| `weather` | unidade física da variável | Temperatura, radiação, humidade, etc. não são energia. |

## Equipamentos

| Elemento | Campo típico | Unidade |
| --- | --- | --- |
| Bateria/armazenamento elétrico | `capacity` | `kWh` |
| Bateria/armazenamento elétrico | `nominal_power` | `kW` |
| Storage térmico | `capacity` | `kWh` |
| Storage térmico | `max_input_power`, `max_output_power` | `kW` |
| Bombas de calor, resistências, chillers | `nominal_power` | `kW` elétrico |
| PV | `nominal_power` | `kW` |
| Carregador EV | `max_charging_power`, `max_discharging_power` | `kW` |
| Veículo elétrico | bateria `capacity` | `kWh` |
| Limites de rede/fase/contrato | limites de potência | `kW` |

Dentro de um passo, qualquer limite em `kW` é comparado contra energia através de:

```text
max_energy_kwh_step = limit_kw * seconds_per_time_step / 3600
```

## Conversão de dados reais

Quando a fonte real chega em potência, o conversor deve produzir colunas
energéticas em `kWh/step` antes de gerar o dataset CityLearn.

```text
energy_kwh_step = mean_power_kw_over_step * seconds_per_time_step / 3600
```

Regras práticas:

- Séries em `kW`: agregar para a cadência do simulador com média temporal e
  converter para `kWh/step`.
- Leituras intervalares já em `kWh`: somar os intervalos que pertencem ao passo.
- Contadores acumulados em `kWh`: fazer diferença entre leituras e depois agregar.
- Preços e emissões: alinhar ou fazer forward-fill para a cadência do simulador,
  sem multiplicar pelo tamanho do passo.
- Limites por fase, potência contratada, ratings de baterias/carregadores/PV:
  manter em `kW`.
- Timestamps de EVs: converter tempos absolutos para os passos inteiros usados na
  simulação de carregadores.
- CSVs de carregadores EV: `electric_vehicle_departure_time` e
  `electric_vehicle_estimated_arrival_time` são contagens de passos até ao evento,
  não horas absolutas do dia. Ao converter de 1h para 15s, um valor `12` passa para
  `2880` no primeiro subpasso e decrece a cada passo de 15s.
- Datasets abaixo de 1 minuto: incluir uma coluna opcional `seconds` no CSV de
  `energy_simulation` quando possível. Sem essa coluna, `hour` + `minutes` não
  conseguem inferir a cadência sub-minuto, e o simulador assume que a cadência do
  dataset é a declarada em `seconds_per_time_step`.
- Flags de qualidade como `generated`: manter no pipeline de ingestão/conversão.
  Elas não são, por si só, campos físicos nativos do CityLearn.

Exemplo para dados a cada 15 segundos:

```text
load_kw = 55.0
seconds_per_time_step = 15
load_kwh_step = 55.0 * 15 / 3600 = 0.2291667
```

Este valor pequeno está correto fisicamente. Se a interface do agente precisar de
valores em `kW` ou normalizados, essa conversão deve acontecer na camada de
observações do agente, não no balanço energético interno.

Na interface entity, a flexibilidade EV deve ser consumida preferencialmente em
features físicas/normalizadas derivadas:

- `hours_until_departure`: tempo físico até saída.
- `time_until_departure_ratio`: `hours_until_departure / 24`, limitado a `[0, 1]`.
- `energy_to_required_soc_kwh`: energia ainda necessária para chegar ao SOC alvo.
- `required_average_power_kw`: potência média necessária até à saída.
- `charging_slack_kw`: margem entre potência máxima do carregador e potência média
  necessária.
- `charging_priority_ratio`: urgência em `[0, 1]`, calculada a partir da razão entre
  potência média necessária e potência máxima do carregador.

## PV

O campo `energy_simulation.solar_generation` é o caso mais fácil de confundir.
O simulador lê esse campo do dataset. Por compatibilidade com o CityLearn
original, o modo default trata-o como perfil de geração por `1 kW` de PV
instalado e depois multiplica por `pv.nominal_power`.

Em termos práticos:

```text
pv_generation_kwh_step = pv.nominal_power_kw * solar_generation / 1000
```

Para usar dados reais absolutos, declarar o modo no schema:

```json
"pv": {
  "type": "citylearn.energy_model.PV",
  "autosize": false,
  "attributes": {
    "nominal_power": 120.0,
    "generation_mode": "absolute"
  }
}
```

Com `generation_mode = "absolute"`, `energy_simulation.solar_generation` passa a
ser interpretado diretamente como energia PV real em `kWh/step`:

```text
pv_generation_kwh_step = solar_generation
```

Neste modo, `pv.nominal_power` continua útil como potência instalada/limite físico
em `kW`, mas já não escala a série do dataset.

Se os dados reais chegam como potência PV absoluta (`pv_power_kw`) e se conhece a
potência instalada (`pv_nominal_kw`), o conversor deve codificar:

```text
pv_energy_kwh_step = pv_power_kw * seconds_per_time_step / 3600
solar_generation = pv_energy_kwh_step
```

Se for necessário manter compatibilidade com datasets antigos em modo `per_kw`,
então o conversor deve codificar:

```text
solar_generation = pv_energy_kwh_step / pv_nominal_kw * 1000
```

Portanto, o simulador usa dados vindos do dataset para PV. O detalhe importante é
escolher o modo correto: `per_kw` para datasets CityLearn antigos; `absolute`
para datasets reais já convertidos para `kWh/step`.

## KPIs

Os KPIs energéticos assumem que consumos, importações, exportações e geração estão
em `kWh/step`. Custos e emissões são calculados multiplicando energia por preço
ou intensidade:

```text
cost = energy_kwh_step * price_per_kwh
emissions = energy_kwh_step * carbon_intensity_per_kwh
```

Por isso, preços e emissões não devem ser reduzidos em passos sub-horários; a
energia já contém a duração do passo.

## Checklist de validação

Antes de usar um dataset novo:

- Confirmar que a cadência do CSV coincide com `seconds_per_time_step`.
- Para cadências sub-minuto, confirmar que a coluna `seconds` existe ou que o
  schema declara a mesma cadência física do CSV.
- Confirmar que cargas, demandas e energia PV codificada resultam em `kWh/step`.
- Confirmar que ratings e limites continuam em `kW`.
- Confirmar que preços e emissões continuam por `kWh`.
- Executar a auditoria física:

```bash
python scripts/audit/audit_physics.py --output outputs/audit/physics_audit.json
```

O audit cobre observações finitas, compatibilidade de ações, invariantes de
runtime, SOC, consistência `kW`/`kWh/step`, balanço líquido, limites de potência
de componentes e avaliação de KPIs.
