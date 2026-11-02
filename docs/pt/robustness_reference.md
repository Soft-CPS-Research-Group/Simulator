# Referencia de Robustez

Robustez e opcional e orientada por dataset. Se `schema["robustness"]["enabled"]` estiver ausente ou `false`, o `CityLearnEnv` comporta-se como antes.

Versao inglesa: [../robustness_reference.md](../robustness_reference.md).

## Schema

```json
"robustness": {
  "enabled": true,
  "events_file": "robustness_events.csv",
  "random_seed": 0,
  "missing_replacement_value": -9999.0,
  "modules": {
    "observations": {"enabled": true},
    "forecasts": {"enabled": true},
    "actions": {"enabled": true},
    "assets": {"enabled": true}
  }
}
```

`events_file` pode ser CSV ou Parquet e e resolvido a partir da root do dataset.

## Ficheiro de Eventos

| Coluna | Significado |
|---|---|
| `event_id` | ID estavel usado em metadata e auditoria. |
| `module` | `observation`, `forecast`, `action` ou `asset`. |
| `target_type` | `district`, `building`, `storage`, `charger`, `ev`, `pv` ou `deferrable_appliance`. |
| `target_id` | ID/nome da entidade, ou `*` para targets compativeis. |
| `target_feature` | Nome da feature/acao. Para assets usa `telemetry`, `control` ou `both`. |
| `start_time_step`, `end_time_step` | Janela inclusiva em timesteps globais. |
| `mode` | Modo de perturbacao dependente do modulo. |
| `value`, `std`, `min_value`, `max_value` | Parametros opcionais do modo. |
| `replacement_value` | Sentinela para missing/telemetry; default `missing_replacement_value`. |
| `delay_steps` | Atraso de acao; default `1`. |

## Modos

| Modulo | Modos |
|---|---|
| observation/forecast | `missing`, `noise`, `bias`, `stuck`, `clip` |
| action | `dropout`, `noise`, `bias`, `stuck`, `delay`, `clip` |
| asset | `unavailable` |

Corrupcao de observations e forecasts afeta apenas o que o agente ve. Rewards, KPIs fisicos e settlement de demand response continuam a usar o estado real do simulador. Eventos de action e asset-control alteram a acao efetivamente aplicada, portanto podem alterar a fisica e KPIs normais.

## Validação e composição

Os identificadores dos eventos têm de ser únicos em todo o ficheiro, incluindo
módulos desativados. Os passos globais de início/fim são inteiros não negativos,
com `end >= start`; `delay_steps` é um inteiro positivo. Valores numéricos integrais
como `1.0` são aceites; valores fracionários são rejeitados, sem truncamento.

Os parâmetros numéricos fornecidos têm de ser finitos, `std >= 0`, e os limites
de clip fornecidos têm de respeitar `min_value <= max_value`. Campos CSV vazios
ou nulos Parquet deixam parâmetros opcionais por especificar. Valores explícitos
inválidos, como `nan`, `inf` ou texto no CSV, são rejeitados. Mantêm-se os valores
por omissão: bias/ruído zero, atraso de um passo e limites clip não especificados
sem restrição desse lado. A seed de robustez no schema é inteira e a sentinela
de missing é finita. Os erros identificam evento/campo ao carregar as definições,
antes da execução do episódio.

Eventos do canal de ações são compostos pela ordem das linhas do ficheiro, mesmo
que as janelas comecem em passos diferentes. Bias seguido de clip pode produzir
um resultado diferente da ordem inversa. A indisponibilidade de `control`/`both`
do ativo é aplicada **depois de todas as transformações do canal de ações**,
independentemente da posição no ficheiro: o comando afetado fica a zero.
As aplicações do canal e a restrição de disponibilidade ficam ambas registadas.
O histórico do atraso conserva os comandos pedidos e o reset limpa o estado.

A indisponibilidade de controlo é uma restrição lógica, não uma desconexão física
ou um modelo universal de avaria do equipamento. As perdas passivas da bateria
ou um ciclo não interrompível já em curso podem continuar; os comandos bloqueados
não podem voltar a carregar ou iniciar um ciclo através de um bias de ação.

## Limites das observações e alvos dinâmicos

Em `known_faults`, os limites são alargados apenas nos canais de telemetria
afetados. O envelope inclui períodos sem falha, composição pela ordem do ficheiro,
sentinelas, clip e valores retidos por `stuck`. Ruído gaussiano positivo tem suporte
ilimitado e os canais afetados passam a ter limites infinitos. Os valores de bias
e clip não são alterados; os limites arredondam para fora para acomodar float32.
Módulos desativados e eventos apenas de ação não alargam os limites da telemetria.
Em `hidden_faults`, os limites numéricos das tabelas mantêm-se uniformemente
ilimitados para não revelarem o canal afetado; a lista permitida e a máscara de
metadados continuam ativas. Os espaços de relações/arestas não são alterados.

Os alvos dinâmicos são validados no inventário declarado de membros/ativos,
incluindo clones futuros e adições de ativos. Uma vista estrutural leve permite
validar sem fazer reset ou controlar os equipamentos reais. IDs/features
desconhecidos e portas de controlo desativadas continuam a ser erros. Um alvo
conhecido pode estar ausente em parte ou em toda a janela: durante a ausência não
há aplicação da perturbação. Os IDs são os IDs estáveis das entity specs. Amostras
stuck e comandos pedidos com atraso conservam a identidade lógica evento/alvo
durante ausências; o reset limpa esse estado. Features futuras declaradas também
são válidas na lista permitida do modo oculto, mantendo as proibições de canais
alternativos limpos e rótulos de diagnóstico.

## Diagnostico Entity

Ativar o bundle de diagnostico quando se usa entity observations:

```json
"observation_bundles": {
  "entity_robustness": {"active": true}
}
```

Isto adiciona features na tabela `district` para estado ativo, contagens do step anterior e outages de assets ativos. Metadata leve tambem fica em `observations["meta"]["robustness"]`.

## KPIs

Quando robustez esta ativa, `evaluate_v2()` adiciona contadores district/building para eventos, timesteps ativos, corrupcoes de observation/forecast/action, assets indisponiveis, observations missing e action dropouts.

Os contadores históricos mantêm o significado: efeitos evento/alvo únicos por
índice global, incluindo **telemetria terminal consultada**. Consultar novamente a
mesma observação não aumenta as contagens. As aplicações são registadas antes da
máscara do modo oculto; canais partilhados do distrito não pertencem a um membro.
`robustness_active` e `active_event_ids` indicam janelas de eventos habilitados,
mesmo com o alvo ausente; `applied_event_ids` identifica efeitos realmente
aplicados no índice consultado. Um evento agendado não prova que afetou equipamento.

`env.get_robustness_audit()` devolve `robustness-audit-v1`, com resumos `legacy`
e `executed_intervals` separados; este último usa só intervalos físicos concluídos.
`district` e `events` expõem eventos configurados/habilitados/agendados, índices
consultados, intervalos concluídos, observações terminais/ainda não executadas e
contagens evento-passo na janela e com aplicação. A cobertura divide os eventos-
passo com pelo menos uma aplicação correspondente pelos eventos-passo habilitados
na janela. Não mede disponibilidade individual de sensores nem cumprimento de
serviço físico; um wildcard continua a contar uma vez por evento-passo. Um
denominador vazio dá `null`. A telemetria usa índices efetivamente consultados,
incluindo o terminal; o controlo usa intervalos concluídos. Uma execução sem
observações não inventa aplicações sensoriais. O registo de índices consultados
fica desligado quando a robustez está desativada. As consultas são passivas e
devolvem um retrato independente.

A mesma auditoria está em `evaluate_v2().attrs['robustness_audit']`. A exportação
de KPIs com robustez ativa acrescenta `<nome-do-ficheiro-kpi>.robustness.json` junto
ao CSV/Parquet. Os KPIs existentes conservam nomes e fórmulas; o JSON explicita
populações, denominadores e cobertura para o investigador, fora da entrada da
política em modo oculto.

Em `MultiCommunityEnv`, a robustez continua independente por comunidade filha. Linhas de portfolio agregam KPIs de robustez com sufixo `_count` como os restantes KPIs de contagem.


## Contrato de observação do agente

O modo por omissão é `known_faults`. Estudos entity sem rótulos de eventos devem
selecionar explicitamente `hidden_faults` e uma lista de medições permitidas.
Os canais numéricos restantes são mascarados; alternativas equivalentes limpas
às medições perturbadas são rejeitadas. Consultar os
[contratos de comparação e observação](../experiment_comparisons.md) para configuração,
limites e migração.
