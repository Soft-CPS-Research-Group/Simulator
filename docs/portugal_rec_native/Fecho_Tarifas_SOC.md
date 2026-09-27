# Fecho das tarifas e das duas sessões EV

Revisão `portugal-native-978be68b-v2.1`, 27 de setembro de 2026.
O núcleo do simulador continua no commit original `978be68b`.

## Tarifas nas histórias

O atlas Premium, os seis ficheiros de histórias das famílias e a coluna `story`
dos quatro catálogos de membros indicam agora a mesma tarifa BTN bi-horária
indexada a OMIE 2023. `source_tariff_id` conserva a atribuição anterior apenas
como proveniência. Os preços e a sua configuração não foram alterados.
A correção foi incorporada no construtor, para se manter numa reconstrução.

## Causa dos dois défices

O visitante `EV-V028-01-1` utiliza o carregador `CHG-M028-01` do Café 028.
A bateria tem 64 kWh e o carregador 11,04 kW. As sessões de 23 de julho e
27 de agosto tinham chegada às 13:30 e partida às 14:45, hora de Lisboa.

Mesmo sem concorrência ou limitação da instalação, em 75 minutos a energia
máxima armazenável é `11,04 × 1,25 × 0,95 × √0,96 = 12,8451 kWh`.
O fator `√0,96` é a eficiência de carga aplicada pelo modelo original da bateria.
Este limite superior é inferior à energia pedida em ambas as visitas:

| Sessão | SOC à chegada | SOC pedido | Energia necessária | Défice inevitável anterior |
|---|---:|---:|---:|---:|
| 23 de julho — S0092 | 67,73% | 90,66% | 14,6766 kWh | 1,8315 kWh |
| 27 de agosto — S0109 | 70,42% | 90,82% | 13,0590 kWh | 0,2139 kWh |

Os défices reproduzem os registos anteriores dentro do arredondamento numérico.
Assim, estes dois pedidos eram inviáveis com a duração declarada; não resultavam
só da escolha de uma referência de carregamento.

## Correção aplicada e verificação

As duas partidas passam para as **15:00**, prolongando cada visita de 75 para
90 minutos. Preservam-se chegadas, SOC pedidos, capacidades, potências e viagens.
O café fecha às 15:00 nesses domingos. Não há conflito com a sessão seguinte
do carregador, outra utilização do veículo ou a viagem posterior.
Os contadores de partida e o catálogo de sessões foram atualizados em conjunto.
A correção é comum às cinco variantes Core-30, através do ficheiro partilhado.

Foram executadas apenas dez janelas curtas no simulador original: as duas
sessões em cada uma das cinco variantes. As ações são um plano fixo de carga
calculado a partir da energia pedida. Em agosto é incluído o visitante do segundo
carregador, com o seu próprio pedido de SOC.

**Todos os pedidos presentes nessas janelas foram cumpridos, sem défice e sem
violações do limite total ou dos limites por fase da instalação.** O SOC final
dos dois visitantes corrigidos foi aproximadamente 90,6598% e 90,8256%.
Não se alteraram o simulador, os limites físicos ou as preferências de SOC.

A prova refere-se a estas duas janelas. Não houve treino, comparação de
controladores, reavaliação anual ou certificação das restantes sessões.
Os ficheiros de mobilidade mantêm os diagnósticos históricos de origem;
o simulador original reinicializa o SOC à chegada e não executa esse balanço
longitudinal. A evidência corrente está em `results/portugal_rec_native/`.
