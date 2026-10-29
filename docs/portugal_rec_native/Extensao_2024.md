# Extensão de 2024

Nota de versão: este documento descreve a extensão física e a tarifa histórica da edição v3.1. A edição v3.2 preserva os inputs físicos e substitui a tarifa por OMIE/1000 com piso zero, com mercado ativo a 80%; ver [Tarifas e mercado](Tarifas_e_mercado.md).

## Continuidade e dimensão

As nove configurações passam a cobrir 731 dias consecutivos: 35 040 registos de 2023 e 35 136 de 2024, total de 70 176. O novo ano é continuação do anterior, não uma reposição da comunidade inicial. Os índices absolutos são 0–35039 para 2023 e 35040–70175 para 2024. Uma experiência iniciada apenas em 2024 precisa de respeitar o estado/topologia acumulados; cortar o calendário não equivale automaticamente a repor esse estado.

Os mesmos membros, painéis, baterias, carregadores, veículos, perfis de máquinas e contratos elétricos foram conservados. Não há redimensionamento. O novo ano não acrescenta eventos de crescimento ou saída; preserva o resultado dos eventos de 2023. As variantes estáticas mantêm o catálogo completo e as dinâmicas mantêm os sobreviventes.

## Novo ano, novas séries

- Calendário civil de Lisboa, 29 de fevereiro e transições de hora de 31 de março e 27 de outubro, com 92 e 100 quartos de hora nesses dias. Feriados móveis e férias escolares correspondem a 2024; a escola fictícia reinicia a 16 de setembro. A pausa letiva de verão começa a 29 de junho. As férias familiares conservam o dia e mês, ajustando o ordinal do ano bissexto.
- Perfis E-REDES de 2024, respeitando a ordem cronológica do ficheiro e os dias de mudança da hora. Servem de referência de forma para as cargas domésticas; não são medições das famílias fictícias.
- Irradiação PVGIS6 SARAH-3 de 2024 e temperatura, humidade e vento ERA5 via Open-Meteo, no ponto de Lisboa usado em 2023. O PV usa a mesma transposição, inclinação, azimute, área disponível, perdas e limite de inversor. A resolução de 15 minutos é interpolada a partir de fontes horárias, não medida.
- Foram encontrados nove instantes com componentes solares em falta (sete diurnos e dois noturnos). Apenas essas componentes foram substituídas por ERA5 do mesmo período; todos os valores substituídos estão em `data/portugal_reference_2024/weather_gap_fills.json`. Os originais preservam as lacunas.
- Novas realizações determinísticas das rotinas, visitas e ciclos. As características físicas dos equipamentos não são sorteadas de novo. A carga base e AQS conservam o orçamento diário de referência; aquecimento/arrefecimento respondem à razão dos graus-hora de 2024 face a 2023. Estes continuam a ser consumos elétricos simplificados, não um modelo térmico de conforto.
- Preços OMIE horários de 2024, convertidos corretamente do calendário de Madrid para UTC. A tarifa BTN comum incorpora os acessos de janeiro e junho e o encargo TSE a partir de abril. Não foi aplicado piso artificial: nesta referência, o mínimo de compra à rede em 2024 é 0,02644 €/kWh.
- Carbono calculado com produção REN de 2024 e intensidade diária espanhola REE nas importações. O coeficiente térmico calibrado com APA 2023 é mantido; não se afirma dispor de uma nova calibração APA 2024. É uma estimativa operacional média, não emissão marginal medida.

## Fronteira entre anos

Foram unidas 54 sessões de estacionamento doméstico entre as quatro bases, correspondendo a 124 continuações efetivas nos schemas. Mantêm identidade e pedido de SOC, sem uma chegada ou reposição de SOC artificial à meia-noite. Atualizou-se apenas a contagem para a partida que o ficheiro anual de 2023 antes truncava.

Os consumos, PV, preços realizados, meteorologia e carbono de 2023 foram preservados. No conjunto contínuo, as previsões de preço das últimas 1/6/24 horas de 2023 passaram a apontar para 2024 e a respetiva tarifa. A edição anual congelada de 2023 não foi alterada. Na revisão v3.1, as previsões de preço no fim de 2024 usam um dia de apoio externo de janeiro de 2025, respeitando os horizontes e a publicação do OMIE. A meteorologia já tinha horizontes completos. O período simulado não muda; as previsões de consumo/PV derivadas pelo motor ainda têm uma limitação própria na fronteira do episódio. Ver Previsoes_e_historias.md.

Na sessão `EV-M044-01-2024-S0229`, um carro de 82 kWh tinha chegada às 18h de 22 de novembro e partida às 7h do dia seguinte, pedindo 100%. A chegada original a 40,4% não permitia cumprir o pedido com o carregador de 3,68 kW. A regra de preparação de uma viagem antecipa a reserva para a sessão anterior: o pedido anterior passa a 59,42% e a chegada desta sessão a 48,15%. Mantêm-se os horários, a viagem, a potência e o pedido final de 100%. O motivo está registado em `target_reason`.

## Eventos

As campanhas de health, cortes, ruído e DR de 2024 mantêm duração e severidade e deslocam as datas de 2023 em 52 semanas de calendário civil, conservando o dia da semana e hora local. Não são observações históricas de falhas reais. As campanhas de confirmação continuam reservadas em ficheiros próprios. O ruído afeta canais de observação, segundo as capacidades originais do motor, sem ser introduzido na verdade física dos dados.

As saídas definitivas de membros, veículos associados, carregadores e máquinas são respeitadas nos inputs de 2024. O standby desaparece com a bateria. Um equipamento substituto de 2023 conserva a sua identidade própria; não se reintroduz o equipamento retirado.

## Verificação

Foram aprovadas 2828 verificações estáticas, sem falhas: eixo temporal, calendário, igualdade das tarifas, preservação física de 2023, inventário, limites dos inversores, identidades e sobreposições EV, horários dos serviços, janelas das máquinas, eventos, integridade de ficheiros e ausência de alterações ao motor. Foram verificadas individualmente 28111 sessões EV completas de 2024, com potência aceite, perdas e curva de redução de potência da bateria.

A verificação não inclui política de controlo, viabilidade anual conjunta, comparação de algoritmos, margem de melhoria nem recalibração comercial de DR. Esses trabalhos permanecem fora do pedido atual.

Os totais anuais por membro encontram-se em `catalogs/member_year_totals.parquet`, separados por ano. Os campos `annual_*` dos catálogos anteriores conservam estatísticas de 2023. Os esquemas em `evaluation/` determinam os eventos efetivamente executáveis; os antigos catálogos em `events/` conservam o desenho de origem. `catalogs/lifecycle_plan.json` é agora gerado dos schemas e concorda com as histórias. O plano anterior está identificado como histórico em `sources/2023_original_lifecycle_plan.json`.
