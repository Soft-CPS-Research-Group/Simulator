# Atlas das famílias e instalações · Premium · motor original

As histórias descrevem um inventário sintético, não pessoas ou instalações medidas. As capacidades, contratos e rotinas de origem são conservados. As fases individuais das máquinas e a potência adicional de inversor de backup não são integralmente impostas pelo motor original; consultar o README corrente para a diferença entre inventário e execução. As histórias descrevem o inventário da família. As datas abaixo correspondem ao catálogo corrente em `evaluation/`; a disponibilidade de cada ativo segue os eventos da configuração. Quem sai deixa o cenário com todos os seus ativos. Não é modelada atividade privada posterior à saída.

## Member_001

Ana Silva · agregado 001: 1 pessoa(s), moradia de 90 m²; trabalho diurno fora de casa. Férias de 14 dias a partir do dia 224 do ano, com atividade doméstica reduzida. Equipamento: 9 módulos de 420 Wp (3.78 kWp), inversor 3.44 kW; bateria fixa de 5.12 kWh / 2.56 kW; 1 carregador(es), 2 lugares de estacionamento no cenário. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M001-01 (50 kWh úteis, AC até 11.04 kW, preferência reserve_80). Os pedidos variam por sessão, entre 50% e 100%. O mesmo automóvel pode carregar em casa e em Member_081, com itinerários descritos nos dados de origem; no motor original, o SOC é prescrito em cada chegada e a energia das viagens não é contabilizada longitudinalmente. No cenário dinâmico corrente: retira definitivamente BESS-M001-01 em 2023-05-16 19:00:00+01:00; pedidos pendentes desse equipamento são cancelados; instala BESS-M001-02 em 2023-05-17 19:00:00+01:00, com a identidade e o estado inicial declarados; sai definitivamente do cenário, com todos os seus ativos, em 2023-08-14 13:00:00+01:00; depois desta data não há consumo, produção, ciclos, viagens ou carregamento atribuídos ao membro. Personagens e instalações são fictícias.

## Member_002

João Santos · agregado 002: 1 pessoa(s), apartamento de 55 m²; teletrabalho às terças e quintas e deslocações nos restantes dias úteis. Férias de 11 dias a partir do dia 244 do ano, com atividade doméstica reduzida. Equipamento: 1 carregador(es), 2 lugares de estacionamento no cenário; ciclos de lavagem programáveis, com janela alargada. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M002-01 (77 kWh úteis, AC até 11.04 kW, preferência reserve_80). Os pedidos variam por sessão, entre 50% e 100%. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_003

Inês Ferreira · agregado 003: 4 pessoa(s), moradia de 131 m²; turnos noturnos em blocos de quatro noites, com regresso de manhã. Férias de 8 dias a partir do dia 239 do ano, com atividade doméstica reduzida. Equipamento: 1 carregador(es), 2 lugares de estacionamento no cenário. Ligação monofásica, 10.35 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M003-01 (64 kWh úteis, AC até 11.04 kW, preferência high_90). Os pedidos variam por sessão, entre 50% e 100%. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_004

Miguel Pereira · agregado 004: 3 pessoa(s), moradia de 122 m²; rotinas de levar e buscar crianças, com regressos intermédios a casa. Pelo menos uma das pessoas é uma criança em idade escolar. Férias de 12 dias a partir do dia 195 do ano, com atividade doméstica reduzida. Equipamento: 11 módulos de 420 Wp (4.62 kWp), inversor 4.20 kW; bateria fixa de 5.12 kWh / 2.56 kW. Ligação monofásica, 4.60 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_005

Sofia Costa · agregado 005: 3 pessoa(s), moradia de 140 m²; presença diurna e deslocações curtas sem horário laboral fixo. Férias de 14 dias a partir do dia 187 do ano, com atividade doméstica reduzida. Equipamento: 14 módulos de 420 Wp (5.88 kWp), inversor 5.34 kW; bateria fixa de 5.12 kWh / 2.56 kW; 1 carregador(es), 2 lugares de estacionamento no cenário. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M005-01 (50 kWh úteis, AC até 11.04 kW, preferência balanced_70). Os pedidos variam por sessão, entre 50% e 100%. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_006

Pedro Rodrigues · agregado 006: 2 pessoa(s), moradia de 73 m²; trabalho até tarde e consumo doméstico deslocado para o fim da noite. Férias de 17 dias a partir do dia 243 do ano, com atividade doméstica reduzida. Equipamento: 2 carregador(es), 2 lugares de estacionamento no cenário; ciclos de lavagem programáveis, com janela curta. Ligação trifásica, 13.80 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M006-01 (64 kWh úteis, AC até 11.04 kW, preferência reserve_80); EV-M006-02 (77 kWh úteis, AC até 11.04 kW, preferência reserve_80). Os pedidos variam por sessão, entre 50% e 100%. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_007

Marta Martins · agregado 007: 2 pessoa(s), moradia de 125 m²; trabalho diurno fora de casa. Férias de 14 dias a partir do dia 201 do ano, com atividade doméstica reduzida. Equipamento: 5 módulos de 420 Wp (2.10 kWp), inversor 1.91 kW. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: sai definitivamente do cenário, com todos os seus ativos, em 2023-02-13 00:00:00+00:00; depois desta data não há consumo, produção, ciclos, viagens ou carregamento atribuídos ao membro. Personagens e instalações são fictícias.

## Member_008

Rui Sousa · agregado 008: 2 pessoa(s), apartamento de 131 m²; teletrabalho às terças e quintas e deslocações nos restantes dias úteis. Férias de 13 dias a partir do dia 229 do ano, com atividade doméstica reduzida. Equipamento: 1 carregador(es), 2 lugares de estacionamento no cenário. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M008-01 (64 kWh úteis, AC até 11.04 kW, preferência high_90). Os pedidos variam por sessão, entre 50% e 100%. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_009

Leonor Fernandes · agregado 009: 2 pessoa(s), moradia de 66 m²; turnos noturnos em blocos de quatro noites, com regresso de manhã. Férias de 7 dias a partir do dia 197 do ano, com atividade doméstica reduzida. Equipamento: ciclos de lavagem programáveis, com janela curta. Ligação monofásica, 5.75 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: retira definitivamente DEF-M009-01 em 2023-02-15 00:00:00+00:00; pedidos pendentes desse equipamento são cancelados; retira definitivamente DEF-M009-02 em 2023-02-15 00:00:00+00:00; pedidos pendentes desse equipamento são cancelados; sai definitivamente do cenário, com todos os seus ativos, em 2023-02-15 00:00:00+00:00; depois desta data não há consumo, produção, ciclos, viagens ou carregamento atribuídos ao membro. Personagens e instalações são fictícias.

## Member_010

Diogo Gonçalves · agregado 010: 4 pessoa(s), moradia de 94 m²; rotinas de levar e buscar crianças, com regressos intermédios a casa. Pelo menos uma das pessoas é uma criança em idade escolar. Férias de 7 dias a partir do dia 219 do ano, com atividade doméstica reduzida. Equipamento: 13 módulos de 420 Wp (5.46 kWp), inversor 4.96 kW; bateria fixa de 10.24 kWh / 5.00 kW; 1 carregador(es), 2 lugares de estacionamento no cenário. Ligação trifásica, 17.25 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M010-01 (95 kWh úteis, AC até 11.04 kW, preferência balanced_70). Os pedidos variam por sessão, entre 50% e 100%. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_011

Teresa Gomes · agregado 011: 5 pessoa(s), apartamento de 152 m²; presença diurna e deslocações curtas sem horário laboral fixo. Férias de 18 dias a partir do dia 224 do ano, com atividade doméstica reduzida. Equipamento: ciclos de lavagem programáveis, com janela alargada. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_012

António Lopes · agregado 012: 1 pessoa(s), moradia de 71 m²; trabalho até tarde e consumo doméstico deslocado para o fim da noite. Férias de 9 dias a partir do dia 225 do ano, com atividade doméstica reduzida. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação monofásica, 4.60 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_013

Ana Marques · agregado 013: 4 pessoa(s), moradia de 141 m²; trabalho diurno fora de casa. Férias de 13 dias a partir do dia 231 do ano, com atividade doméstica reduzida. Equipamento: 12 módulos de 420 Wp (5.04 kWp), inversor 4.58 kW; bateria fixa de 5.12 kWh / 2.56 kW. Ligação monofásica, 10.35 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: adere em 2023-02-13 12:00:00+00:00. Personagens e instalações são fictícias.

## Member_014

João Alves · agregado 014: 2 pessoa(s), moradia de 120 m²; teletrabalho às terças e quintas e deslocações nos restantes dias úteis. Férias de 11 dias a partir do dia 188 do ano, com atividade doméstica reduzida. Equipamento: 8 módulos de 420 Wp (3.36 kWp), inversor 3.06 kW; bateria fixa de 5.12 kWh / 2.56 kW; ciclos de lavagem programáveis, com janela alargada. Ligação monofásica, 5.75 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: retira definitivamente DEF-M014-01 em 2023-08-15 13:00:00+01:00; pedidos pendentes desse equipamento são cancelados. Personagens e instalações são fictícias.

## Member_015

Inês Almeida · agregado 015: 4 pessoa(s), moradia de 133 m²; turnos noturnos em blocos de quatro noites, com regresso de manhã. Férias de 18 dias a partir do dia 226 do ano, com atividade doméstica reduzida. Equipamento: ciclos de lavagem programáveis, com janela curta. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_016

Miguel Ribeiro · agregado 016: 4 pessoa(s), apartamento de 169 m²; rotinas de levar e buscar crianças, com regressos intermédios a casa. Pelo menos uma das pessoas é uma criança em idade escolar. Férias de 10 dias a partir do dia 200 do ano, com atividade doméstica reduzida. Equipamento: 2 carregador(es), 2 lugares de estacionamento no cenário. Ligação trifásica, 13.80 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M016-01 (40 kWh úteis, AC até 11.04 kW, preferência reserve_80); EV-M016-02 (64 kWh úteis, AC até 11.04 kW, preferência reserve_80). Os pedidos variam por sessão, entre 50% e 100%. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_017

Sofia Pinto · agregado 017: 1 pessoa(s), moradia de 56 m²; presença diurna e deslocações curtas sem horário laboral fixo. Férias de 14 dias a partir do dia 191 do ano, com atividade doméstica reduzida. Equipamento: 5 módulos de 420 Wp (2.10 kWp), inversor 1.91 kW. Ligação monofásica, 5.75 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: adere em 2023-02-14 00:00:00+00:00. Personagens e instalações são fictícias.

## Member_018

Pedro Carvalho · agregado 018: 2 pessoa(s), moradia de 77 m²; trabalho até tarde e consumo doméstico deslocado para o fim da noite. Férias de 7 dias a partir do dia 200 do ano, com atividade doméstica reduzida. Equipamento: 11 módulos de 420 Wp (4.62 kWp), inversor 4.20 kW; bateria fixa de 10.24 kWh / 5.00 kW; 1 carregador(es), 2 lugares de estacionamento no cenário; ciclos de lavagem programáveis, com janela curta. Ligação monofásica, 10.35 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M018-01 (77 kWh úteis, AC até 11.04 kW, preferência high_90). Os pedidos variam por sessão, entre 50% e 100%. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_019

Marta Teixeira · agregado 019: 3 pessoa(s), apartamento de 136 m²; trabalho diurno fora de casa. Férias de 15 dias a partir do dia 214 do ano, com atividade doméstica reduzida. Equipamento: ciclos de lavagem programáveis, com janela intermédia. Ligação monofásica, 5.75 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: retira definitivamente DEF-M019-01 em 2023-02-17 00:00:00+00:00; pedidos pendentes desse equipamento são cancelados; retira definitivamente DEF-M019-02 em 2023-02-17 00:00:00+00:00; pedidos pendentes desse equipamento são cancelados; sai definitivamente do cenário, com todos os seus ativos, em 2023-02-17 00:00:00+00:00; depois desta data não há consumo, produção, ciclos, viagens ou carregamento atribuídos ao membro. Personagens e instalações são fictícias.

## Member_020

Rui Moreira · agregado 020: 2 pessoa(s), apartamento de 132 m²; teletrabalho às terças e quintas e deslocações nos restantes dias úteis. Férias de 9 dias a partir do dia 190 do ano, com atividade doméstica reduzida. Equipamento: 1 carregador(es), 2 lugares de estacionamento no cenário. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M020-01 (60 kWh úteis, AC até 11.04 kW, preferência balanced_70). Os pedidos variam por sessão, entre 50% e 100%. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_021

Leonor Silva · agregado 021: 2 pessoa(s), moradia de 71 m²; turnos noturnos em blocos de quatro noites, com regresso de manhã. Férias de 13 dias a partir do dia 242 do ano, com atividade doméstica reduzida. Equipamento: 8 módulos de 420 Wp (3.36 kWp), inversor 3.06 kW. Ligação monofásica, 5.75 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: adere em 2023-02-14 12:00:00+00:00. Personagens e instalações são fictícias.

## Member_022

Diogo Santos · agregado 022: 4 pessoa(s), moradia de 104 m²; rotinas de levar e buscar crianças, com regressos intermédios a casa. Pelo menos uma das pessoas é uma criança em idade escolar. Férias de 10 dias a partir do dia 201 do ano, com atividade doméstica reduzida. Equipamento: 15 módulos de 420 Wp (6.30 kWp), inversor 5.73 kW; bateria fixa de 10.24 kWh / 5.00 kW; 1 carregador(es), 2 lugares de estacionamento no cenário; ciclos de lavagem programáveis, com janela intermédia. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M022-01 (50 kWh úteis, AC até 11.04 kW, preferência reserve_80). Os pedidos variam por sessão, entre 50% e 100%. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_023

Teresa Ferreira · agregado 023: 2 pessoa(s), apartamento de 120 m²; presença diurna e deslocações curtas sem horário laboral fixo. Férias de 17 dias a partir do dia 208 do ano, com atividade doméstica reduzida. Equipamento: ciclos de lavagem programáveis, com janela alargada. Ligação monofásica, 5.75 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: retira definitivamente DEF-M023-01 em 2023-05-16 07:00:00+01:00; pedidos pendentes desse equipamento são cancelados; retira definitivamente DEF-M023-02 em 2023-05-16 07:00:00+01:00; pedidos pendentes desse equipamento são cancelados; sai definitivamente do cenário, com todos os seus ativos, em 2023-05-16 07:00:00+01:00; depois desta data não há consumo, produção, ciclos, viagens ou carregamento atribuídos ao membro. Personagens e instalações são fictícias.

## Member_024

António Pereira · agregado 024: 3 pessoa(s), moradia de 83 m²; trabalho até tarde e consumo doméstico deslocado para o fim da noite. Férias de 11 dias a partir do dia 187 do ano, com atividade doméstica reduzida. Equipamento: 1 carregador(es), 2 lugares de estacionamento no cenário. Ligação monofásica, 10.35 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M024-01 (105 kWh úteis, AC até 11.04 kW, preferência needs_50). Os pedidos variam por sessão, entre 50% e 100%. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_025

Ana Costa · agregado 025: 5 pessoa(s), apartamento de 96 m²; trabalho diurno fora de casa. Férias de 16 dias a partir do dia 213 do ano, com atividade doméstica reduzida. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: adere em 2023-08-14 01:00:00+01:00. Personagens e instalações são fictícias.

## Member_026

João Rodrigues · agregado 026: 2 pessoa(s), moradia de 120 m²; teletrabalho às terças e quintas e deslocações nos restantes dias úteis. Férias de 8 dias a partir do dia 248 do ano, com atividade doméstica reduzida. Equipamento: 12 módulos de 420 Wp (5.04 kWp), inversor 4.58 kW; bateria fixa de 5.12 kWh / 2.56 kW; 1 carregador(es), 2 lugares de estacionamento no cenário; ciclos de lavagem programáveis, com janela alargada. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M026-01 (60 kWh úteis, AC até 11.04 kW, preferência reserve_80). Os pedidos variam por sessão, entre 50% e 100%. O mesmo automóvel pode carregar em casa e em Member_081, com itinerários descritos nos dados de origem; no motor original, o SOC é prescrito em cada chegada e a energia das viagens não é contabilizada longitudinalmente. No cenário dinâmico corrente: adere em 2023-02-16 00:00:00+00:00. Personagens e instalações são fictícias.

## Member_027

Inês Martins · agregado 027: 2 pessoa(s), moradia de 87 m²; turnos noturnos em blocos de quatro noites, com regresso de manhã. Férias de 18 dias a partir do dia 247 do ano, com atividade doméstica reduzida. Equipamento: ciclos de lavagem programáveis, com janela curta. Ligação monofásica, 5.75 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_028

Miguel Sousa · agregado 028: 3 pessoa(s), apartamento de 86 m²; rotinas de levar e buscar crianças, com regressos intermédios a casa. Pelo menos uma das pessoas é uma criança em idade escolar. Férias de 11 dias a partir do dia 223 do ano, com atividade doméstica reduzida. Equipamento: 1 carregador(es), 2 lugares de estacionamento no cenário. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M028-01 (60 kWh úteis, AC até 22.08 kW, preferência high_90). Os pedidos variam por sessão, entre 50% e 100%. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_029

Sofia Fernandes · agregado 029: 2 pessoa(s), moradia de 114 m²; presença diurna e deslocações curtas sem horário laboral fixo. Férias de 18 dias a partir do dia 248 do ano, com atividade doméstica reduzida. Equipamento: 6 módulos de 420 Wp (2.52 kWp), inversor 2.29 kW. Ligação monofásica, 4.60 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_030

Pedro Gonçalves · agregado 030: 2 pessoa(s), moradia de 89 m²; trabalho até tarde e consumo doméstico deslocado para o fim da noite. Férias de 13 dias a partir do dia 234 do ano, com atividade doméstica reduzida. Equipamento: 13 módulos de 420 Wp (5.46 kWp), inversor 4.96 kW; bateria fixa de 10.24 kWh / 5.00 kW; 1 carregador(es), 2 lugares de estacionamento no cenário; ciclos de lavagem programáveis, com janela curta. Ligação trifásica, 17.25 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M030-01 (50 kWh úteis, AC até 11.04 kW, preferência balanced_70). Os pedidos variam por sessão, entre 50% e 100%. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_031

Marta Gomes · agregado 031: 1 pessoa(s), apartamento de 101 m²; trabalho diurno fora de casa. Férias de 17 dias a partir do dia 220 do ano, com atividade doméstica reduzida. Equipamento: ciclos de lavagem programáveis, com janela intermédia. Ligação monofásica, 4.60 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_032

Rui Lopes · agregado 032: 2 pessoa(s), apartamento de 107 m²; teletrabalho às terças e quintas e deslocações nos restantes dias úteis. Férias de 10 dias a partir do dia 218 do ano, com atividade doméstica reduzida. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação monofásica, 4.60 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_033

Leonor Marques · agregado 033: 3 pessoa(s), moradia de 134 m²; turnos noturnos em blocos de quatro noites, com regresso de manhã. Férias de 17 dias a partir do dia 217 do ano, com atividade doméstica reduzida. Equipamento: 11 módulos de 420 Wp (4.62 kWp), inversor 4.20 kW. Ligação monofásica, 4.60 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: adere em 2023-02-16 12:00:00+00:00. Personagens e instalações são fictícias.

## Member_034

Diogo Alves · agregado 034: 1 pessoa(s), moradia de 87 m²; rotinas de levar e buscar crianças, com regressos intermédios a casa. O residente acompanha netos que vivem noutro agregado. Férias de 15 dias a partir do dia 216 do ano, com atividade doméstica reduzida. Equipamento: 7 módulos de 420 Wp (2.94 kWp), inversor 2.67 kW; 1 carregador(es), 2 lugares de estacionamento no cenário; ciclos de lavagem programáveis, com janela intermédia. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M034-01 (64 kWh úteis, AC até 11.04 kW, preferência needs_50). Os pedidos variam por sessão, entre 50% e 100%. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_035

Teresa Almeida · agregado 035: 3 pessoa(s), apartamento de 76 m²; presença diurna e deslocações curtas sem horário laboral fixo. Férias de 8 dias a partir do dia 234 do ano, com atividade doméstica reduzida. Equipamento: ciclos de lavagem programáveis, com janela alargada. Ligação monofásica, 5.75 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_036

António Ribeiro · agregado 036: 1 pessoa(s), moradia de 79 m²; trabalho até tarde e consumo doméstico deslocado para o fim da noite. Férias de 7 dias a partir do dia 209 do ano, com atividade doméstica reduzida. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação monofásica, 4.60 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_037

Ana Pinto · agregado 037: 2 pessoa(s), apartamento de 103 m²; trabalho diurno fora de casa. Férias de 7 dias a partir do dia 232 do ano, com atividade doméstica reduzida. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: sai definitivamente do cenário, com todos os seus ativos, em 2023-05-19 07:00:00+01:00; depois desta data não há consumo, produção, ciclos, viagens ou carregamento atribuídos ao membro. Personagens e instalações são fictícias.

## Member_038

João Carvalho · agregado 038: 1 pessoa(s), moradia de 55 m²; teletrabalho às terças e quintas e deslocações nos restantes dias úteis. Férias de 18 dias a partir do dia 210 do ano, com atividade doméstica reduzida. Equipamento: 8 módulos de 420 Wp (3.36 kWp), inversor 3.06 kW; bateria fixa de 5.12 kWh / 2.56 kW; 1 carregador(es), 2 lugares de estacionamento no cenário; ciclos de lavagem programáveis, com janela alargada. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M038-01 (95 kWh úteis, AC até 11.04 kW, preferência high_90). Os pedidos variam por sessão, entre 50% e 100%. O mesmo automóvel pode carregar em casa e em Member_081, com itinerários descritos nos dados de origem; no motor original, o SOC é prescrito em cada chegada e a energia das viagens não é contabilizada longitudinalmente. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_039

Inês Teixeira · agregado 039: 4 pessoa(s), moradia de 123 m²; turnos noturnos em blocos de quatro noites, com regresso de manhã. Férias de 14 dias a partir do dia 193 do ano, com atividade doméstica reduzida. Equipamento: ciclos de lavagem programáveis, com janela curta. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_040

Miguel Moreira · agregado 040: 3 pessoa(s), apartamento de 141 m²; rotinas de levar e buscar crianças, com regressos intermédios a casa. Pelo menos uma das pessoas é uma criança em idade escolar. Férias de 15 dias a partir do dia 210 do ano, com atividade doméstica reduzida. Equipamento: 2 carregador(es), 2 lugares de estacionamento no cenário. Ligação trifásica, 13.80 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M040-01 (64 kWh úteis, AC até 11.04 kW, preferência balanced_70); EV-M040-02 (60 kWh úteis, AC até 11.04 kW, preferência reserve_80). Os pedidos variam por sessão, entre 50% e 100%. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_041

Sofia Silva · agregado 041: 3 pessoa(s), moradia de 154 m²; presença diurna e deslocações curtas sem horário laboral fixo. Férias de 7 dias a partir do dia 247 do ano, com atividade doméstica reduzida. Equipamento: 8 módulos de 420 Wp (3.36 kWp), inversor 3.06 kW. Ligação monofásica, 5.75 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: adere em 2023-05-14 19:00:00+01:00. Personagens e instalações são fictícias.

## Member_042

Pedro Santos · agregado 042: 3 pessoa(s), moradia de 97 m²; trabalho até tarde e consumo doméstico deslocado para o fim da noite. Férias de 16 dias a partir do dia 237 do ano, com atividade doméstica reduzida. Equipamento: 13 módulos de 420 Wp (5.46 kWp), inversor 4.96 kW; 1 carregador(es), 2 lugares de estacionamento no cenário; ciclos de lavagem programáveis, com janela curta. Ligação monofásica, 10.35 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M042-01 (60 kWh úteis, AC até 22.08 kW, preferência reserve_80). Os pedidos variam por sessão, entre 50% e 100%. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_043

Marta Ferreira · agregado 043: 2 pessoa(s), apartamento de 119 m²; trabalho diurno fora de casa. Férias de 7 dias a partir do dia 219 do ano, com atividade doméstica reduzida. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: adere em 2023-08-15 01:00:00+01:00. Personagens e instalações são fictícias.

## Member_044

Rui Pereira · agregado 044: 3 pessoa(s), apartamento de 119 m²; teletrabalho às terças e quintas e deslocações nos restantes dias úteis. Férias de 12 dias a partir do dia 213 do ano, com atividade doméstica reduzida. Equipamento: 1 carregador(es), 2 lugares de estacionamento no cenário. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M044-01 (82 kWh úteis, AC até 11.04 kW, preferência needs_50). Os pedidos variam por sessão, entre 50% e 100%. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_045

Leonor Costa · agregado 045: 4 pessoa(s), moradia de 119 m²; turnos noturnos em blocos de quatro noites, com regresso de manhã. Férias de 12 dias a partir do dia 198 do ano, com atividade doméstica reduzida. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: adere em 2023-08-16 01:00:00+01:00. Personagens e instalações são fictícias.

## Member_046

Diogo Rodrigues · agregado 046: 2 pessoa(s), moradia de 76 m²; rotinas de levar e buscar crianças, com regressos intermédios a casa. Pelo menos uma das pessoas é uma criança em idade escolar. Férias de 17 dias a partir do dia 225 do ano, com atividade doméstica reduzida. Equipamento: 9 módulos de 420 Wp (3.78 kWp), inversor 3.44 kW; bateria fixa de 5.12 kWh / 2.56 kW; 1 carregador(es), 2 lugares de estacionamento no cenário; ciclos de lavagem programáveis, com janela intermédia. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M046-01 (60 kWh úteis, AC até 11.04 kW, preferência reserve_80). Os pedidos variam por sessão, entre 50% e 100%. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_047

Teresa Martins · agregado 047: 2 pessoa(s), apartamento de 132 m²; presença diurna e deslocações curtas sem horário laboral fixo. Férias de 9 dias a partir do dia 226 do ano, com atividade doméstica reduzida. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: adere em 2023-08-17 01:00:00+01:00. Personagens e instalações são fictícias.

## Member_048

António Sousa · agregado 048: 4 pessoa(s), moradia de 97 m²; trabalho até tarde e consumo doméstico deslocado para o fim da noite. Férias de 14 dias a partir do dia 190 do ano, com atividade doméstica reduzida. Equipamento: 1 carregador(es), 2 lugares de estacionamento no cenário. Ligação monofásica, 10.35 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M048-01 (40 kWh úteis, AC até 11.04 kW, preferência high_90). Os pedidos variam por sessão, entre 50% e 100%. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_049

Ana Fernandes · agregado 049: 1 pessoa(s), apartamento de 76 m²; trabalho diurno fora de casa. Férias de 12 dias a partir do dia 190 do ano, com atividade doméstica reduzida. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_050

João Gonçalves · agregado 050: 4 pessoa(s), moradia de 128 m²; teletrabalho às terças e quintas e deslocações nos restantes dias úteis. Férias de 10 dias a partir do dia 240 do ano, com atividade doméstica reduzida. Equipamento: 14 módulos de 420 Wp (5.88 kWp), inversor 5.34 kW; bateria fixa de 10.24 kWh / 5.00 kW; 1 carregador(es), 2 lugares de estacionamento no cenário; ciclos de lavagem programáveis, com janela alargada. Ligação trifásica, 17.25 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M050-01 (40 kWh úteis, AC até 11.04 kW, preferência balanced_70). Os pedidos variam por sessão, entre 50% e 100%. O mesmo automóvel pode carregar em casa e em Member_081, com itinerários descritos nos dados de origem; no motor original, o SOC é prescrito em cada chegada e a energia das viagens não é contabilizada longitudinalmente. No cenário dinâmico corrente: instala BESS-M050-01 em 2023-05-15 07:00:00+01:00, com a identidade e o estado inicial declarados. Personagens e instalações são fictícias.

## Member_051

Inês Gomes · agregado 051: 1 pessoa(s), moradia de 96 m²; turnos noturnos em blocos de quatro noites, com regresso de manhã. Férias de 9 dias a partir do dia 200 do ano, com atividade doméstica reduzida. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação monofásica, 5.75 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: sai definitivamente do cenário, com todos os seus ativos, em 2023-08-16 13:00:00+01:00; depois desta data não há consumo, produção, ciclos, viagens ou carregamento atribuídos ao membro. Personagens e instalações são fictícias.

## Member_052

Miguel Lopes · agregado 052: 3 pessoa(s), apartamento de 141 m²; rotinas de levar e buscar crianças, com regressos intermédios a casa. Pelo menos uma das pessoas é uma criança em idade escolar. Férias de 12 dias a partir do dia 199 do ano, com atividade doméstica reduzida. Equipamento: 1 carregador(es), 2 lugares de estacionamento no cenário. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M052-01 (64 kWh úteis, AC até 11.04 kW, preferência reserve_80). Os pedidos variam por sessão, entre 50% e 100%. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_053

Sofia Marques · agregado 053: 2 pessoa(s), moradia de 129 m²; presença diurna e deslocações curtas sem horário laboral fixo. Férias de 18 dias a partir do dia 236 do ano, com atividade doméstica reduzida. Equipamento: 6 módulos de 420 Wp (2.52 kWp), inversor 2.29 kW. Ligação monofásica, 5.75 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: adere em 2023-05-15 19:00:00+01:00. Personagens e instalações são fictícias.

## Member_054

Pedro Alves · agregado 054: 2 pessoa(s), moradia de 101 m²; trabalho até tarde e consumo doméstico deslocado para o fim da noite. Férias de 7 dias a partir do dia 187 do ano, com atividade doméstica reduzida. Equipamento: 14 módulos de 420 Wp (5.88 kWp), inversor 5.34 kW; 1 carregador(es), 2 lugares de estacionamento no cenário; ciclos de lavagem programáveis, com janela curta. Ligação monofásica, 10.35 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M054-01 (82 kWh úteis, AC até 11.04 kW, preferência needs_50). Os pedidos variam por sessão, entre 50% e 100%. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_055

Marta Almeida · agregado 055: 4 pessoa(s), apartamento de 179 m²; trabalho diurno fora de casa. Férias de 7 dias a partir do dia 248 do ano, com atividade doméstica reduzida. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação monofásica, 10.35 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: adere em 2023-08-17 13:00:00+01:00. Personagens e instalações são fictícias.

## Member_056

Rui Ribeiro · agregado 056: 4 pessoa(s), apartamento de 139 m²; teletrabalho às terças e quintas e deslocações nos restantes dias úteis. Férias de 12 dias a partir do dia 202 do ano, com atividade doméstica reduzida. Equipamento: 2 carregador(es), 2 lugares de estacionamento no cenário. Ligação trifásica, 13.80 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M056-01 (50 kWh úteis, AC até 22.08 kW, preferência reserve_80); EV-M056-02 (60 kWh úteis, AC até 22.08 kW, preferência reserve_80). Os pedidos variam por sessão, entre 50% e 100%. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_057

Leonor Pinto · agregado 057: 1 pessoa(s), moradia de 74 m²; turnos noturnos em blocos de quatro noites, com regresso de manhã. Férias de 18 dias a partir do dia 229 do ano, com atividade doméstica reduzida. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação monofásica, 5.75 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: sai definitivamente do cenário, com todos os seus ativos, em 2023-11-13 06:00:00+00:00; depois desta data não há consumo, produção, ciclos, viagens ou carregamento atribuídos ao membro. Personagens e instalações são fictícias.

## Member_058

Diogo Carvalho · agregado 058: 3 pessoa(s), moradia de 75 m²; rotinas de levar e buscar crianças, com regressos intermédios a casa. Pelo menos uma das pessoas é uma criança em idade escolar. Férias de 17 dias a partir do dia 187 do ano, com atividade doméstica reduzida. Equipamento: 7 módulos de 420 Wp (2.94 kWp), inversor 2.67 kW; ciclos de lavagem programáveis, com janela intermédia. Ligação monofásica, 5.75 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: adere em 2023-05-17 07:00:00+01:00. Personagens e instalações são fictícias.

## Member_059

Teresa Teixeira · agregado 059: 5 pessoa(s), apartamento de 134 m²; presença diurna e deslocações curtas sem horário laboral fixo. Férias de 16 dias a partir do dia 239 do ano, com atividade doméstica reduzida. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: adere em 2023-08-18 13:00:00+01:00. Personagens e instalações são fictícias.

## Member_060

António Moreira · agregado 060: 2 pessoa(s), moradia de 106 m²; trabalho até tarde e consumo doméstico deslocado para o fim da noite. Férias de 7 dias a partir do dia 211 do ano, com atividade doméstica reduzida. Equipamento: 1 carregador(es), 2 lugares de estacionamento no cenário. Ligação monofásica, 10.35 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M060-01 (50 kWh úteis, AC até 11.04 kW, preferência balanced_70). Os pedidos variam por sessão, entre 50% e 100%. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_061

Ana Silva · agregado 061: 1 pessoa(s), apartamento de 78 m²; trabalho diurno fora de casa. Férias de 8 dias a partir do dia 204 do ano, com atividade doméstica reduzida. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação monofásica, 5.75 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: adere em 2023-11-13 18:00:00+00:00. Personagens e instalações são fictícias.

## Member_062

João Santos · agregado 062: 1 pessoa(s), moradia de 100 m²; teletrabalho às terças e quintas e deslocações nos restantes dias úteis. Férias de 7 dias a partir do dia 210 do ano, com atividade doméstica reduzida. Equipamento: 10 módulos de 420 Wp (4.20 kWp), inversor 3.82 kW; bateria fixa de 5.12 kWh / 2.56 kW; 1 carregador(es), 2 lugares de estacionamento no cenário; ciclos de lavagem programáveis, com janela alargada. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M062-01 (50 kWh úteis, AC até 11.04 kW, preferência reserve_80). Os pedidos variam por sessão, entre 50% e 100%. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_063

Inês Ferreira · agregado 063: 1 pessoa(s), moradia de 73 m²; turnos noturnos em blocos de quatro noites, com regresso de manhã. Férias de 15 dias a partir do dia 229 do ano, com atividade doméstica reduzida. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: sai definitivamente do cenário, com todos os seus ativos, em 2023-11-14 06:00:00+00:00; depois desta data não há consumo, produção, ciclos, viagens ou carregamento atribuídos ao membro. Personagens e instalações são fictícias.

## Member_064

Miguel Pereira · agregado 064: 1 pessoa(s), apartamento de 97 m²; rotinas de levar e buscar crianças, com regressos intermédios a casa. O residente acompanha netos que vivem noutro agregado. Férias de 8 dias a partir do dia 227 do ano, com atividade doméstica reduzida. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação monofásica, 4.60 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_065

Sofia Costa · agregado 065: 2 pessoa(s), apartamento de 126 m²; presença diurna e deslocações curtas sem horário laboral fixo. Férias de 13 dias a partir do dia 224 do ano, com atividade doméstica reduzida. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: adere em 2023-11-14 18:00:00+00:00. Personagens e instalações são fictícias.

## Member_066

Pedro Rodrigues · agregado 066: 2 pessoa(s), moradia de 132 m²; trabalho até tarde e consumo doméstico deslocado para o fim da noite. Férias de 18 dias a partir do dia 196 do ano, com atividade doméstica reduzida. Equipamento: ciclos de lavagem programáveis, com janela curta. Ligação monofásica, 5.75 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_067

Marta Martins · agregado 067: 4 pessoa(s), apartamento de 123 m²; trabalho diurno fora de casa. Férias de 8 dias a partir do dia 195 do ano, com atividade doméstica reduzida. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: adere em 2023-11-15 06:00:00+00:00. Personagens e instalações são fictícias.

## Member_068

Rui Sousa · agregado 068: 4 pessoa(s), apartamento de 143 m²; teletrabalho às terças e quintas e deslocações nos restantes dias úteis. Férias de 9 dias a partir do dia 206 do ano, com atividade doméstica reduzida. Equipamento: 1 carregador(es), 2 lugares de estacionamento no cenário. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M068-01 (40 kWh úteis, AC até 11.04 kW, preferência high_90). Os pedidos variam por sessão, entre 50% e 100%. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_069

Leonor Fernandes · agregado 069: 1 pessoa(s), moradia de 94 m²; turnos noturnos em blocos de quatro noites, com regresso de manhã. Férias de 14 dias a partir do dia 231 do ano, com atividade doméstica reduzida. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: sai definitivamente do cenário, com todos os seus ativos, em 2023-11-15 18:00:00+00:00; depois desta data não há consumo, produção, ciclos, viagens ou carregamento atribuídos ao membro. Personagens e instalações são fictícias.

## Member_070

Diogo Gonçalves · agregado 070: 3 pessoa(s), moradia de 144 m²; rotinas de levar e buscar crianças, com regressos intermédios a casa. Pelo menos uma das pessoas é uma criança em idade escolar. Férias de 7 dias a partir do dia 232 do ano, com atividade doméstica reduzida. Equipamento: 12 módulos de 420 Wp (5.04 kWp), inversor 4.58 kW; bateria fixa de 10.24 kWh / 5.00 kW; 1 carregador(es), 2 lugares de estacionamento no cenário; ciclos de lavagem programáveis, com janela intermédia. Ligação trifásica, 17.25 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M070-01 (77 kWh úteis, AC até 22.08 kW, preferência balanced_70). Os pedidos variam por sessão, entre 50% e 100%. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_071

Teresa Gomes · agregado 071: 3 pessoa(s), apartamento de 98 m²; presença diurna e deslocações curtas sem horário laboral fixo. Férias de 16 dias a partir do dia 211 do ano, com atividade doméstica reduzida. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: adere em 2023-02-12 12:00:00+00:00. Personagens e instalações são fictícias.

## Member_072

António Lopes · agregado 072: 2 pessoa(s), moradia de 98 m²; trabalho até tarde e consumo doméstico deslocado para o fim da noite. Férias de 11 dias a partir do dia 212 do ano, com atividade doméstica reduzida. Equipamento: 1 carregador(es), 2 lugares de estacionamento no cenário. Ligação monofásica, 10.35 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M072-01 (64 kWh úteis, AC até 11.04 kW, preferência reserve_80). Os pedidos variam por sessão, entre 50% e 100%. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_073

Ana Marques · agregado 073: 2 pessoa(s), apartamento de 94 m²; trabalho diurno fora de casa. Férias de 7 dias a partir do dia 189 do ano, com atividade doméstica reduzida. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: adere em 2023-11-16 06:00:00+00:00. Personagens e instalações são fictícias.

## Member_074

João Alves · agregado 074: 3 pessoa(s), apartamento de 99 m²; teletrabalho às terças e quintas e deslocações nos restantes dias úteis. Férias de 17 dias a partir do dia 200 do ano, com atividade doméstica reduzida. Equipamento: ciclos de lavagem programáveis, com janela alargada. Ligação monofásica, 5.75 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_075

Inês Almeida · agregado 075: 2 pessoa(s), moradia de 82 m²; turnos noturnos em blocos de quatro noites, com regresso de manhã. Férias de 7 dias a partir do dia 248 do ano, com atividade doméstica reduzida. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_076

Miguel Ribeiro · agregado 076: 2 pessoa(s), apartamento de 88 m²; rotinas de levar e buscar crianças, com regressos intermédios a casa. Pelo menos uma das pessoas é uma criança em idade escolar. Férias de 12 dias a partir do dia 245 do ano, com atividade doméstica reduzida. Equipamento: 1 carregador(es), 2 lugares de estacionamento no cenário. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M076-01 (60 kWh úteis, AC até 11.04 kW, preferência reserve_80). Os pedidos variam por sessão, entre 50% e 100%. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_077

Sofia Pinto · agregado 077: 2 pessoa(s), apartamento de 110 m²; presença diurna e deslocações curtas sem horário laboral fixo. Férias de 13 dias a partir do dia 223 do ano, com atividade doméstica reduzida. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação monofásica, 5.75 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_078

Pedro Carvalho · agregado 078: 1 pessoa(s), moradia de 84 m²; trabalho até tarde e consumo doméstico deslocado para o fim da noite. Férias de 15 dias a partir do dia 208 do ano, com atividade doméstica reduzida. Equipamento: 6 módulos de 420 Wp (2.52 kWp), inversor 2.29 kW; ciclos de lavagem programáveis, com janela curta. Ligação monofásica, 5.75 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: adere em 2023-05-18 19:00:00+01:00. Personagens e instalações são fictícias.

## Member_079

Marta Teixeira · agregado 079: 4 pessoa(s), apartamento de 125 m²; trabalho diurno fora de casa. Férias de 14 dias a partir do dia 211 do ano, com atividade doméstica reduzida. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_080

Rui Moreira · agregado 080: 4 pessoa(s), apartamento de 116 m²; teletrabalho às terças e quintas e deslocações nos restantes dias úteis. Férias de 17 dias a partir do dia 249 do ano, com atividade doméstica reduzida. Equipamento: 1 carregador(es), 2 lugares de estacionamento no cenário. Ligação monofásica, 6.90 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. Automóveis residentes: EV-M080-01 (64 kWh úteis, AC até 11.04 kW, preferência balanced_70). Os pedidos variam por sessão, entre 50% e 100%. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_081

Office 081: estabelecimento de 650 m², 2.ª-6.ª 08-19h; fecha fins de semana, feriados e férias; os consumos elétricos incluem usos de serviço, sem pressupor equivalência com uma habitação. Equipamento: 80 módulos de 420 Wp (33.60 kWp), inversor 30.55 kW; bateria fixa de 10.24 kWh / 5.12 kW; 12 carregador(es), 36 lugares de estacionamento no cenário; ciclos de lavagem programáveis, com janela curta. Ligação trifásica, 82.80 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_082

Retail 082: estabelecimento de 241 m², 2.ª-6.ª 09-20h; sábado 09:30-19h; domingos/feriados 10-17h, exceto Ano Novo/Natal; os consumos elétricos incluem usos de serviço, sem pressupor equivalência com uma habitação. Equipamento: 23 módulos de 420 Wp (9.66 kWp), inversor 8.78 kW; bateria fixa de 10.24 kWh / 5.12 kW. Ligação trifásica, 20.70 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_083

Clinic 083: estabelecimento de 144 m², 2.ª-6.ª 08-20h; sábado 09-13h; fecha domingos e feriados; os consumos elétricos incluem usos de serviço, sem pressupor equivalência com uma habitação. Equipamento: ciclos de lavagem programáveis, com janela alargada. Ligação trifásica, 20.70 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_084

Cafe 084: estabelecimento de 103 m², 2.ª-sábado 07-22h; domingos/feriados 08-15h; fecha Ano Novo/Natal; os consumos elétricos incluem usos de serviço, sem pressupor equivalência com uma habitação. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação trifásica, 20.70 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: sai definitivamente do cenário, com todos os seus ativos, em 2023-11-17 06:00:00+00:00; depois desta data não há consumo, produção, ciclos, viagens ou carregamento atribuídos ao membro. Personagens e instalações são fictícias.

## Member_085

Workshop 085: estabelecimento de 103 m², 2.ª-6.ª 08-18h; sábado 09-13h; fecha domingos, feriados e férias; os consumos elétricos incluem usos de serviço, sem pressupor equivalência com uma habitação. Equipamento: 14 módulos de 420 Wp (5.88 kWp), inversor 5.34 kW; 1 carregador(es), 5 lugares de estacionamento no cenário; ciclos de lavagem programáveis, com janela intermédia. Ligação trifásica, 20.70 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: retira definitivamente CHG-M085-01 em 2023-08-18 01:00:00+01:00; pedidos pendentes desse equipamento são cancelados. Personagens e instalações são fictícias.

## Member_086

Office 086: estabelecimento de 230 m², 2.ª-6.ª 08-19h; fecha fins de semana, feriados e férias; os consumos elétricos incluem usos de serviço, sem pressupor equivalência com uma habitação. Equipamento: 29 módulos de 420 Wp (12.18 kWp), inversor 11.07 kW; bateria fixa de 10.24 kWh / 5.12 kW; 4 carregador(es), 12 lugares de estacionamento no cenário. Ligação trifásica, 41.40 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: instala CHG-M086-04 em 2023-02-15 12:00:00+00:00, com a identidade e o estado inicial declarados. Personagens e instalações são fictícias.

## Member_087

Retail 087: estabelecimento de 193 m², 2.ª-6.ª 09-20h; sábado 09:30-19h; fecha domingos e feriados; os consumos elétricos incluem usos de serviço, sem pressupor equivalência com uma habitação. Equipamento: 2 carregador(es), 10 lugares de estacionamento no cenário. Ligação trifásica, 20.70 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_088

Clinic 088: estabelecimento de 117 m², 2.ª-6.ª 08-20h; sábado 09-13h; fecha domingos e feriados; os consumos elétricos incluem usos de serviço, sem pressupor equivalência com uma habitação. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação trifásica, 20.70 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_089

Cafe 089: estabelecimento de 181 m², 2.ª-sábado 07-22h; domingos/feriados 08-15h; fecha Ano Novo/Natal; os consumos elétricos incluem usos de serviço, sem pressupor equivalência com uma habitação. Equipamento: 26 módulos de 420 Wp (10.92 kWp), inversor 9.93 kW; 1 carregador(es), 10 lugares de estacionamento no cenário; ciclos de lavagem programáveis, com janela alargada. Ligação trifásica, 20.70 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_090

Workshop 090: estabelecimento de 197 m², 2.ª-6.ª 08-18h; sábado 09-13h; fecha domingos, feriados e férias; os consumos elétricos incluem usos de serviço, sem pressupor equivalência com uma habitação. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação trifásica, 20.70 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_091

Office 091: estabelecimento de 230 m², 2.ª-6.ª 08-19h; fecha fins de semana, feriados e férias; os consumos elétricos incluem usos de serviço, sem pressupor equivalência com uma habitação. Equipamento: 2 carregador(es), 12 lugares de estacionamento no cenário. Ligação trifásica, 20.70 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_092

Retail 092: estabelecimento de 126 m², 2.ª-6.ª 09-20h; sábado 09:30-19h; domingos/feriados 10-17h, exceto Ano Novo/Natal; os consumos elétricos incluem usos de serviço, sem pressupor equivalência com uma habitação. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação trifásica, 20.70 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_093

Clinic 093: estabelecimento de 150 m², 2.ª-6.ª 08-20h; sábado 09-13h; fecha domingos e feriados; os consumos elétricos incluem usos de serviço, sem pressupor equivalência com uma habitação. Equipamento: 16 módulos de 420 Wp (6.72 kWp), inversor 6.11 kW. Ligação trifásica, 20.70 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: retira definitivamente PV-M093-01 em 2023-11-16 18:00:00+00:00; pedidos pendentes desse equipamento são cancelados. Personagens e instalações são fictícias.

## Member_094

Cafe 094: estabelecimento de 260 m², 2.ª-sábado 07-22h; domingos/feriados 08-15h; fecha Ano Novo/Natal; os consumos elétricos incluem usos de serviço, sem pressupor equivalência com uma habitação. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação trifásica, 20.70 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_095

Workshop 095: estabelecimento de 169 m², 2.ª-6.ª 08-18h; sábado 09-13h; fecha domingos, feriados e férias; os consumos elétricos incluem usos de serviço, sem pressupor equivalência com uma habitação. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação trifásica, 20.70 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: sai definitivamente do cenário, com todos os seus ativos, em 2023-11-17 18:00:00+00:00; depois desta data não há consumo, produção, ciclos, viagens ou carregamento atribuídos ao membro. Personagens e instalações são fictícias.

## Member_096

Office 096: estabelecimento de 230 m², 2.ª-6.ª 08-19h; fecha fins de semana, feriados e férias; os consumos elétricos incluem usos de serviço, sem pressupor equivalência com uma habitação. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação trifásica, 20.70 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_097

Retail 097: estabelecimento de 138 m², 2.ª-6.ª 09-20h; sábado 09:30-19h; fecha domingos e feriados; os consumos elétricos incluem usos de serviço, sem pressupor equivalência com uma habitação. Apenas consumo; sem PV, bateria, EV ligado à comunidade ou eletrodomésticos controláveis. Ligação trifásica, 20.70 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_098

Sports Centre 098: estabelecimento de 750 m², dias úteis 07-23h; fim de semana/feriados 08-20h; fecha Ano Novo/Natal; os consumos elétricos incluem usos de serviço, sem pressupor equivalência com uma habitação. Equipamento: 101 módulos de 420 Wp (42.42 kWp), inversor 38.56 kW; bateria fixa de 40.96 kWh / 20.00 kW; 4 carregador(es), 41 lugares de estacionamento no cenário; ciclos de lavagem programáveis, com janela alargada. Ligação trifásica, 41.40 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Member_099

Grocery 099: estabelecimento de 500 m², 2.ª-sábado 08-21h; domingos/feriados 09-19h; fecha Ano Novo/Natal; os consumos elétricos incluem usos de serviço, sem pressupor equivalência com uma habitação. Equipamento: 73 módulos de 420 Wp (30.66 kWp), inversor 27.87 kW; 2 carregador(es), 27 lugares de estacionamento no cenário. Ligação trifásica, 34.50 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: instala PV-M099-01 em 2023-05-18 07:00:00+01:00, com a identidade e o estado inicial declarados. Personagens e instalações são fictícias.

## Member_100

Care Home 100: estabelecimento de 900 m², 24 horas por dia, todos os dias, incluindo feriados; os consumos elétricos incluem usos de serviço, sem pressupor equivalência com uma habitação. Equipamento: 115 módulos de 420 Wp (48.30 kWp), inversor 43.91 kW; bateria fixa de 40.96 kWh / 20.00 kW; ciclos de lavagem programáveis, com janela intermédia. Ligação trifásica, 41.40 kVA, fases e limites explícitos. Tarifa de referência comum a todos os membros: btn_bi, indexada a OMIE 2023–2024. No cenário dinâmico corrente: mantém a adesão durante todo o ano. Personagens e instalações são fictícias.

## Continuação em 2024

As mesmas famílias, sítios e equipamentos continuam em 2024, com novas rotinas e calendário. Nas variantes dinâmicas, quem saiu em 2023 permanece fora, com todos os seus ativos. As datas e a composição executadas seguem os schemas de cada variante; as narrativas e estatísticas anuais de origem descrevem 2023. Ver Extensao_2024.md para os novos totais e a continuidade dos veículos.
