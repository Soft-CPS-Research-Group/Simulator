# Releases

A política, o modelo de notas e o procedimento estão no [guia de publicação](publishing.md).
Os comprovativos históricos usam ligações imutáveis no Git e conservam o âmbito
da execução original; não são uma nova validação deste checkout.

Este é o histórico de versões do fork, com impacto para os utilizadores,
compatibilidade, alterações de contratos e âmbito da validação.

Responsavel default por releases deste fork: [@calofonseca](https://github.com/calofonseca).

## v2.0.1 — 2026-11-04 — seleção de episódios e correções sub-horárias

Responsável: **Tiago Fonseca** ([@calofonseca](https://github.com/calofonseca)).

- Respeita `True`/`False` explícitos em `random_episode_split`; `None` herda o schema.
- Compõe as perdas em repouso a partir da fração horária e da duração física
  do passo, independentemente da cadência dos dados.
- Dimensiona bombas de calor, aquecedores e baterias a partir da potência média
  e calcula picos diários em períodos de 24 horas. Os depósitos térmicos mantêm
  capacidade em energia para um passo de controlo.
- Mantém interfaces de energia, equipamento explícito, lógica BAU, definições
  dos KPIs, datasets e exports. SOC, KPIs derivados e equipamentos dimensionados
  automaticamente podem diferir; comparar versões e configurações equivalentes.
- Simplifica a documentação e usa wheels PyTorch apenas para CPU no CI alojado,
  evitando falta de espaço no runner sem mudar as dependências de execução.

O runtime corrigido passou 1.300 testes de CI, a auditoria física/topológica,
janelas portuguesas de inverno/verão e verificações dos datasets suportados.
O subconjunto local de perdas, dimensionamento, BAU e referências passou 223 testes.
As referências foram revistas contra a fórmula antiga; o caso horário mantém os
resultados. O CI da release e a validação do pacote instalado acompanham a release.
Não foi repetida a campanha de dois anos nem se reivindicam novos ganhos de
desempenho, equivalência anual completa ou calibração empírica nesta atualização.

Pacote/import mantidos: `softcpsrecsimulator` / `citylearn`. O tag e a referência
de dados por omissão são `softcpsrecsimulator-v2.0.1`; os dados mantêm-se iguais
aos da 2.0.0, sem migração. Mantêm-se as opções explícitas de referência de dados.

## v2.0.0 — 2026-11-02 — marco de consolidação do simulador auditado

- Fecha a fase de auditoria, robustez, organização e desempenho, publicando o
  runtime da 1.9.13 com o número 2.0.0. Este marco não introduz alterações
  funcionais nem exige migração.
- Mantém modelos físicos, BAU/NormalPolicy, fórmulas/tolerâncias dos KPIs,
  recompensas, históricos completos, schemas de exportação, dependências e datasets.
- O pacote continua `softcpsrecsimulator`, com imports `citylearn`. O tag do fork
  é `softcpsrecsimulator-v2.0.0`; o tag histórico `v2.0.0` do upstream é preservado.

Validação: os 81 ficheiros Python correspondem byte a byte à 1.9.13, admitindo
apenas a mudança do número em `__init__.py` e do prefixo do tag de dados
em `DataSet.REPOSITORY_TAG`. O carregamento online aponta para o tag do fork,
com os mesmos dados, preservando as opções explícitas por ambiente/construtor
e com teste de regressão. Build/check e exportações instaladas aprovados.
Workflows, datasets, física, BAU, KPIs e otimizações mantidos. Mantém a
validação da 1.9.13: 1.186 testes locais, quatro auditores strict e nove cenários
longos (631.575 transições, 197.347.896 verificações e 77.603 comparações exatas de
KPIs nas sete referências completas). O CI e a publicação desta versão ficam
registados nos anexos da release. A campanha longa é reutilizada, sem repetir
execuções apenas para mudar a numeração. Medições e método na entrada 1.9.13.

Limites mantidos: alvos exatos SOC isolados por diagnosticar, com todos os mínimos
aceitáveis observados cumpridos; campanha longa anterior à v3.3, dados correntes
v3.3 com testes de integração limitados; calibração empírica fora deste âmbito.

## v1.9.13 — 2026-11-02 — menos memória e avaliação mais rápida

- progresso da avaliação por etapa, membro e família de KPIs, sem acrescentar
  tempos aos valores dos indicadores.
- históricos completos do mercado e da robustez em representações compactas
  sem perdas, com consultas por janela e leitura sequencial compatíveis.
- reutilização de populações durante a avaliação e de replays de viabilidade
  EV por partida/contexto, sem partilhar estado físico entre simulações.
- indexação dos ciclos de máquinas diferíveis e reutilização local da
  preparação de observações/previsões. Recompensas personalizadas mantêm o
  contrato completo; o caminho rápido da recompensa predefinida é restrito.
- integração e campanha longa final de regressão e desempenho.

Campanha: nove cenários, 70.175 transições cada, semente 0 e NormalPolicy/BAU,
com verificações físicas ativas. Os valores são observações de execuções separadas
na mesma máquina, não um benchmark repetido e controlado.

| Cenário | Pico de RAM antes → depois (MiB) | Tempo total antes → depois |
|---|---:|---:|
| Micro4 | 920,5 → 691,6 | 7m04s → 4m31s |
| Premium All-in | 13.709,7 → 5.333,4 | 4h23m01s → 2h08m40s |
| Core30 Topology | 3.521,1 → 1.844,1 | 57m57s → 26m51s |
| Core30 Combined | 3.536,2 → 1.849,2 | 52m43s → 27m25s |
| Core30 Safety | 3.821,7 → 1.899,7 | 50m04s → 29m26s |
| Core30 Normal | 3.822,7 → 1.902,2 | 50m49s → 29m30s |
| Core15 | 2.036,3 → 1.158,3 | 24m39s → 14m46s |

Premium All-in: menos 61,1% de pico de RAM, 51,1% de tempo total e 89,7% no fecho
KPIs/exportação (4.667,2 → 479,9 s). A média por transição com auditoria desceu de
157,97 para 102,79 ms. Core30 Robustness terminou em 43m51s, pico 2.045,8 MiB;
Premium Normal em 1h53m07s, pico 5.523,6 MiB. Estes dois não têm referência
anterior completa: não se afirma uma melhoria final histórica para eles.

Validação: 631.575 transições, 197.347.896 verificações físicas/contabilísticas e
77.603 linhas de KPIs exatamente iguais nas sete referências completas, incluindo
valores indefinidos. Reconciliação independente aprovada nos nove cenários;
3.499 inputs preservados. A campanha usou a edição anterior à v3.3; não certifica
uma execução anual dos novos eventos reservados de 2024 agora na principal.
Testes e auditorias da integração corrente ficam identificados separadamente nos
anexos da release. O runtime medido coincide com esta versão exceto o número da
versão; o auditor congelado tinha instrumentação de medição validada em separado.

Integração corrente: 1186 testes locais aprovados (22 avisos conhecidos
Gymnasium float32), quatro auditores strict, lint, desempenho, distribuições e
exportação do wheel fora do código-fonte. Comprovativos nos anexos da release.

Sem amostragem/descarte de informação, alteração dos modelos físicos, BAU,
recompensas, preços, definições/tolerâncias dos KPIs ou schemas de exportação.
Todos os mínimos SOC observados foram cumpridos; mantêm-se alvos exatos de SOC
isolados por diagnosticar. A aprovação contabilística não garante cumprimento de
todos os alvos nem calibração com equipamentos reais.

## v1.9.12 — 2026-11-02 — contratos de robustez e organização interna

- Estabelece contratos de robustez: composição determinística de eventos de robustez, contratos de
  observações/ações/alvos e auditoria versionada disponível na avaliação e exports.
- Partilha a distribuição dos limites elétricos entre NormalPolicy e runtime,
  preservando o orçamento sequencial durante cortes e o carregamento imediato BAU.
- Separa responsabilidades dos KPIs, observações de entidades, carregamento de
  dados e robustez, mantendo as fachadas públicas e os contratos de exportação.
- Acrescenta referências de auditoria verificáveis, fixtures portáteis e regressões
  físicas; remove notas temporárias e comprovativos de trabalho do código-fonte.
- Preserva os datasets Portugal v3.3 e referências de 2024 já integrados na
  principal; atualiza o pacote documental após a limpeza, sem gerar inputs físicos.

Validação histórica do runtime corrigido: 1.020 testes, 16 cenários físicos, 63 interfaces de datasets,
  duas janelas portuguesas de cortes, seis referências instaladas e quatro formatos
  instalados. A integração desta versão é novamente testada sobre a principal;
  comprovativos da execução corrente nos anexos da GitHub Release.
  As otimizações de desempenho são publicadas separadamente na 1.9.13.

Integração da versão: 1063 testes passaram; quatro auditores em modo strict,
lint crítico, desempenho e exportação do wheel instalado aprovados, sem atualizar
as referências esperadas.

Compatibilidade: estratégia BAU, modelos físicos, definições e tolerâncias dos KPIs
mantidos. Cumprir o mínimo de serviço é distinto de atingir o alvo exato de SOC.
Não se afirma calibração com equipamentos reais.

Versões anteriores, com as notas de validação originais: [arquivo de versões](releases_archive.md).
