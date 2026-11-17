# Releases

A política, o modelo de notas e o procedimento estão no [guia de publicação](publishing.md).
Os comprovativos históricos usam ligações imutáveis no Git e conservam o âmbito
da execução original; não são uma nova validação deste checkout.

Este é o histórico de versões do fork, com impacto para os utilizadores,
compatibilidade, alterações de contratos e âmbito da validação.

Responsavel default por releases deste fork: [@calofonseca](https://github.com/calofonseca).

## v2.2.0 — 2026-11-17 — alinhada com o CityLearn 3.5.0

Responsável: **Tiago Fonseca** ([@calofonseca](https://github.com/calofonseca)).

O SoftCPS REC Simulator 2.2.0 tem o mesmo código do simulador, testes e
documentação que o [CityLearn 3.5.0](https://github.com/citylearn-project/CityLearn/releases/tag/v3.5.0).
Diferem apenas o nome da distribuição (`softcpsrecsimulator`), a versão, os
valores por defeito dos downloads, a documentação de releases e os datasets
adicionais do SoftCPS. O import Python continua a ser `citylearn`.

- **Novo BAU; os KPIs relativos ao BAU mudam.** O `BusinessAsUsualAgent` é o BAU
  do CityLearn 3.4.0: decide a partir das observações entity e carrega um EV
  ligado de imediato até 100% por defeito (`immediate-full-charge-v2`). Com
  `ev_charge_target='requested'` (`immediate-user-target-v1`) para no SOC de
  partida pedido pelo utilizador; `business_as_usual_ev_charge_target` escolhe a
  referência usada pelo `evaluate_v2()`. **Todos os KPIs `business_as_usual`,
  `delta_to_business_as_usual` e `ratio_to_business_as_usual` podem diferir da
  2.1.1**, incluindo os dos datasets de Portugal. Repetir o controlador e a
  referência em conjunto.
- **Baselines Basic, Smart e Community.** `RBCBasicPolicy`, `RBCSmartPolicy` e
  `RBCCommunityPolicy` passam a ser as baselines validadas no repositório
  Algorithms do Soft-CPS. `NormalPolicy`, `NormalTargetPolicy`,
  `NormalNoBatteryPolicy`, `ServiceOnlyBaselineAgent`, `GridAwareBaselineAgent` e
  `CommunityAwareBaselineAgent` foram removidos.
- **Margem ilimitada publicada como 1.000.000 kW.** Nas tabelas entity, um
  edifício sem limite de serviço elétrico publica a margem como 1.000.000 kW (o
  limite da observação) em vez de 0 kW, que se lia como "sem capacidade".
- **Datasets de Portugal renomeados para os `pt_rec_*` do CityLearn.** Os
  `pt_rec_4`, `pt_rec_15`, `pt_rec_30` e `pt_rec_100`, idênticos byte a byte aos
  do CityLearn 3.5.0, substituem os `rec_pt_2023_*_v2`, que são removidos. Os
  ficheiros de dados lidos pelos schemas são os mesmos; mudam os nomes dos
  datasets e dos schemas de avaliação. As vistas `preparation_2023`, os catálogos
  das campanhas, as tabelas, os registos de proveniência e os checksums ficam no
  depósito do dataset. Atualizar os nomes e caminhos nas configurações:

| 2.1.1 (removido) | 2.2.0 |
|---|---|
| `rec_pt_2023_micro_4_v2` | `pt_rec_4` |
| `rec_pt_2023_micro_4_v2/schema.json` | `pt_rec_4/schema.json` |
| `rec_pt_2023_micro_4_v2/evaluation/core_4.json` | `pt_rec_4/evaluation/rec_4.json` |
| `rec_pt_2023_core_15_v2` | `pt_rec_15` |
| `rec_pt_2023_core_15_v2/schema.json` | `pt_rec_15/schema.json` |
| `rec_pt_2023_core_15_v2/evaluation/core_15.json` | `pt_rec_15/evaluation/rec_15.json` |
| `rec_pt_2023_core_30_v2` | `pt_rec_30` |
| `rec_pt_2023_core_30_v2/schema.json` | `pt_rec_30/schema.json` |
| `rec_pt_2023_core_30_v2/evaluation/core_30_normal.json` | `pt_rec_30/evaluation/rec_30_normal.json` |
| `rec_pt_2023_core_30_v2/evaluation/core_30_topology.json` | `pt_rec_30/evaluation/rec_30_topology.json` |
| `rec_pt_2023_core_30_v2/evaluation/core_30_safety.json` | `pt_rec_30/evaluation/rec_30_safety.json` |
| `rec_pt_2023_core_30_v2/evaluation/core_30_robustness.json` | `pt_rec_30/evaluation/rec_30_robustness.json` |
| `rec_pt_2023_core_30_v2/evaluation/core_30_combined.json` | `pt_rec_30/evaluation/rec_30_combined.json` |
| `rec_pt_2023_core_30_v2/evaluation/references/core_30_combined_no_disturbances.json` | `pt_rec_30/evaluation/references/rec_30_combined_no_disturbances.json` |
| `rec_pt_2023_core_30_v2/evaluation/references/core_30_combined_nominal_limits.json` | `pt_rec_30/evaluation/references/rec_30_combined_nominal_limits.json` |
| `rec_pt_2023_core_30_v2/evaluation/references/core_30_topology_frozen_2024.json` | `pt_rec_30/evaluation/references/rec_30_topology_frozen_2024.json` |
| `rec_pt_2023_premium_100_v2` | `pt_rec_100` |
| `rec_pt_2023_premium_100_v2/schema.json` | `pt_rec_100/schema.json` |
| `rec_pt_2023_premium_100_v2/evaluation/premium_100_normal.json` | `pt_rec_100/evaluation/rec_100_normal.json` |
| `rec_pt_2023_premium_100_v2/evaluation/premium_100_allin.json` | `pt_rec_100/evaluation/rec_100_allin.json` |
| `rec_pt_2023_premium_100_v2/evaluation/references/premium_100_allin_frozen_2024.json` | `pt_rec_100/evaluation/references/rec_100_allin_frozen_2024.json` |
| `rec_pt_2023_premium_100_v2/evaluation/references/premium_100_allin_no_disturbances.json` | `pt_rec_100/evaluation/references/rec_100_allin_no_disturbances.json` |
| `rec_pt_2023_premium_100_v2/evaluation/references/premium_100_allin_no_dr.json` | `pt_rec_100/evaluation/references/rec_100_allin_no_dr.json` |

  Ver o guia [datasets REC de Portugal](portugal_rec_datasets.md).
- **Downloads de pastas aninhadas.** O `DataSet.get_dataset()` usa a
  implementação do CityLearn: dois pedidos à API do GitHub por dataset, ficheiros
  de raw.githubusercontent.com, `CITYLEARN_GITHUB_TOKEN` ou `GITHUB_TOKEN` quando
  definidos, e a conta e repositório dados ao `DataSet`. Os valores por defeito
  continuam a ser `Soft-CPS-Research-Group/Simulator` na tag
  `softcpsrecsimulator-v{versão}`.
- **Correções e documentação do CityLearn.** As raízes dos schemas resolvem-se a
  partir do ficheiro, as verificações físicas com topologia dinâmica consideram
  apenas os membros ativos, e passam a incluir-se os guias, exemplos, construção
  da documentação e matriz de CI do CityLearn (Python 3.9, 3.10 e 3.12; NumPy 1 e 2).
- **Dependências.** `doe_xstock`, `openstudio` e `torchvision` deixam de ser
  instalados por defeito; o `citylearn.end_use_load_profiles` precisa do extra
  `end_use_load_profiles`.
- O SoftCPS mantém os 19 datasets `ALADI *`, o `EC_Ermesinde` e a correção da
  CLI sem Stable Baselines3. **O gerador de Portugal continua no SoftCPS**: os
  scripts, fontes, resultados e catálogo continuam a gerar os pacotes completos
  `rec_pt_2023_*_v2`, agora como cópia de trabalho local em `data/datasets`,
  ignorada pelo Git, enquanto os datasets distribuídos coincidem com os do
  CityLearn 3.5.0. Os testes do gerador que precisam dessa cópia são ignorados
  sem ela; os restantes correm sobre os `pt_rec_*`. Ver o
  [guia de migração](../migration.md#upgrade-from-softcps-211-to-220).
- A validação incluiu:
  - **Testes:** 1.741 passaram e 35 foram ignorados. Desses, 34 são testes do gerador de Portugal que precisam da cópia local `rec_pt_2023_*_v2` e um é exclusivo de Windows. Com os pacotes da 2.1.1 presentes como essa cópia, os 149 testes de Portugal passaram. O `test_energy_provenance_toggle` falhou uma vez com quatro workers em paralelo e passou quando repetido sozinho. A fixture copia o dataset para a cache partilhada do utilizador, o que cria uma corrida entre workers; a fixture do CityLearn é igual.
  - **Auditoria e pacote:** a auditoria completa passou: 16 cenários físicos, contratos entity, os 61 datasets incluídos e paridade. O wheel e o sdist passaram o `twine check`. O wheel instalado passou a verificação de exportação de KPIs, BAU e manifesto fora do checkout.
  - **REC de Portugal:** os 19 schemas `pt_rec_*` correram 96 passos com o `BusinessAsUsualAgent` e o `evaluate_v2()` com a referência BAU nativa: os quatro `schema.json`, nove cenários principais e seis de comparação. A memória máxima foi 1,0 GB. Os rácios face ao BAU ao nível do distrito foram exatamente 1, como esperado para o BAU face à sua própria referência. Janelas maiores não foram corridas. Cada pasta `pt_rec_*` é idêntica byte a byte à do CityLearn `v3.5.0`.

## v2.1.1 — 2026-11-16 — referência BAU nativa em schemas entity estáticos

Responsável: **Tiago Fonseca** ([@calofonseca](https://github.com/calofonseca)).

- O `evaluate_v2()` com a referência BAU nativa (o valor por defeito) falhava em
  schemas entity com topologia estática e eventos de robustez sobre observações
  de EVs, baterias ou PV, como o REC-30 robustez de Portugal. A referência era
  criada em modo flat, que não expõe esses alvos, salvo com topologia dinâmica ou
  demand response. Passa a usar a interface entity sempre que o ambiente avaliado
  a usa.
- O agente BAU lê o estado do simulador, por isso as suas ações e KPIs não
  dependem da interface. Datasets, schemas e API mantêm-se.
- A validação incluiu a suite completa (1.758 testes, 6 ignorados), com um novo
  teste de regressão, e a auditoria completa (física, contratos entity, os 61
  datasets incluídos e paridade). Os 13 schemas de Portugal correram 96 passos
  com a referência BAU. Os 12 que já corriam na 2.1.0 deram KPIs idênticos e o
  REC-30 robustez passou a correr. Janelas maiores não foram repetidas.

## v2.1.0 — 2026-11-13 — baterias, ciclo de vida dos EVs e compatibilidade

Responsável: **Tiago Fonseca** ([@calofonseca](https://github.com/calofonseca)).

- A reserva limita a descarga e a bateria inicia no SOC configurado. Um EV
  mantém a identidade quando muda de carregador ou volta a ligar-se. O SOC
  evolui com perdas e mobilidade, e os exports mostram o carregador efetivo.
- O BAU usa o mesmo equipamento, PV e cortes de rede que o controlador.
  Corrige os rácios com referências negativas, a comparação de custos do
  mercado e a agregação dos KPIs.
- Suporta NumPy 1 e 2 e Gymnasium 1.x. Mantém os modelos de ocupação antigos,
  retoma agentes guardados e protege a cache de datasets com subpastas.
- A validação incluiu 1.762 testes locais, 16 auditorias físicas, cenários
  portugueses de 4, 30 e 100 membros e exports CSV/Parquet abertos na UI.
  Os ficheiros dos datasets Portugal são os mesmos da 2.0.2.

## v2.0.2 — 2026-11-05 — datasets Portugal reproduzíveis e downloads completos

Responsável: **Tiago Fonseca** ([@calofonseca](https://github.com/calofonseca)).

- Publica `portugal-realism-4` e `portugal-ev-mobility-1` nas quatro comunidades,
  em 2023/2024: diversidade nos carregamentos, atividade residencial, ocupação e
  equipamentos comerciais, orçamentos anuais, falhas locais e pedidos de DR.
  Preserva os casos de stress identificados, equipamentos, nove cenários principais
  e seis referências comparáveis.
- Inclui gerador determinístico, referências empíricas compactas, hipóteses e
  evidência de sementes, sensibilidade e reprodução. São benchmarks sintéticos;
  não constituem medições representativas da população portuguesa.
- Corrige o descarregamento de subpastas e só disponibiliza a cache depois de
  completar todos os ficheiros. Downloads interrompidos podem ser repetidos.
  Acrescenta regressões do carregamento e obtém o commit de origem fixa no CI.
- Permite iniciar a interface de linha de comandos sem Stable Baselines3;
  carregar um agente SB3 indica explicitamente a dependência em falta. Corrige
  as horas UTC em Python 3.9/3.10 e alinha a exportação da avaliação com os
  intervalos executados, excluindo a observação terminal. Remove cinco colunas
  obsoletas de carga parcial; nomes inválidos explícitos passam a falhar, em vez
  de repetir silenciosamente a coluna anterior.
- Mantém a dinâmica da simulação, schemas dos datasets e convenção de preços OMIE.
  Resultados anteriores precisam da revisão original dos dados para comparação.

A validação dos dados anterior à release passou 161 testes específicos,
72 janelas de fronteiras, 27 sondas de falhas e 18 janelas contabilísticas;
duas regenerações produziram Parquet idênticos. A release no GitHub acompanha
os testes gerais, auditorias e verificação da instalação. Não inclui campanha
anual de controladores nem treino MARL. As limitações empíricas estão na
[validação dos datasets](../portugal_rec_native/Heldout_2024.md).

Distribuição/import mantêm-se `softcpsrecsimulator` / `citylearn`. A tag e referência
online por omissão são `softcpsrecsimulator-v2.0.2`. Instalar o extra `parquet`
para Portugal. Não exige migração de schemas; mantém referências explícitas.

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
