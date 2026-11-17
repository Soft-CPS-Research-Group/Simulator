# Guia de Desenvolvimento

Esta pagina junta comandos de desenvolvimento local, validacao, smoke tests de performance e o mapa atual da arquitetura interna.

Versao inglesa: [../developer_guide.md](../developer_guide.md).

## Ambiente Local

Usar o virtual environment do repositorio quando existir:

```console
.venv/bin/pip install -e .
```

Dependencias opcionais:

| Funcionalidade | Comando |
|---|---|
| Datasets Parquet | `.venv/bin/pip install pyarrow` |
| PV autosizing | `.venv/bin/pip install ".[pysam]"` |
| Checks de build | `.venv/bin/pip install build twine` |

## Evidência portátil de experiências

O `.gitattributes` torna cada checkout idêntico ao repositório byte a byte: os
ficheiros de texto usam LF e os ficheiros em `data/` e `tests/data/` nunca são
convertidos, mesmo com o `core.autocrlf=true` por omissão do Git for Windows. Os
manifestos de experiências calculam hashes destes ficheiros, que assim coincidem
em todas as plataformas. Um checkout anterior ao `.gitattributes` pode ainda ter
cópias CRLF; o `git status` mostra os ficheiros de dados convertidos como
modificados. Fazer um clone novo e configurá-lo como em
[Ambiente Local](#ambiente-local): o clone aplica as novas regras de checkout e
deixa intacta a cópia de trabalho existente, incluindo as alterações não
commitadas. Depois, limpar a cache de datasets, que pode conter cópias feitas a
partir do checkout antigo:

```console
git clone https://github.com/Soft-CPS-Research-Group/Simulator.git
python -c "from citylearn.data import DataSet; DataSet().clear_cache()"
```

Os manifestos registam cada ficheiro de entrada pelo seu caminho abaixo da raiz
do dataset, com separadores `/`, pelo que um dataset movido ou copiado mantém a
sua evidência. No Windows, onde `\` e `/` são ambos separadores e os nomes de
ficheiros não distinguem maiúsculas de minúsculas, uma referência que só difere
do caminho do ficheiro nos separadores ou nas maiúsculas regista a grafia do
ficheiro no disco: com a raiz `C:\data\demo` e o ficheiro
`C:\data\demo\sub\weather.csv`, as referências `sub\Weather.csv`,
`SUB/weather.csv` e `C:\Data\Demo\sub\WEATHER.csv` registam todas
`sub/weather.csv`. As outras grafias de um ficheiro de entrada mantêm o seu
texto, com `/` em vez de `\` no Windows: componentes `.` ou `..`, separadores
repetidos, ligações (links), nomes curtos 8.3 e ficheiros fora da raiz do
dataset. As strings que não são ficheiros de entrada com hash (ver `FILE_TYPES`
em `citylearn/internal/temporal.py`) mantêm o seu texto sem a raiz do dataset.
No Linux e no macOS todas as referências mantêm o seu texto, sem a raiz do
dataset; por omissão, os nomes de ficheiros no macOS não distinguem maiúsculas
de minúsculas, mas a evidência distingue. Se duas execuções escreverem o mesmo
ficheiro de formas que não são normalizadas, a comparação indica `incompatible`
e não um par.

## Validacao Principal

| Check | Comando |
|---|---|
| Suite completa | `.venv/bin/pytest -q` |
| Lint critico | `.venv/bin/python -m ruff check citylearn tests scripts/manual scripts/ci --select E9,F821` |
| Auditoria entity contract | `.venv/bin/python scripts/audit/audit_entity_contract.py --strict` |
| Auditoria fisica | `.venv/bin/python scripts/audit/audit_physics.py` |
| Whitespace diff | `git diff --check` |

A auditoria fisica exercita de proposito caminhos de conversao de resolucao, por isso podem aparecer avisos de unit conversion quando a fixture força diferencas entre dataset e schema.

## Testes portáteis e verificação obrigatória

Os dados dos testes são localizados a partir dos próprios ficheiros, sem depender
da pasta corrente. A falta de um schema obrigatório dá um erro claro; um dataset
sintético ou em cache não deve substituir silenciosamente a fixture em falta.
Verificar a portabilidade executando a mesma suite na raiz e noutra pasta, usando
os caminhos absolutos do Python do checkout e da pasta `tests` nesta última.

```console
.venv/bin/python scripts/audit/run_full_audit.py --output-dir /tmp/simulator-audit
```

O modo por omissão é de verificação: auditor em falta, erro ou dependência que
impeça a execução bloqueiam a aprovação. `--strict` mantém-se como alias.
`--report-only` recolhe explicitamente diagnósticos com saída zero; consultar
`verification_passed` no resumo, pois essa saída não representa aprovação.
Caminhos de saída relativos são resolvidos na pasta do chamador antes de lançar
os subprocessos.

A cobertura por omissão inclui física, contrato entity, execuções curtas offline
dos datasets e referências de comandos fixos para refatorização. O resumo indica
as famílias escolhidas e se contém todas as famílias por omissão. `--audits`
permite um subconjunto; esse subconjunto não certifica todo o simulador nem
substitui pytest, campanhas prolongadas ou calibração empírica. O auditor legado
`performance_results` inclui comparações RBC/SAC e exige seleção explícita,
fora das verificações normais do simulador. Atualizar snapshots exige
`--report-only`; a verificação não altera as próprias referências. Relatórios de
trabalho ficam fora da árvore versionada.

## Referências para refatorização interna

`audit_refactor_parity.py` usa seis casos curtos com comandos fixos: interfaces
flat/entity, passos de 60/300/900/3600 segundos, limites por fase, cortes e ciclo
de vida de membros com falhas combinadas. Reutiliza a fábrica de fixtures curtas
do checkout e requer `test_requirements.txt` além das dependências de execução.
Não executa agentes, treino ou BAU auxiliar. A fixture versionada
`scripts/audit/snapshots/refactor_parity.json` fixa o comportamento após as
correções de robustez na base 1.9.11, antes da reorganização interna seguinte.

Cada caso verifica independentemente energia, dinheiro e proveniência, e compara
inputs, ações pedidas/executadas, SOC, fluxos, recompensa, serviço das máquinas,
KPIs, layouts/limites das interfaces e semântica das exportações CSV/JSON.
Versões das dependências, commits e caminhos temporários não são referências de
comportamento. Os layouts são guardados uma vez por mudança de topologia. Há
pedidos de lavagem servidos e não servidos; estas referências provam paridade de
comportamento, sem avaliar a qualidade do escalonamento.

```console
.venv/bin/python scripts/audit/audit_refactor_parity.py --output /tmp/parity-check.json
```

As tolerâncias absolutas seguem as unidades: kWh `1e-6`, EUR `1e-7`, SOC/rácios
`1e-7`, kW `1e-5`, restantes valores numéricos `1e-6`, com tolerância relativa
`1e-7`. Contagens, política de tolerâncias, flags, nomes, ordenação, chaves em
falta e desconhecido/zero são exatos. Não finitos têm representação explícita.
Diferenças ou referência em falta dão erro; a verificação nunca a regenera.

Uma mudança intencional de contrato exige uma proposta **nova** separada, revisão
das diferenças e consequências físicas/de serviço e, só depois, substituição
deliberada da fixture revista:

```console
.venv/bin/python scripts/audit/audit_refactor_parity.py --capture-reference /tmp/parity-proposal.json --output /tmp/parity-capture.json
```

A captura recusa sobrescrever ficheiros e indica `reference_proposal` com
`verification_passed=false`. Não é uma verificação aprovada. O JSON guardado é
uma referência de teste; diários e relatórios de execução não são fixtures.
O relatório também tem de ter um destino distinto das referências/propostas,
incluindo caminhos equivalentes.

## Scripts Manuais

Scripts utilitarios vivem em `scripts/manual` e estao excluidos da colecao default do pytest.

```console
.venv/bin/python scripts/manual/demo_ev_rbc.py
.venv/bin/python scripts/manual/demo_ev_rbc_export_end.py
```

Benchmark de runtime:

```console
.venv/bin/python scripts/manual/bench_runtime.py --seconds 5 60 --render-modes none end --episode-steps 1200
```

Smoke check de performance para CI:

```console
.venv/bin/python scripts/ci/perf_smoke.py --episode-steps 600 --seconds 60 --baseline-file scripts/ci/perf_baseline.json
```

Profiling de componentes do `step`:

```console
.venv/bin/python scripts/audit/profile_step_breakdown.py --episode-steps 300 --seconds 60 --agent rbc --interface flat --render-mode none --no-write --table-limit 14
```

Profiling de lifecycle e KPI/BAU:

```console
.venv/bin/python scripts/audit/profile_lifecycle.py --episode-steps 300 --seconds 60 --skip-exports --no-write
```

Remover `--skip-exports` quando for preciso medir tambem os caminhos de export CSV.

Para diagnostico ad hoc, passar `debug_timing=True` ao `CityLearnEnv` e ler chaves do `info` como `apply_actions_time`, `reward_observations_time`, `next_observations_time`, `terminal_export_time` e `step_total_time`.

## Arquitetura Interna

As APIs publicas continuam centradas em `CityLearnEnv` e `Building`. A orquestracao interna esta separada em modulos de servico em `citylearn/internal`.

| Modulo | Responsabilidade |
|---|---|
| `loading.py` | Loading por schema e montagem dos buildings. |
| `loading_inputs.py` | Séries de entrada, cache partilhada, janelas dinâmicas e conversão temporal. |
| `loading_components.py` | Resolução dos construtores, modelos opcionais, carregadores, sementes e dimensionamento. |
| `runtime.py` | Runtime de episodios, `step`, parsing de acoes, progressao temporal e associacao EV/charger. |
| `building_ops.py` | Orquestracao de observacoes e acoes por building. |
| `electrical_limits.py` | Cálculo puro em kW usado pelos limites do serviço no motor. |
| `kpi.py` | Pipeline de KPIs/avaliacao. |
| `kpi_energy.py` | Registos de energia, carbono, custos, conforto e forma da carga do distrito. |
| `kpi_equipment.py` | Cálculos dos equipamentos e agregação EV/BESS/PV/fases por avaliação. |
| `kpi_community.py` | Registos do mercado comunitário e equidade. |
| `kpi_service_metrics.py` | Agregação do serviço pedido/executado das máquinas e escadas rolantes. |
| `entity_interface.py` | Tabelas entity, edges, action specs e bundles de observacao. |
| `entity_observations.py` | Preenchimento das tabelas, máscaras das relações, saneamento e cópias independentes. |
| `entity_layout.py` | Linhas estáticas, espaços numéricos e especificações declarativas da interface. |

O cálculo dos limites elétricos recebe pedidos com sinal, totais e por fase, e
uma estimativa das cargas base. Devolve reduções proporcionais e a pressão
pedida/residual sem modificar equipamentos. Preserva a ordem total antes de fases,
importação antes de exportação e o máximo de oito passagens, recalculando os
totais após cada redução. Uma sobrecarga não controlável continua visível na
pressão residual; este cálculo não cria energia nem corta cargas fixas.

Um controlador pode prever com cargas prospetivas. O motor recalcula os limites, converte potência
em ações, regista o feedback e aplica a física das baterias e dos carregadores.
Durante cortes, consulta ainda o saldo real em kWh antes de cada carga
controlável; essa verificação sequencial não pode ser substituída pela previsão.
O limitador antigo exclusivo dos carregadores e a viabilidade futura dos ciclos
das máquinas mantêm os seus contratos distintos. As baselines submetem os seus
pedidos e o runtime partilha o limite do serviço; ver [baselines](../baselines.md)
para cada política.

## Regras de Desenvolvimento

| Regra | Motivo |
|---|---|
| Documentar mudancas de schema/API na mesma alteracao. | Algoritmos e datasets dependem de contratos explicitos. |
| Adicionar testes para fisica ou observacoes novas. | Sub-hourly e entity podem regredir sem falhas obvias. |
| Preferir observacoes aditivas em patch releases. | Evita quebrar algoritmos e wrappers ja treinados. |
| Correr auditorias antes de taggar. | Apanham drift de contrato fora dos unit tests normais. |

## Conteúdo do repositório

Conservar código, documentação necessária do produto/datasets, licenças,
changelog central, fixtures normativas e proveniência necessária dos dados.
Planos, diários, JSON de execução de releases, revisão de PDFs e dumps de
observações ficam fora do checkout ou em artefactos de CI/releases.
Antes de retirar material de trabalho, arquivar e verificar hashes; guardar a
correspondência de origem/destino nessa evidência externa. Não reescrever o Git.
O [catálogo dos datasets](../portugal_rec_native/README.md#proveniência-conservada-no-produto)
identifica os comprovativos conservados e os seus consumidores. Os checksums da
distribuição descrevem o pacote corrente, não um pacote documental anterior.

## Verificação e módulos internos na 3.1

O comando `python scripts/audit/run_full_audit.py --strict --output-dir /tmp/citylearn-audit`
verifica física, contrato de entidades, datasets oficiais e seis trajetórias de
referência com comandos fixos. Falhas, dependências ausentes e referências divergentes
produzem erro; a execução normal não reescreve snapshots. Auditorias de algoritmos
continuam opcionais e explícitas. As referências de teste usam casos temporários
pequenos e os fixtures existentes, sem incluir datasets do SoftCPS.

As fachadas públicas delegam em módulos internos de energia/equipamentos/serviço/
comunidade nos KPIs, observações/layout de entidades, componentes/inputs no loading
e contratos/auditoria de robustez. O módulo de limites elétricos é partilhado pelo
controlo e pelo runtime. Importações públicas, exemplos e catálogo mantêm a
identidade CityLearn; estes módulos internos não são APIs de extensão estáveis.

### Desempenho e leitura dos históricos

A avaliação pode receber `progress_callback`, com eventos de etapa/membro/família
de KPIs; os tempos não são acrescentados à tabela dos indicadores. O auditor
contínuo distingue dispatch, avaliação e exportação nos ficheiros de progresso.

Os históricos do mercado e da robustez mantêm todo o detalhe e a ordem em
representações compactas sem perdas. Consultas por janela/iterador evitam
materializar a execução inteira; leituras são snapshots independentes e alterações
históricas devem ser atribuídas de volta. O reset limpa histórico e contadores.
As máscaras de população dos KPIs e os replays EV são reutilizados apenas no
contexto de avaliação/partida apropriado. Ciclos diferíveis são indexados e a
preparação de previsões/observações é reutilizada dentro do pedido, mantendo a
semântica e o fallback de recompensas personalizadas.
