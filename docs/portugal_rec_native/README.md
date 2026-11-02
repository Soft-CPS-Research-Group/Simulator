# Portugal REC — 2023 e 2024

## Edição corrente v3.3 — campanha reservada e referências emparelhadas

A edição corrente é **portugal-native-2023-2024-v3.3**, revisão de campanha
**portugal-heldout-2024-1**, mantendo a revisão de fronteiras **portugal-boundaries-1**.
O [relatório de passagem de trabalho](Heldout_2024.md) é a descrição atual dos
eventos de 2024, populações, recursos, seis referências emparelhadas e verificações.
As quatro bases estáticas e nove configurações principais mantêm-se. As três
configurações dinâmicas recebem seis eventos cada; há uma identidade suplementar
por base dinâmica, disponível apenas após a admissão. As referências estão em
`evaluation/references/`, sem duplicar as séries que permanecem iguais.
Para treino/seleção, usar as vistas limitadas a 2023 em
`evaluation/preparation_2023/`; os schemas completos de avaliação contêm o
catálogo e os eventos reservados de 2024 nos seus metadados.

O manifesto corrente está em `results/portugal_rec_native/manifest.json` e a
evidência desta edição em `results/portugal_rec_native/heldout_2024/`.
O PDF descritivo, as figuras, as tabelas em `dataset_description/` e as narrativas
anteriores conservam explicitamente o âmbito **v3.2**; não descrevem a nova campanha.
Os inputs físicos originais permanecem disponíveis e identificados por checksum.
Os resultados anteriores não são resultados de v3.3. Ver [reprodução](Reproduzir.md).

## Descrição histórica da edição v3.2

Versão **portugal-native-2023-2024-v3.2**. As mesmas **quatro bases físicas e nove configurações**, agora com **70 176 registos de 15 minutos**, de 1 de janeiro de 2023 a 31 de dezembro de 2024. Os caminhos mantêm «2023» por estabilidade; os inputs correntes incluem ambos os anos.

2024 continua cronologicamente 2023. Quem saiu não regressa; os equipamentos retirados permanecem retirados. Os eventos de entrada/saída de 2023 não são repetidos. Não foram acrescentados membros, equipamentos ou variantes. Nas configurações estáticas mantém-se o catálogo completo; nas dinâmicas mantém-se a composição existente no fim de 2023.

A preparação original fixou o motor no commit
`978be68b0e0740deab15ecf536805f323b86fa57` (1.8.0). Essa é uma referência
histórica de construção; a versão corrente do código é 1.9.11, descrita no
[changelog](../releases.md). A edição dos dados v3.2 e a revisão
`portugal-boundaries-1` conservam a sua identidade própria. A revisão tarifária
de 29/10/2026 alterou preços e mercado, preservando os inputs físicos. Integrar
dados e motor não constitui uma nova campanha anual de controladores.

| Configuração | Schema relativo à raiz do Simulator |
|---|---|
| Core-4 | `data/datasets/rec_pt_2023_micro_4_v2/evaluation/core_4.json` |
| Core-15 | `data/datasets/rec_pt_2023_core_15_v2/evaluation/core_15.json` |
| Core-30 Normal | `data/datasets/rec_pt_2023_core_30_v2/evaluation/core_30_normal.json` |
| Core-30 Topologia | `data/datasets/rec_pt_2023_core_30_v2/evaluation/core_30_topology.json` |
| Core-30 Safety | `data/datasets/rec_pt_2023_core_30_v2/evaluation/core_30_safety.json` |
| Core-30 Robustness | `data/datasets/rec_pt_2023_core_30_v2/evaluation/core_30_robustness.json` |
| Core-30 Combinado | `data/datasets/rec_pt_2023_core_30_v2/evaluation/core_30_combined.json` |
| Premium Normal | `data/datasets/rec_pt_2023_premium_100_v2/evaluation/premium_100_normal.json` |
| Premium All-in | `data/datasets/rec_pt_2023_premium_100_v2/evaluation/premium_100_allin.json` |

DR apenas no Premium All-in; o pedido deliberadamente impossível de 1500 kW durante quatro horas foi mantido, com ocorrência nos dois anos. Health, cortes e ruído conservam as configurações e os níveis anteriores, com campanhas sazonais em 2024. A topologia não é reiniciada no novo ano.

A tarifa de compra à rede é comum a todos: `shared/pricing_omie.parquet`, com `max(OMIE português em EUR/MWh / 1000, 0)`, em EUR/kWh. Não inclui componentes Coopérnico, perdas, MIBEL, TAR ou impostos. O mercado comunitário está ativo em todos os schemas: o comprador paga e o vendedor recebe 80% desse preço. A exportação à rede vale zero. As previsões seguem a mesma regra com disponibilidade causal do OMIE.

## Consultar

- [Relatório descritivo em inglês, com gráficos](Portugal_REC_Dataset_Report.pdf) e [versão editável](Portugal_REC_Dataset_Report.md).
- [Tabelas, estatísticas e reprodução do relatório](dataset_description/README.md).
- [Extensão de 2024 e verificações](Extensao_2024.md).
- [Histórias, previsões e variabilidade](Previsoes_e_historias.md).
- [Reproduzir os dois anos](Reproduzir.md).
- [Atlas de membros Premium](Atlas_Membros_Premium.md).
- [Tarifas e mercado](Tarifas_e_mercado.md).
- [Fontes e método](Fontes_e_metodo.md).
- [Registo histórico do fecho de 2023](https://github.com/Soft-CPS-Research-Group/Simulator/blob/8157f605e26fb75651037451fb453b1a48056787/docs/portugal_rec_native/Fecho_2023.md).

O manifesto corrente está em `results/portugal_rec_native/manifest.json`; a verificação desta revisão em `pricing_validation.json`, na mesma pasta. `two_year_integrity.json` documenta a construção histórica dos inputs físicos. O pacote corrente é `docs/portugal_rec_native.zip`.

O fecho de 2023 está no commit `393cd200bb66b671d8904eba00688d56d91f08d3`.
Os [comprovativos dessa edição](https://github.com/Soft-CPS-Research-Group/Simulator/tree/8157f605e26fb75651037451fb453b1a48056787/results/portugal_rec_native/frozen_2023_evidence)
e os [registos das tarifas/sessões EV](https://github.com/Soft-CPS-Research-Group/Simulator/blob/8157f605e26fb75651037451fb453b1a48056787/docs/portugal_rec_native/Fecho_Tarifas_SOC.md)
conservam o âmbito histórico. Não são mais um dataset ativo nem prova de
desempenho do motor corrente. O [histórico de construção](https://github.com/Soft-CPS-Research-Group/Simulator/blob/8157f605e26fb75651037451fb453b1a48056787/docs/portugal_rec_native/Historico.md)
identifica as edições anteriores e as adaptações entre anos.

## Limites da afirmação de fecho

Os consumos, rotinas e famílias são sintéticos, apoiados nas fontes documentadas. A meteorologia e o mercado usam fontes do ano correspondente. Os pedidos EV foram verificados individualmente face a duração, potência, perdas e redução de potência perto de carga completa; isto não prova viabilidade conjunta sob limites da instalação, falhas, cortes e decisões de controlo.

Mantêm-se as capacidades do simulador original: SOC prescrito à chegada de cada sessão EV, fases das máquinas por instalação, backup agregado, possível exposição dos eventos de health e DR com média da hora anterior. Os itinerários não constituem um balanço longitudinal de viagens executado pelo motor. Não existe atividade privada após saída de um membro.

Nesta revisão, as histórias são geradas dos eventos por configuração. O apoio de um dia de janeiro de 2025 completa as previsões de preço no fim de 2024, sem alargar a simulação. As previsões derivadas de consumo/PV continuam sujeitas ao corte de horizonte do motor original no fim de cada episódio; ver o relatório de previsões.

## Proveniência conservada no produto

O diretório `results/portugal_rec_native/` contém oito ficheiros de proveniência
necessários à distribuição/reconstrução corrente:

| Ficheiro | Uso |
|---|---|
| `manifest.json` | Identidade da edição, variantes, configuração e referência histórica do motor. |
| `pricing_policy.json` | Fórmula tarifária, fontes e condições das previsões. |
| `pricing_validation.json` | Verificações tarifárias exigidas pelo empacotador. |
| `boundary_integrity.json` | Correções dos limites da presença e censura de sessões. |
| `boundary_validation.json` | Checksums dos inputs e verificação exigida pelo empacotador. |
| `forecasts_and_stories.json` | Métricas/fontes consumidas pelo relatório descritivo. |
| `two_year_integrity.json` | Proveniência da continuidade e construção dos dois anos. |
| `publication.json` | Identidade e verificações da entrega dos dados. |

Relatórios de execução do motor e notas de trabalho são guardados fora do
checkout ou como artefactos de CI. Os registos desta tabela não representam
resultados atuais de uma política nem substituem as verificações do simulador.
