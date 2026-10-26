# Portugal REC — 2023 e 2024

Versão **portugal-native-2023-2024-v3.1**. As mesmas **quatro bases físicas e nove configurações**, agora com **70 176 registos de 15 minutos**, de 1 de janeiro de 2023 a 31 de dezembro de 2024. Os caminhos mantêm «2023» por estabilidade; os inputs correntes incluem ambos os anos.

2024 continua cronologicamente 2023. Quem saiu não regressa; os equipamentos retirados permanecem retirados. Os eventos de entrada/saída de 2023 não são repetidos. Não foram acrescentados membros, equipamentos ou variantes. Nas configurações estáticas mantém-se o catálogo completo; nas dinâmicas mantém-se a composição existente no fim de 2023.

A preparação original destes dados manteve o motor no commit `978be68b0e0740deab15ecf536805f323b86fa57` (1.8.0). Essa é a referência histórica dos relatórios arquivados. Desde a consolidação de 26/10/2026, a pasta de trabalho combina estes mesmos dados com o motor 1.8.3 e as alterações locais descritas em [Releases](../releases.md). Os ficheiros dos datasets foram preservados. Esta integração não constitui uma nova campanha anual de controladores.

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

A tarifa de compra à rede é comum a todos: `shared/pricing_btn_bi.parquet`, com OMIE e componentes próprias de cada ano. A exportação à rede vale zero. O mercado interno fica desligado; os 80% configuram-se posteriormente no schema da experiência, usando a liquidação original.

## Consultar

- [Relatório descritivo em inglês, com gráficos](Portugal_REC_Dataset_Report.pdf) e [versão editável](Portugal_REC_Dataset_Report.md).
- [Tabelas, estatísticas e reprodução do relatório](dataset_description/README.md).
- [Extensão de 2024 e verificações](Extensao_2024.md).
- [Histórias, previsões e variabilidade](Previsoes_e_historias.md).
- [Reproduzir os dois anos](Reproduzir.md).
- [Atlas de membros Premium](Atlas_Membros_Premium.md).
- [Tarifas e mercado](Tarifas_e_mercado.md).
- [Fontes e método](Fontes_e_metodo.md).
- [Fecho imutável de 2023](Fecho_2023.md).

O manifesto corrente está em `results/portugal_rec_native/manifest.json`; a verificação dos inputs em `two_year_integrity.json`, na mesma pasta. O pacote corrente é `docs/portugal_rec_native.zip`.

O fecho de 2023 está no commit `393cd200bb66b671d8904eba00688d56d91f08d3` e no pacote separado `Simulator-recovery-archive/2026-09-27-final-2023/portugal_rec_2023_frozen.zip`. Esse arquivo é proveniência, não mais um dataset ativo.

## Limites da afirmação de fecho

Os consumos, rotinas e famílias são sintéticos, apoiados nas fontes documentadas. A meteorologia e o mercado usam fontes do ano correspondente. Os pedidos EV foram verificados individualmente face a duração, potência, perdas e redução de potência perto de carga completa; isto não prova viabilidade conjunta sob limites da instalação, falhas, cortes e decisões de controlo.

Mantêm-se as capacidades do simulador original: SOC prescrito à chegada de cada sessão EV, fases das máquinas por instalação, backup agregado, possível exposição dos eventos de health e DR com média da hora anterior. Os itinerários não constituem um balanço longitudinal de viagens executado pelo motor. Não existe atividade privada após saída de um membro.

Nesta revisão, as histórias são geradas dos eventos por configuração. O apoio de um dia de janeiro de 2025 completa as previsões de preço no fim de 2024, sem alargar a simulação. As previsões derivadas de consumo/PV continuam sujeitas ao corte de horizonte do motor original no fim de cada episódio; ver o relatório de previsões.
