# Portugal REC — catálogo corrente

Versão **portugal-native-978be68b-v2.1**. Nove configurações sobre quatro bases
físicas de **2023**, com intervalos de **15 minutos**. O motor permanece no commit
`978be68b0e0740deab15ecf536805f323b86fa57`, sem alterações ao pacote `citylearn`.
As designações `_v2` nos diretórios são mantidas para estabilidade dos caminhos.

O trabalho atual é a construção dos datasets. Nesta revisão foram verificadas a integridade dos ficheiros e a coerência dos
schemas. Duas sessões EV foram verificadas em janelas de 90 minutos no motor
original, para comprovar viabilidade dos pedidos. Não houve campanha anual,
comparação de controladores ou avaliação de desempenho.

## Configurações

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

DR apenas no Premium All-in. Os quatro `schema.json` são atalhos para o normal
de cada família, não configurações adicionais. `CURRENT_EVALUATION.json` é o
índice de configurações; o nome histórico do ficheiro não significa que exista
uma avaliação desta revisão. A composição efetiva segue os eventos de cada schema.

## Tarifa e mercado

Todos os membros, em todas as configurações, usam a mesma série de compra à rede:
`shared/pricing_btn_bi.parquet`, referência BTN bi-horária indexada a OMIE 2023.
Os valores em EUR/kWh e as previsões foram preservados, sem multiplicação por 0,8.
A série representa o preço variável documentado nas fontes, não uma fatura completa.

O mercado interno está desligado nos schemas de dados. Nas experiências,
ativa-se o settlement original e configura-se o fator de **80%** no schema.
A exportação à rede vale **zero**. Ver [Tarifas e mercado](Tarifas_e_mercado.md).

## Ficheiros e documentação

- [Reprodução dos dados](Reproduzir.md).
- [Atlas das famílias e instalações](Atlas_Membros_Premium.md).
- [Fontes e método](Fontes_e_metodo.md).
- [Correções de tarifas e das duas sessões EV](Fecho_Tarifas_SOC.md).
- [Histórico e arquivo da revisão anterior](Historico.md).
- Manifesto: `results/portugal_rec_native/manifest.json`.
- Verificação estática: `results/portugal_rec_native/static_integrity.json`.
- Pacote corrente: `docs/portugal_rec_native.zip`.

Os perfis de carga, PV, meteorologia, carbono, rotinas, equipamentos e eventos
mantêm-se, exceto duas visitas EV do Core-30 prolongadas em 15 minutos para
cumprir o SOC solicitado. Os catálogos e as histórias indicam a tarifa comum;
`source_tariff_id` conserva a atribuição anterior apenas como proveniência.
As outras séries de tarifas preservadas em `shared/` não são usadas pelos schemas.

## Semântica do motor original

EVs usam SOC prescrito por sessão; as histórias e os itinerários não constituem
um balanço longitudinal simulado. As máquinas usam a repartição de fases da
instalação. O backup é o modelo agregado original. Os metadados de health podem
revelar eventos. A DR usa a média da hora anterior, sem qualificação comercial.
Quem sai deixa de ter atividade no cenário, juntamente com os seus ativos.
Estas fronteiras constam também de `native_compatibility` em cada schema.

[Fecho de 2023](Fecho_2023.md): histórias revistas e pedido extremo de DR mantido.
