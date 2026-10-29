# Preço OMIE e mercado comunitário

Decisão de 29 de outubro de 2026, edição `portugal-native-2023-2024-v3.2`.
As quatro bases e nove configurações usam `shared/pricing_omie.parquet`,
com 70 176 intervalos de 15 minutos, cobrindo 2023 e 2024.

## Regra de preço

`preço_rede[t] = max(OMIE_português[t] / 1000, 0)`

O OMIE de entrada está em EUR/MWh. A divisão por 1000 converte-o em EUR/kWh;
um preço positivo mantém-se, um preço negativo passa a zero e zero mantém-se.
Cada preço horário é repetido nos quatro quartos de hora, após o alinhamento
do calendário de Madrid com UTC. Não se somam margem, perdas, MIBEL, TAR,
custos de sistema, taxas ou impostos. Esta é uma convenção experimental baseada
no mercado grossista, não uma reprodução de uma fatura de consumidor.

As componentes Coopérnico/ERSE anteriores permanecem apenas como proveniência.
A antiga série está em `sources/legacy_retail_pricing_2023_2024.parquet`;
nenhum schema corrente a seleciona. A coluna ativa `tariff_id` é
`omie_nonnegative`; `source_tariff_id` conserva a atribuição histórica.

## Mercado ativo nos datasets

Tanto `schema.json` como os nove schemas em `evaluation/` contêm:

```json
"community_market": {
  "enabled": true,
  "local_price_ratio_to_grid_import": 0.8,
  "grid_export_price": 0.0
}
```

O simulador aplica `preço_local[t] = 0.8 × preço_rede[t]` uma única vez,
na liquidação. O mesmo preço remunera o vendedor e é cobrado ao comprador.
A série de entrada contém sempre 100% do preço da rede. A exportação para
a rede pública continua a valer zero, independentemente do OMIE.

| OMIE (EUR/MWh) | Compra à rede (EUR/kWh) | Troca interna (EUR/kWh) |
|---:|---:|---:|
| 100 | 0,100 | 0,080 |
| 0 | 0 | 0 |
| -10 | 0 | 0 |

## Previsões e reprodução

As previsões a 1, 6 e 24 horas usam a mesma conversão e piso a zero.
Utilizam o preço do destino apenas após a publicação conservadora às 14h
de Madrid; antes disso recorrem ao mesmo instante do dia anterior. No início
do arquivo, a persistência é limitada à primeira observação disponível.
O dia externo de 1 de janeiro de 2025 completa o horizonte final, sem criar
um terceiro ano simulado. Não há componentes tarifárias futuras nas previsões.

`python scripts/apply_portugal_omie_pricing.py` reproduz esta revisão offline,
a partir dos preços OMIE arquivados, e atualiza schemas, catálogos,
manifestos e hashes. Ver [Reproduzir](Reproduzir.md).
Os resultados da verificação corrente estão em
`results/portugal_rec_native/pricing_validation.json`; os relatórios anteriores
conservam o seu âmbito histórico. As comparações de controladores devem usar
a mesma edição de preços, incluindo os respetivos baselines.
