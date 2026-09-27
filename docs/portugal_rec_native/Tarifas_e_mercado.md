# Tarifa comum e mercado interno

Decisão de 27 de setembro de 2026: uma série de compra à rede igual para todos
os membros e para as nove configurações. A referência é `pricing_btn_bi.parquet`,
BTN bi-horária indexada a OMIE 2023, incluindo os componentes variáveis já
documentados em [Fontes e método](Fontes_e_metodo.md). As previsões seguem a mesma
referência. Não se aplicou desconto de 20%, piso a zero ou deslocação temporal.
Custos fixos e IVA não integram esta série variável; não é uma fatura completa.

## Dados e configuração experimental

Os schemas de dados contêm apenas:

```json
"community_market": {
  "enabled": false,
  "grid_export_price": 0.0
}
```

Para uma experiência com mercado comunitário, substituir esse bloco na cópia
do schema da experiência pela configuração nativa:

```json
"community_market": {
  "enabled": true,
  "local_price_ratio_to_grid_import": 0.8,
  "grid_export_price": 0.0
}
```

O campo `local_price_ratio_to_grid_import` é lido pelo simulador original.
Não é necessário alterar o simulador nem implementar uma liquidação externa.
Para cada intervalo, o preço local será 0,8 vezes o preço comum de compra à rede;
o mesmo preço é aplicado a compradores e vendedores. A exportação à rede vale
zero. O schema resolvido de cada experiência deverá conservar esta escolha.
A ativação do mercado não cria outro dataset físico.

Esta decisão substitui a atribuição de tarifas por instalação. A coluna
`tariff_id` dos catálogos passa a `btn_bi`; `source_tariff_id` é histórica.
Os parâmetros elétricos, potências contratadas e ligações de fase mantêm-se:
a uniformização é uma convenção de preços, não uma uniformização das instalações.

As simulações da versão anterior não avaliam esta revisão. Nenhuma nova avaliação
de utilização ou controlador foi realizada nesta alteração.
