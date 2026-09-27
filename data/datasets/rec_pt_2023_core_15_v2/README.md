# Portugal — core_15

Versão portugal-native-978be68b-v2.1. Apenas dados; motor original 978be68b0e0740deab15ecf536805f323b86fa57.

As nove configurações do catálogo usam quatro bases de 2023, com intervalos de 15 minutos. Os schemas correntes desta família estão em evaluation/. A tarifa de compra é comum: shared/pricing_btn_bi.parquet, em EUR/kWh, sem desconto de mercado. O mercado está desligado nos schemas de dados; os 80% configuram-se depois no schema da experiência, através de community_market.local_price_ratio_to_grid_import. Exportação à rede: zero.

Os catálogos e ficheiros de mobilidade conservam os dados de origem; a execução nativa usa SOC por sessão e não simula o registo longitudinal de viagens. Ver native_compatibility em cada schema. As séries tarifárias alternativas em shared/ e as atribuições source_tariff_id são proveniência, não tarifas ativas. Esta revisão só constrói dados: não foram feitas novas avaliações de utilização.
