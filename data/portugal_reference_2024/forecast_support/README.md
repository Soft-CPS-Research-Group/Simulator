# Apoio às previsões — não simulado

96 registos de preço de 01/01/2025 UTC completam os destinos dos últimos horizontes de 2024. Não acrescentam um ano nem um dia à simulação.

`price_components.parquet` conserva OMIE, acessos, TSE e preço completo de apoio. `boundary_forecasts.parquet` identifica as 124 células de previsão, emissão, destino, publicação e origem causal do OMIE. `sources.json` regista URLs e hashes dos originais.

A primeira hora do ficheiro OMIE de 02/01/2025 (Madrid) corresponde a 01/01/2025 às 23h UTC. As restantes horas desse ficheiro são apenas arquivo de origem, não usadas como destinos.

TAR e TSE de 2025 anunciados em dezembro de 2024; margem, CGS, perdas e IEC do contrato de referência conservados. Preços sem IVA nem encargos fixos. O apoio não corrige o recorte interno das previsões derivadas de consumo/PV no motor original.
