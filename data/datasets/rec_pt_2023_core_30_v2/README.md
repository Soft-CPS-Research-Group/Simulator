# Portugal — 2023 e 2024

Mesma comunidade e equipamentos, 70176 registos de 15 minutos.
Os caminhos com «2023» conservam o nome histórico; os inputs correntes abrangem os dois anos.
Os schemas em evaluation/ são a fonte executável. As saídas de membros/ativos
mantêm-se definitivas; as histórias de topologia incluem a campanha reservada de 2024.
Ver extension_2024.json e docs/portugal_rec_native/Extensao_2024.md.
Os campos annual_* dos catálogos e os ficheiros antigos em events/ são proveniência de 2023,
não contagens atualizadas nem eventos executados. O mercado comunitário está ativo: preço local = 80% do preço de compra à rede; exportação à rede = zero.
## Preços OMIE e mercado comunitário

Versão portugal-native-2023-2024-v3.2. Série ativa: `shared/pricing_omie.parquet`. Compra à rede = máximo(OMIE português em EUR/MWh / 1000, 0), em EUR/kWh. Sem margem, perdas, MIBEL, TAR, taxas ou impostos. Previsões a 1/6/24 h seguem a mesma regra, com OMIE publicado ou persistência diária. As referências de retalho em sources/ são apenas históricas.

## Revisão de realismo corrente

`portugal-realism-4` / `portugal-ev-mobility-1`: atividade partilhada, ciclos variáveis, mobilidade contínua de referência e programas de falha distintos por ano. Ver `realism_revision.json` e `ev_mobility_revision.json` para configuração, distribuições e hashes; `catalogs/household_activity.parquet` para o calendário. As narrativas e valores de dimensionamento antigos são proveniência, não a descrição operacional desta revisão. Os parâmetros de variabilidade do consumo foram ajustados em sete casas SHEERM; seis outras casas são reservadas para avaliação independente. Os restantes pressupostos e desvios estão identificados em `results/portugal_rec_native/empirical_validation.json`. Método e limites em `docs/portugal_rec_native/Fontes_e_metodo.md`.
