# Reconstruir os datasets

Usar o Simulator original `978be68b0e0740deab15ecf536805f323b86fa57`.
O construtor apenas prepara ficheiros, sem importar, modificar ou executar CityLearn.
Requer Python, pandas, numpy e pyarrow; ver `scripts/requirements-portugal-generator.txt`.

O arquivo `data/portugal_native_source_v23.tar.gz` contém os inputs congelados
anteriores à adaptação e o manifesto de origem. É fonte de reconstrução, não
outro catálogo ativo. Extrair numa pasta separada do repositório:

```bash
mkdir -p /tmp/portugal-native-source
tar -xzf data/portugal_native_source_v23.tar.gz -C /tmp/portugal-native-source
python scripts/build_portugal_native_suite.py   --source /tmp/portugal-native-source --output . --tariff common
```

O comando substitui as quatro famílias correntes e os manifestos gerados.
`common` é a única política tarifária desta revisão. Não modifica o motor.
Fazer a reconstrução com os datasets fora de uso. Os ficheiros de checksums
identificam cada ficheiro publicado; o manifesto contém os hashes dos schemas.

A revisão corrente foi verificada estaticamente: série comum completa e finita,
igualdade entre famílias, preservação dos dados físicos, metadados tarifários,
mercado desligado nos schemas de dados e integridade do motor original.
Não foi realizada uma nova campanha de simulação ou avaliação de algoritmos.
O comprovativo de publicação está em `results/portugal_rec_native/static_integrity.json`.
Os resultados individuais anteriores são históricos, conforme [Histórico](Historico.md).

Os ficheiros de eventos de confirmação existentes continuam reservados nos
mesmos datasets. Não foram usados nesta alteração. O caminho nativo de health
está em `experimental_design.native_confirmation_robustness_file`.

## Verificação das duas sessões corrigidas

O construtor aplica a correção documentada em `native_input_corrections.json`:
duas visitas ao Café 028 passam de 75 para 90 minutos, com os mesmos SOC
solicitados e potências físicas. Para repetir apenas os dez ensaios curtos
(duas visitas em cinco variantes Core-30):

```bash
python scripts/audit/verify_portugal_core30_sessions.py --repo .
```

Esta verificação usa ações fixas para demonstrar uma trajetória viável, sem
comparar algoritmos. O resultado está em `core30_session_feasibility.json` e
o traço em `core30_session_witness_trace.parquet`, na pasta de resultados nativos.
A evidência fica restrita a essas sessões; não é uma certificação anual.
As séries `mobility/reference_soc` conservam o diagnóstico de origem; não são
a trajetória longitudinal executada pelo motor original.
