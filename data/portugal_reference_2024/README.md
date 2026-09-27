# Fontes Portugal 2024

Fontes públicas arquivadas para construção offline. A aquisição está descrita em `static_provenance.json` e nos envelopes de cada resposta (URL incluído). OMIE/REN cobrem todos os dias requeridos, sem pedidos falhados. Preços horários OMIE são alinhados de Madrid para UTC; REN usa os dias de Lisboa e a sua mudança de hora. O ficheiro E-REDES contém 92/100 intervalos nos dias DST, preservados em ordem cronológica.

Reproduzir os alinhamentos com `scripts/prepare_portugal_reference_2024.py`. Ver `aligned_reference_metadata.json`, `weather_gap_fills.json` e `docs/portugal_rec_native/Extensao_2024.md` para fórmula, método e limites. Os fatores de carbono são estimativas derivadas; a calibração térmica continua a ser a de 2023. Os preços de 2024 não foram artificialmente truncados a zero.
