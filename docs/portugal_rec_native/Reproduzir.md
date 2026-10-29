# Reconstruir 2023 e 2024

Usar o motor original `978be68b0e0740deab15ecf536805f323b86fa57` e Python com numpy, pandas, pyarrow e pvlib 0.15.2. Nenhum construtor importa CityLearn. As fontes públicas de ambos os anos estão arquivadas no repositório; a reconstrução abaixo funciona sem rede.

```bash
mkdir -p /tmp/pt-source
tar -xzf data/portugal_native_source_v23.tar.gz -C /tmp/pt-source
python scripts/build_portugal_native_suite.py --source /tmp/pt-source --output /tmp/pt-2023 --tariff common
python scripts/prepare_portugal_reference_2024.py
python scripts/extend_portugal_2024.py --source /tmp/pt-2023 --output /tmp/pt-2023-2024
python scripts/portugal_lifecycle_stories.py --repo /tmp/pt-2023-2024
python scripts/audit/verify_portugal_two_year_inputs.py --repo /tmp/pt-2023-2024 --source /tmp/pt-2023 --engine .
python scripts/apply_portugal_omie_pricing.py --repo /tmp/pt-2023-2024 --reference-root .
python scripts/audit/verify_portugal_forecasts_stories.py --repo /tmp/pt-2023-2024
python scripts/repair_portugal_dataset_boundaries.py --repo /tmp/pt-2023-2024
```

A preparação de fontes escreve apenas `data/portugal_reference_2024`. A meteorologia histórica de 2023 é lida das primeiras 35040 linhas do catálogo existente para inicializar previsões causais. A extensão recusa substituir uma pasta de saída já existente; usar uma pasta nova. Os diretórios de entrada e saída devem estar separados.

O primeiro construtor recria a edição de 2023. O segundo acrescenta 2024 sobre os mesmos quatro diretórios e nove schemas. Os caminhos históricos conservam «2023» mas contêm os dois anos. A alteração manual de constantes de ano não é suficiente: não respeitaria feriados, hora legal, fontes, saídas nem janelas entre anos.

A verificação é estática; as sessões EV usam um limite físico calculado com os parâmetros declarados, sem executar uma política. Os comprovativos antigos das duas visitas do Café 028 estão em `results/portugal_rec_native/frozen_2023_evidence/` e aplicam-se à edição congelada. O pacote de 2023 e o commit `393cd200` preservam essa edição integral.

A extensão gera automaticamente as narrativas e o plano de ciclo de vida por configuração. A ferramenta de histórias publica o atlas a partir do catálogo. A auditoria de previsões produz o relatório e as figuras (requer matplotlib); com `--previous /caminho/para/a/edicao-v3` verifica também a preservação de todos os perfis físicos e dos preços anteriores à fronteira. O dia de apoio e as fontes de janeiro estão arquivados em `data/portugal_reference_2024/forecast_support`; a reconstrução não depende de downloads.

## Edição corrente v3.2

A auditoria dos inputs de dois anos acima aplica-se à construção histórica, antes da conversão tarifária. O passo final `apply_portugal_omie_pricing.py` substitui os preços e ativa o mercado comunitário. É idempotente, usa fontes arquivadas e não altera dados físicos. Para atualizar o checkout corrente, executar esse passo sem argumentos. Validar com `python -m pytest -q tests/test_portugal_omie_pricing.py` e `python scripts/audit/audit_portugal_windows.py`. Esta auditoria executa os schemas com a configuração de mercado já gravada, sem a substituir.

### Revisão de integridade `portugal-boundaries-1`

Executar `repair_portugal_dataset_boundaries.py` por último, incluindo após reconstruções. Este passo idempotente elimina cortes fora da presença dos membros e corrige as 54 sessões domésticas que continuam no fim do período observado. O gerador histórico usa o fim do ano como limite do estacionamento, sem gerar a viagem seguinte: esse limite não é uma partida prevista. O catálogo conserva o intervalo em `observed_end_time_step` (exclusivo), assinala `right_censored=true` e deixa `departure_time_step` nulo. As restantes sessões conservam partidas conhecidas. Nos horários executáveis, a contagem até à partida fica nula apenas nessas sessões; chegada, SOC e objetivo mantêm-se. Não se prolonga o calendário nem se presume uma partida em janeiro. As sessões parciais não entram na avaliação de cumprimento à partida, mas a sua energia continua contabilizada.

Usar o simulador corrente com o tratamento de partida desconhecida (`-1`) nos KPIs e recompensas; o commit histórico indicado no início serve para reconstrução e não para executar esta revisão. Validar com `python -m pytest -q tests/test_portugal_dataset_boundaries.py tests/test_kpi_v2.py tests/test_portugal_omie_pricing.py`. Os comprovativos históricos mantêm o âmbito da edição que validaram. Os relatórios atuais estão em `results/portugal_rec_native/boundary_integrity.json` e `boundary_validation.json`. Depois de validar, `python scripts/package_portugal_native.py` atualiza o ZIP e os seus checksums.
