# Reconstruir 2023 e 2024

## Revisão corrente de realismo

Depois da construção histórica descrita abaixo, executar no checkout corrente:

```bash
python scripts/revise_portugal_realism.py --apply
python -m pytest -q tests/test_portugal_realism.py tests/test_portugal_ev_mobility.py tests/test_portugal_empirical.py tests/test_portugal_commercial_appliances.py tests/test_portugal_dataset_boundaries.py tests/test_portugal_heldout_2024.py tests/test_portugal_omie_pricing.py tests/test_ev_arrivals.py tests/test_ev_soc_behavior.py tests/test_ev_feasibility_replay.py
python scripts/audit/portugal_empirical_validation.py
python scripts/audit/portugal_commercial_appliance_validation.py --evaluate --seeds --figure
python scripts/audit/audit_portugal_ensembles.py
python scripts/audit/audit_portugal_realism.py --runtime
python scripts/audit/audit_portugal_heldout_2024.py --runtime
python scripts/audit/audit_portugal_windows.py --output results/portugal_rec_native/accounting_validation.json
```

Este passo inclui a revisão EV e substitui a repetição de eventos entre anos.
Lê a origem diretamente do commit `c54b13b62823fb1256397ea98cb37a830a24124d`;
o histórico desse commit tem de estar disponível (a geração não funciona num
clone superficial que o omita). Os testes que comparam a revisão com os horários
e inventários originais também precisam desse commit; num clone superficial,
obtê-lo com `git fetch --no-tags --depth=1 origin c54b13b62823fb1256397ea98cb37a830a24124d`.
A fonte adicional de humidade de 2023 está incluída no repo.
Não executar novamente os construtores históricos por cima destes resultados
sem repetir a revisão. `--seed` e `--config` permitem ensembles explícitos,
mantendo a mesma realização nos braços de cada comparação.
Se a auditoria contabilística for interrompida, `--resume` reaproveita apenas
janelas completas com a mesma duração e a mesma impressão SHA-256 dos dados,
schemas e motor. Uma alteração desses inputs obriga a nova execução.
As sementes podem retomar com `--resume`; as janelas de fronteiras com
`--resume-runtime` e as sondas de falhas com `--reuse-runtime`, sempre após
verificação dos inputs. As fontes empíricas compactas já estão incluídas;
`portugal_empirical_validation.py --acquire` permite readquiri-las com validação
dos hashes publicados. `--calibrate-load` é uma etapa de desenvolvimento:
não se deve repetir o ajuste depois de consultar as casas reservadas e continuar
a apresentar essas casas como uma avaliação independente.

As referências BDG2, REFIT e FIKElectricity são readquiridas, se necessário, com
`portugal_commercial_appliance_validation.py --acquire-commercial --acquire-appliances --acquire-kitchens`.
Os ficheiros grandes permanecem apenas em memória; retêm-se resumos compactos.
`--regenerate-and-reproduce` executa duas gerações completas e exige igualdade
byte a byte de todos os Parquet, sem criar cópias ou arquivos.
`--baseline` e `--freeze-protocol` são passos únicos de desenvolvimento já
concluídos; não os repetir para redefinir os grupos reservados. Não reajustar
parâmetros após consultar esses grupos e alegar uma nova avaliação independente.

Ver [Fontes e método](Fontes_e_metodo.md) para hipóteses, evidência científica e
o que ainda falta validar empiricamente. Não é necessário gerar ZIPs nem cópias
locais para esta revisão. Os passos de empacotamento mais abaixo são opcionais
e históricos; não foram executados nesta entrega.

## Último passo para a edição corrente v3.3

Depois de construir v3.2 e aplicar `repair_portugal_dataset_boundaries.py` na ordem
abaixo, executar:

```bash
python scripts/prepare_portugal_heldout_2024.py --repo /tmp/pt-2023-2024
python scripts/audit/audit_portugal_heldout_2024.py --repo /tmp/pt-2023-2024 --runtime
```

No checkout corrente, omitir `--repo`. A preparação é offline e idempotente;
verifica os checksums dos inputs anteriores, arquiva os schemas v3.2 e cria séries
derivadas apenas onde a campanha o exige. Nunca sobrescreve as séries físicas
anteriores. Não executar novamente os construtores históricos sobre v3.3 sem
voltar a aplicar este último passo. A auditoria usa o motor do checkout em que o
script é executado; `--repo` seleciona os dados. Usar o motor corrente, nunca o
commit histórico de construção como substituto do motor atual.
As vistas de preparação ficam em `evaluation/preparation_2023/`, sem os novos
membros nem os programas de 2024. Manter os blocos de treino/validação dentro de
2023; não alimentar a preparação com os schemas completos de avaliação.

`python -m pytest -q tests/test_portugal_heldout_2024.py` mantém a verificação dos
pares e de todos os novos eventos no CI. A auditoria de entrega percorre 72 janelas
reais de eventos/fronteiras, sem deslocar datas para um episódio artificial.
`python scripts/package_portugal_native.py` exige essa evidência e os checksums
atuais antes de reconstruir o pacote.

A edição anterior pode ser recuperada integralmente no commit
`8157f605e26fb75651037451fb453b1a48056787`. O PDF e as tabelas descritivas existentes
são históricos de v3.2; a descrição de v3.3 está em [Heldout_2024.md](Heldout_2024.md).

## Construção histórica de v3.2

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

A verificação é estática; as sessões EV usam um limite físico calculado com os parâmetros declarados, sem executar uma política. Os [comprovativos antigos das duas visitas do Café 028](https://github.com/Soft-CPS-Research-Group/Simulator/tree/8157f605e26fb75651037451fb453b1a48056787/results/portugal_rec_native/frozen_2023_evidence) estão preservados no histórico Git e aplicam-se à edição congelada. O pacote de 2023 e o commit `393cd200` preservam essa edição integral.

A extensão gera automaticamente as narrativas e o plano de ciclo de vida por configuração. A ferramenta de histórias publica o atlas a partir do catálogo. A auditoria de previsões produz o relatório e as figuras (requer matplotlib); com `--previous /caminho/para/a/edicao-v3` verifica também a preservação de todos os perfis físicos e dos preços anteriores à fronteira. O dia de apoio e as fontes de janeiro estão arquivados em `data/portugal_reference_2024/forecast_support`; a reconstrução não depende de downloads.

## Edição corrente v3.2

A auditoria dos inputs de dois anos acima aplica-se à construção histórica, antes da conversão tarifária. O passo final `apply_portugal_omie_pricing.py` substitui os preços e ativa o mercado comunitário. É idempotente, usa fontes arquivadas e não altera dados físicos. Para atualizar o checkout corrente, executar esse passo sem argumentos. Validar com `python -m pytest -q tests/test_portugal_omie_pricing.py` e `python scripts/audit/audit_portugal_windows.py`. Esta auditoria executa os schemas com a configuração de mercado já gravada, sem a substituir.

### Revisão de integridade `portugal-boundaries-1`

Executar `repair_portugal_dataset_boundaries.py` por último, incluindo após reconstruções. Este passo idempotente elimina cortes fora da presença dos membros e corrige as 54 sessões domésticas que continuam no fim do período observado. O gerador histórico usa o fim do ano como limite do estacionamento, sem gerar a viagem seguinte: esse limite não é uma partida prevista. O catálogo conserva o intervalo em `observed_end_time_step` (exclusivo), assinala `right_censored=true` e deixa `departure_time_step` nulo. As restantes sessões conservam partidas conhecidas. Nos horários executáveis, a contagem até à partida fica nula apenas nessas sessões; chegada, SOC e objetivo mantêm-se. Não se prolonga o calendário nem se presume uma partida em janeiro. As sessões parciais não entram na avaliação de cumprimento à partida, mas a sua energia continua contabilizada.

Usar o simulador corrente com o tratamento de partida desconhecida (`-1`) nos KPIs e recompensas; o commit histórico indicado no início serve para reconstrução e não para executar esta revisão. Validar com `python -m pytest -q tests/test_portugal_dataset_boundaries.py tests/test_kpi_v2.py tests/test_portugal_omie_pricing.py`. Os comprovativos históricos mantêm o âmbito da edição que validaram. Os relatórios atuais estão em `results/portugal_rec_native/boundary_integrity.json` e `boundary_validation.json`. Depois de validar, `python scripts/package_portugal_native.py` atualiza o ZIP e os seus checksums.
