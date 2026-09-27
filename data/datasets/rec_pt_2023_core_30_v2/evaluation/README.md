# Catálogo corrente · core_30

Usar exclusivamente os schemas identificados em `../CURRENT_EVALUATION.json`.
As histórias atualizadas estão em [MEMBER_STORIES.md](MEMBER_STORIES.md).
Os ficheiros `lifecycle/` encerram a atividade após saídas definitivas. As identidades dos EV retirados continuam na tabela de observações do simulador existente, mas sem viagens, carregamento ou ligações futuras a carregadores. Os dados encerram a atividade; não removem essas linhas da interface do motor.

O ruído de medição existe apenas em Core-30 Robustness e Premium All-in. `events/*/measurement_noise.parquet` conserva os canais separadamente de `health.parquet`; `observations_and_health.parquet` é a composição consumida pelo simulador existente. São perturbados a carga não controlável, o canal PV de potência e o SOC dos EV e das baterias estacionárias. Desvios padrão: 20 W equivalentes e 0,5 pontos percentuais de SOC; limites de domínio aplicados depois do ruído. Os valores derivados não são perturbados: este é um ensaio de canais selecionados, não de corrupção de toda a informação disponível ao controlador.

A verificação anual corrente e as assinaturas dos ficheiros estão em `Simulator/results/portugal_rec_v2/annual_v23/`. Os nove cenários conservam as definições acordadas; DR só no Premium All-in. O código físico do simulador não foi alterado nesta revisão.
