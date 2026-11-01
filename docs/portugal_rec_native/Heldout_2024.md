# Portugal v3.3 — passagem de trabalho para a tese

Data: 1 de novembro de 2026. Âmbito: questões 53-F1 e 53-F3 do plano do capítulo 5.
Nenhum ficheiro da tese foi alterado. Esta entrega modifica condições experimentais;
não apresenta resultados anuais de controladores nem resultados de aprendizagem.

## Edição e proveniência

Edição: **portugal-native-2023-2024-v3.3**. Revisão da campanha:
**portugal-heldout-2024-1**. Mantém-se **portugal-boundaries-1** para as fronteiras EV.
O motor de partida desta intervenção é 1.9.11, commit
`8157f605e26fb75651037451fb453b1a48056787`. O trabalho foi isolado das alterações
locais de outra auditoria do simulador; esta extensão não substitui o motor.

Esse mesmo commit preserva integralmente a edição v3.2 anterior. O manifesto
anterior está também em `results/portugal_rec_native/heldout_2024/previous_manifest.json`.
Cada configuração principal conserva o seu schema de origem em
`evaluation/campaign_2024/source_schemas/`. As séries físicas originais não são
sobrescritas: os novos inputs derivados ficam em `evaluation/campaign_2024/inputs/`.
`source_input_checksums.json` identifica os inputs anteriores preservados. A
reconstrução recusa continuar se esses inputs forem alterados sem revisão.

Os PDF, figuras, atlas e tabelas descritivas anteriores continuam a identificar
v3.2. Não devem ser usados como prova dos eventos ou populações de v3.3. A presente
descrição, os catálogos suplementares e a evidência de auditoria são a passagem
de trabalho atual. Os resultados antigos não são reatribuídos à nova edição.

## Bases, configurações e identidades

Mantêm-se as quatro bases físicas originais, com inventários estáticos de 4, 15,
30 e 100 membros, e as nove configurações principais. Core-4, Core-15, Core-30
Normal/Safety/Robustness e Premium Normal conservam os seus inputs físicos.
Core-30 Topology/Combined e Premium All-in recebem a campanha abaixo.

Todos os membros originalmente disponíveis nos dois catálogos dinâmicos já
tinham entrado em 2023; as saídas são permanentes. Por isso, uma nova entrada
efetiva exige uma identidade adicional: **Member_031** em Core-30 e **Member_101**
em Premium. As duas variantes Core-30 partilham a mesma Member_031. Não se
readmite Member_001, Member_008, Member_011 ou qualquer outro membro que saiu.

As novas identidades representam instalações de serviços fictícias sem PV,
bateria, EV ou pedidos de aparelhos controláveis. Reutilizam explicitamente o
arquétipo físico e o contrato elétrico de Member_027 (Core) e Member_088 (Premium),
respetivamente. Isto é reutilização determinística de um perfil para um novo
local sintético, não uma nova observação independente. Antes da admissão, carga,
ocupação, PV e cortes são zero. Depois, há procura efetiva e liquidação comunitária.
Preços, meteorologia e carbono são os mesmos da base. Não se inventam veículos
nem obrigações de serviço para preencher o novo membro.

O inventário base não é aumentado nas configurações estáticas. O conjunto de
identidades possíveis nas configurações dinâmicas passa a 31 e 101, mas os máximos
ativos durante 2024 são 28 e 89. O catálogo suplementar em cada base é
`evaluation/campaign_2024/catalogue.json`, incluindo identidades, origem dos perfis,
contratos, baterias instaladas e inventários cronológicos.

## Calendário reservado de 2024

Todas as horas desta tabela são de Lisboa, com a mudança de hora refletida nos
timestamps UTC dos ficheiros. Os eventos são executados nas datas reais; não
foram deslocados para o início de episódios de teste.

| Data e hora local | Core-30 Topology e Combined | Premium All-in |
|---|---|---|
| 12 fevereiro, 12:00 | Primeira entrada de Member_031 | Primeira entrada de Member_101 |
| 13 maio, 12:00 | Instalação de BESS-M010-01 em Member_010 | Instalação de BESS-M029-01 em Member_029 |
| 20 maio, 12:00 | Retirada de BESS-M014-01 | Retirada de BESS-M014-01 |
| 22 maio, 12:00 | Instalação da substituta BESS-M014-02 | Instalação da substituta BESS-M014-02 |
| 12 agosto, 12:00 | Saída permanente de Member_020 e do carregador associado | Saída permanente de Member_020 e do carregador associado |
| 11 novembro, 12:00 | Reforma permanente de PV-M023-01 | Reforma permanente de PV-M099-01 |

São seis eventos por história, 18 registos nas três configurações principais.
As duas configurações Core partilham a campanha, pelo que não são 18 intervenções
independentes. As referências que mantêm a topologia usam os mesmos seis eventos.
Os 13 eventos Core e os 46 eventos físicos Premium de 2023 permanecem inalterados;
o registo interno Premium que oculta um template no passo zero continua separado.

As baterias novas/substitutas têm 5,12 kWh, 2,56 kW, eficiência 0,94, profundidade
de descarga 0,9 e SOC inicial de 10%. A substituição tem um intervalo real de
48 horas sem bateria, e a nova unidade tem identidade própria. A componente
exógena de standby acompanha a presença da bateria (8 W). A reforma do PV elimina
a produção efetiva e o recurso na interface. Não se reativa equipamento reformado.
No motor atual, a tabela de baterias usa o identificador do slot
`Member_014/electrical_storage`; a identidade física `BESS-M014-02` fica no evento
e no catálogo. O slot desaparece durante a substituição e regressa com uma nova
instância e o SOC de comissionamento, sem se afirmar que o identificador da linha
da interface distingue os dois produtos físicos.

Para analisar continuidade e recuperação, usar janelas com 14 dias antes e 14
dias após cada grupo: 29 janeiro–26 fevereiro; 29 abril–5 junho para o grupo de
maio; 29 julho–26 agosto; 28 outubro–25 novembro, sempre ancoradas às 12:00 locais.
As janelas de maio incluem instalação, retirada, intervalo sem bateria,
substituição e recuperação. Estes recortes não substituem a avaliação anual.

## Populações e recursos

As contagens de aparelhos referem-se a recursos controláveis, não a todos os
aparelhos presentes no inventário descritivo. As identidades exatas constam da
cronologia do catálogo suplementar e das verificações de execução.

| História / instante | Membros | Carregadores | Aparelhos controláveis | PV | Baterias |
|---|---:|---:|---:|---:|---:|
| Core-30 dinâmica, início de 2024 | 27 | 13 | 18 | 9 | 2 |
| Após entrada de fevereiro | 28 | 13 | 18 | 9 | 2 |
| Após instalação de maio | 28 | 13 | 18 | 9 | 3 |
| Durante intervalo de substituição | 28 | 13 | 18 | 9 | 2 |
| Após substituição | 28 | 13 | 18 | 9 | 3 |
| Após saída de agosto | 27 | 12 | 18 | 9 | 3 |
| Após reforma de novembro / fim de 2024 | 27 | 12 | 18 | 8 | 3 |
| Premium All-in, início de 2024 | 88 | 63 | 57 | 33 | 19 |
| Após entrada de fevereiro | 89 | 63 | 57 | 33 | 19 |
| Após instalação de maio | 89 | 63 | 57 | 33 | 20 |
| Durante intervalo de substituição | 89 | 63 | 57 | 33 | 19 |
| Após substituição | 89 | 63 | 57 | 33 | 20 |
| Após saída de agosto | 88 | 62 | 57 | 33 | 20 |
| Após reforma de novembro / fim de 2024 | 88 | 62 | 57 | 32 | 20 |

## Referências emparelhadas

As referências são schemas executáveis em `evaluation/references/`, relativos à
respetiva base. Não são novos controladores BAU/RBC. Um candidato e o seu
controlador de referência continuam a executar **o mesmo schema** e a mesma
história; os pares abaixo servem para estudar fatores experimentais.

| Cenário | Referência (nome do ficheiro, sem `.json`) | Única intervenção removida |
|---|---|---|
| Core-30 Combined | `core_30_combined_no_disturbances` | Campanha conjunta de health e cortes; mantém reconfiguração e limites apertados |
| Core-30 Combined | `core_30_combined_nominal_limits` | Aperto adicional dos contratos elétricos; mantém reconfiguração, health e cortes |
| Premium All-in | `premium_100_allin_no_disturbances` | Campanha conjunta de health, ruído e cortes; mantém reconfiguração e DR |
| Premium All-in | `premium_100_allin_no_dr` | Ativação de DR; conserva os pedidos como registo do contrafactual, além de reconfiguração e perturbações |
| Core-30 Topology | `core_30_topology_frozen_2024` | Apenas os eventos de reconfiguração de 2024; mantém toda a história de 2023 |
| Premium All-in | `premium_100_allin_frozen_2024` | Apenas os eventos de reconfiguração de 2024; mantém a história de 2023, as perturbações e DR |

Não se criam novas bases nem se duplicam variantes para cada falha. As duas
referências congeladas são necessárias porque as histórias Core/Premium são
distintas. Topology já fornece o cenário Core sem perturbações nem aperto de
limites; não é uma referência adequada para remover apenas perturbações de
Combined, porque também removeria o fator elétrico.

Retirar cortes significa pôr `power_outage=0` e repor o PV que a construção
original tinha suprimido exclusivamente por falta de backup durante esses cortes,
respeitando a presença do membro e do PV. Esta reposição é consequência do fator
retirado; não é alteração meteorológica. Todas as outras colunas são iguais.
A referência elétrica repõe os contratos de Member_001, Member_006 e Member_025;
Member_001 já saiu antes de 2024, pelo que o contraste elétrico reservado atua
efetivamente sobre os dois membros ainda presentes. Não se eliminam limites de
segurança: repõem-se os limites nominais publicados.

As referências sem perturbações estimam um contraste de **campanha conjunta**.
Não permitem atribuir o efeito a uma única falha. Um diagnóstico individual deve
partir do mesmo cenário, remover apenas os IDs e canais dessa falha e verificar
novamente o pareamento. Não foram criados ou executados esses contrastes
individuais, pois não existe aqui uma pergunta de atribuição selecionada.

Nos contrastes de reconfiguração, as condições e a comunidade ativa coincidem
até ao primeiro evento de 2024. Depois, participação e recursos são precisamente
o fator estudado: os totais absolutos deixam de representar o mesmo universo.
Usar a exposição completa de cada história, os membros comuns e as coortes de
serviço identificadas, declarando o contrafactual sem campanha. Não impor
artificialmente a mesma população nem interpretar menor procura como melhor controlo.
Premium mantém as mesmas seeds/configurações estocásticas; uma topologia diferente
pode alterar a ordem de consumo de números aleatórios do motor. Isto não prova
ruído realizado idêntico por entidade depois da divergência. O contraste Core
Topology não tem esse fator de ruído.

## Serviços, fronteiras e denominadores

A saída de Member_020 foi colocada num intervalo sem visitas em curso em ambas
as bases, verificado sobre todas as visitas dos seus veículos, incluindo visitas
noutros membros. Não há aparelhos controláveis associados a esse membro. Os
pedidos anteriores à saída mantêm identidades, duração, SOC e objetivos. Depois
da saída não existe carregamento privado dentro desta REC.

| História | Visitas oferecidas na trajetória anterior | Visitas na nova trajetória | Excluídas pela saída de 2024 | Visitas em curso cortadas |
|---|---:|---:|---:|---:|
| Core-30 Topology | 8 587 | 8 481 | 106 | 0 |
| Core-30 Combined | 8 587 | 8 481 | 106 | 0 |
| Premium All-in | 40 072 | 39 982 | 90 | 0 |

Estas são contagens de visitas oferecidas nos dois anos pelas histórias
executáveis, não o total do catálogo estático nem resultados de cumprimento.
As visitas excluídas representam pedidos de energia de referência de cerca de
485,24 kWh em cada variante Core e 694,12 kWh em Premium. Esses valores são
pedidos de input, não energia consumida nem poupada. As variantes Core reutilizam
as mesmas visitas e não devem ser somadas como amostras independentes.

Os ficheiros `*_service_cohorts.parquet` em `results/portugal_rec_native/heldout_2024/`
retêm uma linha por visita do catálogo, com identidade, intervalos, pedido,
presença nas duas trajetórias e motivo de exclusão. As exclusões anteriores à
campanha ficam noutra categoria. Visitas terminais sem partida conhecida
continuam com `right_censored=true`, sem as converter em partidas ou sucessos.
Todos os horários e perfis de aparelhos controláveis permanecem iguais.

As quatro referências que mantêm a reconfiguração têm exatamente a mesma
população de serviços do seu cenário. As referências congeladas conservam a
população anterior; por isso, a comparação deve publicar os denominadores de
cada trajetória e a coorte comum. A energia dos membros presentes não deve ser
dividida silenciosamente pelo inventário completo de 30/100, nem pelo conjunto
de 31/101 identidades possíveis.

## Separação temporal e estado

Mantêm-se 2023 para preparação/seleção e 2024 para avaliação reservada. As quatro
quinzenas de validação de 2023 e os restantes intervalos de treino não mudam.
Todos os eventos e inputs executáveis de 2023 são preservados. Os nomes legados
`development`/`confirmation` dos programas de perturbação não autorizam o uso dos
seus registos de 2024 para selecionar políticas, normalizações ou hiperparâmetros.
O calendário reservado pertence ao avaliador, não à informação futura do agente.

Usar para preparação os nove schemas em **`evaluation/preparation_2023/`**, listados
em `preparation_schemas` no manifesto. São vistas temporais das mesmas nove
configurações, não novas variantes experimentais. Limitam a execução ao passo
35 039, conservam o conjunto original de membros e removem os programas de
perturbação/DR de 2024 dos ficheiros selecionados. As séries físicas continuam
partilhadas, e o carregamento respeita a janela de execução; normalizadores e
seleção de dados externos também têm de respeitar esse limite e os blocos de validação.

Não usar diretamente o schema completo de avaliação durante a preparação: o
motor inclui nos metadados de ciclo de vida as identidades pré-carregadas ainda
inativas. As observações e ações ativas antes da primeira mudança coincidem com
o contrafactual, mas esses metadados de catálogo não são cegos. As vistas de 2023
excluem as identidades suplementares e os eventos reservados, incluindo nos
metadados expostos pelo motor. Este ponto foi verificado explicitamente.

O schema completo continua de 0 a 70 175: nenhuma reinicialização foi introduzida
em janeiro. Para transportar SOC e outras condições físicas entre anos é preciso
executar continuamente ou restaurar um checkpoint equivalente. Iniciar um episódio
diretamente em 2024 reconstrói a topologia anterior, mas não simula o estado físico
omitido. Os testes curtos de eventos são testes de integridade, não prova de
continuidade energética de um replay anual. Candidato e referência devem usar
a mesma convenção de arranque, checkpoint, período avaliado e seeds de execução.

## Verificações e ficheiros para integração editorial

- `scripts/prepare_portugal_heldout_2024.py`: transformação offline reproduzível,
  com deteção de alteração dos inputs de origem; não importa CityLearn.
- `scripts/audit/audit_portugal_heldout_2024.py`: comparação automática dos seis
  pares, identidades, disponibilidade, fronteiras, coortes e execução das janelas.
- `tests/test_portugal_heldout_2024.py`: regressões de todos os 18 eventos,
  referências, preservação da história e mudanças reais das interfaces.
- `results/portugal_rec_native/heldout_2024/campaign.json`: eventos com passos e
  timestamps, correspondência dos pares, inventários exatos e linhagem dos inputs.
- `results/portugal_rec_native/heldout_2024/validation.json`: resultados correntes,
  snapshots de execução, eventos efetivamente aplicados e limites da verificação.
- `evaluation/campaign_2024/catalogue.json`: suplemento de membros/equipamentos
  por base, sem alterar as tabelas de composição estática originais.

A auditoria de entrega usa 72 janelas: todos os novos eventos dos cenários e das
referências dinâmicas, mais a passagem de ano e o fim do período nas 15
configurações principais/de referência. Verifica IDs dos recursos, interfaces de
observação/ação, eventos aplicados, recompensas finitas, mercado a 80%, balanço
das trocas e exportação de KPIs. As janelas dos eventos têm 12 horas antes/depois;
não substituem as janelas experimentais de 14 dias nem a campanha anual.
O estado efetivo e as contagens executadas devem ser lidos em `validation.json`.

Os testes gerais de fronteiras e preços continuam necessários. A verificação dos
pares não aceita alterações adicionais de seeds, preços, previsões, meteorologia,
serviços, proteções não estudadas ou condições iniciais. O pedido DR excessivo
de 1500 kW por quatro horas mantém a categoria `impossible_stress`, separada dos
pedidos operacionais. O ficheiro de pedidos não é alterado na referência sem DR.
Os pares de perturbações, limites e DR retiram o fator declarado na trajetória
completa de dois anos; não representam automaticamente uma intervenção iniciada
apenas em janeiro de 2024. Os mesmos estados iniciais referem-se ao início comum
da execução. Num replay contínuo, diferenças posteriores de estado causadas pelo
fator são parte do contraste. Para atribuir apenas o efeito de ativação num
intervalo reservado, usar um checkpoint comum e declarar esse recorte adicional.

## Limitações e tarefas que permanecem fora do pedido

Não houve treino nem uma nova campanha anual de desempenho. As verificações
individuais dos pedidos EV não provam viabilidade conjunta sob limites partilhados.
Os novos membros reutilizam arquétipos explicitamente, sem acrescentar evidência
estatística independente. Não foi construída uma referência de informação
perfeita nem um novo controlador convencional.

Permanecem para a tese a integração editorial de 53-F1/53-F3, atualização das
tabelas/figuras e reconciliação dos capítulos 6/7. Continuam em aberto o isolamento
do efeito do tamanho (53-F2), a escolha de experiências de transferência (53-F4)
e a seleção de diagnósticos individuais quando houver uma pergunta de atribuição
(53-F5). Nenhum desses resultados deve ser inferido da presente entrega.
