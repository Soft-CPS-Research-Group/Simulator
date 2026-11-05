# Fontes e método · Portugal REC

## Revisão corrente de realismo — 5 de novembro de 2026

`portugal-realism-4` revê os dois anos das quatro bases. Substitui as hipóteses
temporais antigas descritas mais abaixo. É um gerador de cenários sintéticos
informado por fontes públicas; **ainda não é um modelo calibrado e validado da
população portuguesa**. Os testes de coerência física não substituem essa validação.

O código e os parâmetros estão em `scripts/portugal_activity.py`,
`scripts/portugal_stochastic_demand.py`, `scripts/portugal_commercial_activity.py`,
`scripts/revise_portugal_realism.py` e `scripts/portugal_realism_config.json`.
O último ficheiro identifica cada fonte, a sua função e o estatuto dos parâmetros.
A transferência de referências estrangeiras fica explícita: programas domésticos
usam REFIT; os comerciais usam âncoras BDG2. Não são estimativas populacionais
portuguesas. ACN-Data continua apenas como referência metodológica.

### Fundamentação e hipóteses

| Componente | Base científica ou observada | Implementação e limite |
|---|---|---|
| Atividade doméstica | [Richardson et al., CREST](https://doi.org/10.1016/j.enbuild.2010.05.023); [Chen et al., ResStock](https://arxiv.org/abs/2111.01881) | Um calendário latente partilhado condiciona presença, deslocações e usos elétricos. Adota-se o princípio de dependência comportamental; este código não reproduz as matrizes de transição desses modelos. |
| Variabilidade do consumo | [SHEERM, dados v3](https://doi.org/10.5281/zenodo.13972210) | Ficheiros `raw` de 13 casas portuguesas: sete para desenvolvimento e seis reservadas para avaliação. Ajuste de estatísticas agregadas; não identifica separadamente aparelhos ou ocupantes. |
| Aparelhos | [REFIT, Murray et al.](https://doi.org/10.1038/sdata.2016.122), medições britânicas por aparelho | 48 variantes por aparelho doméstico/copa, obtidas de candidatos medidos nas casas 3/5; casas 2/11 reservadas. Energia e duração amostradas conjuntamente; horários/prazos continuam a ser pressupostos. Não se transfere essa validação para máquinas profissionais. |
| Carregamento | [ACN-Data](https://ev.caltech.edu/assets/pub/ACN_Data_Analysis_and_Applications.pdf); [Electric Nation](https://eatechnology.com/media/girhcnsc/electric-nation-customer-trial-report.pdf) | Dependência entre local, duração, energia e hábitos não diários. ACN-Data é sobretudo laboral e não fornece SOC real de chegada; Electric Nation inclui BEV/PHEV britânicos. Os pesos dos perfis e SOC são hipóteses. |
| Continuidade elétrica | [E-REDES 2024, tabela 3.10](https://www.e-redes.pt/sites/eredes/files/2025-06/Relat%C3%B3rio%20da%20Qualidade%20de%20Servi%C3%A7o%202024_v2_0.pdf) | Frequência/duração esperadas da coorte exposta ancoradas nos totais BT publicados: 1,33 interrupções e 70,57 minutos/cliente. Exclui o stress imposto; a decomposição local/comum e a distribuição de durações continuam a ser hipóteses. |
| Meteorologia | PVGIS e [ERA5 via Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api) | Humidade de reanálise nos dois anos, substituindo a fórmula artificial de 2023. O novo pedido bruto de 2023 fica retido, com URL, em `data/portugal_reference_2023/era5_open_meteo_2023.json.gz` (cerca de 30 kB). Interpolação horária não equivale a medição a 15 minutos. |

Os parâmetros ficam congelados na configuração antes da avaliação dos
controladores. Fins de semana fora usam contagem Poisson com propensão individual
Gamma(forma 1,5; escala 1,5). Cada ponte elegível tem probabilidade 0,22;
viajar durante as férias tem probabilidade 0,8, com duração e início variáveis.
Estes valores são escolhas de desenho sujeitas a análise de sensibilidade.
As ausências sobrepostas fundem-se; o mesmo episódio informa todos os usos.
O início das férias varia entre anos (desvio normal de 4 dias, limitado a ±10)
e a duração varia em ±2 dias. Horários profissionais têm preferências pessoais,
variação diária e exceções de 2,5%; a rotina de transporte escolar tem duas
deslocações nos dias letivos. Estes parâmetros são priors, não medições de viagens.
As viagens no destino têm probabilidade diária 0,6, em vez de energia nula em
todos os dias intermédios. O calendário inclui hora civil, feriados e dias de
92/100 intervalos nas mudanças de hora.

O teletrabalho híbrido parte de terças/quintas, com exceção diária de 8% nos
dias úteis, e passa a ser comum à mobilidade e consumo. Elimina-se a antiga
regra de consumo `(dia_do_ano + membro) % 5`. `occupant_count` passa a indicar
**presença esperada**, podendo ser fracionário; não é uma contagem medida de
pessoas. Durante uma ausência familiar prolongada é zero. Quando o condutor
sai sozinho, adota-se presença esperada de 50% dos restantes residentes.

A energia doméstica de base conserva o orçamento de 2023; em 2024 recebe
variação lognormal por membro com sigma 0,04, em vez de todos os membros terem
exatamente o fator 366/365. Aquecimento e arrefecimento continuam a ser proxies
elétricos condicionados pelo clima e presença, não um modelo térmico validado.
Na água quente, 12% do orçamento médio são perdas constantes e os restantes
consumos dependem da presença. Não se repõem artificialmente, noutros dias,
os banhos que deixaram de ocorrer durante uma ausência.

A revisão empírica substitui a base demasiado suave por episódios de consumo
com chegadas Poisson compostas e persistência entre intervalos. O nível diário
tem sigma lognormal 0,35 e persistência 0,35; o componente episódico tem peso
0,65, taxa nominal de 24 episódios/dia e persistência 0,7. A forma média sazonal
e civil da casa mantém-se, com desvio diário dos horários de 0,75 h. Um fator
de atividade partilhado também modula a utilização dos aparelhos. São estados
agregados a 15 minutos, não formas de onda de equipamentos medidos. A energia
anual de base é conservada; episódios que atingem o contrato elétrico originam
redistribuição conservativa dentro do ano, sem corte de energia. Isto é uma
construção offline condicionada ao inventário, não uma política de controlo.

Os programas domésticos e das copas de pessoal passam a usar 48 variantes por
aparelho. Primeiro sorteia-se uma das duas casas REFIT de desenvolvimento com
probabilidade igual e depois um candidato detetado dessa casa. Conservam-se
conjuntamente a forma energética, a duração e a energia do candidato; aplica-se
uma perturbação energética lognormal pequena, sigma 0,08, limitada a 0,75–1,25.
A integração para 15 minutos conserva energia. Se necessário, redistribui-se
energia dentro do programa para respeitar a potência nominal instalada; só se
estende a duração quando o total excede a capacidade energética da janela.
Frequência e prazos são pressupostos do inventário, não horários medidos REFIT.
As lavandarias profissionais usam 48 variantes de engenharia, com referência
à ficha portuguesa Miele PWM 912 (12 kg, água fria, 60 °C: 1,68 kWh e 49 min
numa máquina com aquecimento de 9 kW). A potência instalada permanece 4,6 kW;
a duração inclui a penalização de aquecimento mais lento, arredondada para
15 minutos. O modelo assume entrada de água a 15 °C, 0,25 kWh de componente
motor/auxiliar e escala a restante energia com a massa e o salto térmico;
essa decomposição é uma aproximação de engenharia, não uma medição Miele. Carga parcial, mistura de temperaturas, forma temporal e frequência
são pressupostos, não medições de campo. Não se usa o menor consumo elétrico
da ligação a água quente sem contabilizar a energia térmica externa.

As janelas variam por utilização, respeitam abertura/presença e não deixam
pedidos pendentes para além de uma ausência prolongada. Os aparelhos fixos
entram exatamente uma vez no consumo não deslocável; os controláveis ficam
fora dele e entram quando executados pelo simulador. A máquina de restauração
passa a ter lotes agregados de 15–45 minutos, com energia mediana-prior de
0,45 kWh e taxa-prior de 5,5 pedidos por dia de serviço, condicionada pela
atividade. São hipóteses de lotes à resolução do simulador, não ciclos
individuais de poucos minutos identificados nas medições. Aquecimento inicial
(0,5–1 kWh/dia) e manutenção quente (0,10–0,25 kW durante abertura) entram em
`professional_service_kwh`, não nos pedidos deslocáveis. Esse consumo é
reclassificado do orçamento de base do estabelecimento, evitando contagem dupla.

A energia anual de condução de cada EV mantém-se igual ao orçamento original.
As distâncias são, por isso, equivalentes energéticos após normalização, não
quilometragens medidas. O SOC é calculado continuamente no testemunho de
geração, incluindo carga pública/destino explícita e finita em viagens longas,
perdas, taper e margem elétrica para os aparelhos. A antecipação de cargas
necessárias é um testemunho offline de viabilidade, **não um modelo validado
de previsão perfeita dos condutores**. O motor continua a receber SOC prescrito
em cada chegada; os balanços de viagem são evidência do gerador.
As decisões de ligação passam a respeitar a semente principal. A preferência
de SOC para ligar tem variação por oportunidade (desvio 0,08); há ligação
atrasada em 30% das oportunidades elegíveis e variação da duração das visitas.
Se uma redução opcional da janela inviabilizar as viagens, o gerador tenta o
estacionamento completo e regista a rejeição da variação. Não altera a energia
das viagens nem o SOC para esconder um défice. Cargas pequenas essenciais e
com SOC superior a 98% são admitidas quando necessárias para a viagem seguinte.

### Falhas, stress e comparação

`generation_regime` e `cause_id` distinguem falha local, causa comum e stress
controlado. Uma falha de sensor pode gerar várias linhas de canais equivalentes:
essas linhas permanecem sincronizadas. As instalações independentes têm
calendários próprios. A taxa de rede é 1,13 falhas locais/ano por membro exposto;
a de telemetria PV mantém o prior de 1,2. Uma causa comum tem probabilidade anual
0,2 na rede e 0,25 na telemetria PV. As interrupções locais usam duração
lognormal com mediana de 42,45 minutos e sigma 0,65, discretizada para 15 minutos
e limitada a 3 horas. A causa comum usa mediana 49,63 minutos, obtendo média
discretizada de 60 minutos. Estes valores igualam os dois momentos BT citados
após discretização. Não são um ajuste das caudas, causas ou localização das
avarias; os membros fora da coorte continuam sem cortes neste benchmark.
Não se representam interrupções inferiores à resolução. Os valores de stress
abaixo não entram nessa correspondência com os indicadores publicados.

Mantêm-se os controlos exigentes: uma interrupção comum de 2 horas/ano na
coorte selecionada, ruído de sensores, atrasos/indisponibilidade de ativos,
limites apertados, histórias de topologia e o pedido de DR **1500 kW / 4 horas**
marcado `impossible_stress`. As datas são sorteadas por ano, sem cópia a 52
semanas. Nos restantes pedidos DR, duração e potência variam; a ocorrência é
ponderada pelo preço para redução e pela irradiância para aumento. São pedidos
sintéticos, não ativação observada do operador nem serviço qualificado.

Os regimes de ruído contínuo e campanhas de duração limitada continuam como
controlos de stress identificados. Não são uma calibração de sensores reais.
O contrato de observação é agora explicitamente `known_faults`; as etiquetas
de falha podem estar disponíveis ao agente. Uma experiência de diagnóstico
com falhas ocultas exige `hidden_faults` e uma lista explícita de observações
admissíveis, suportados pelo simulador. Não se deve publicar resultados desta
configuração como aprendizagem sem informação das falhas.

As sementes derivam de família, membro/causa, ano e processo. O nome do cenário
não entra na semente física. Micro/Core conservam fluxos comuns para identidades
partilhadas; Premium é independente. Cada comparação usa a mesma realização
de consumo/mobilidade/eventos e remove apenas o fator declarado. As saídas
permanentes filtram os eventos e serviços pela existência efetiva do membro.
Os seis pares de referência e as vistas de preparação de 2023 são auditados.

As previsões meteorológicas são um benchmark causal explícito: persistência
diária mais anomalia recente com decaimento exponencial de 6 horas. Não usam
meteorologia futura observada nem ruído fabricado para imitar uma previsão
operacional. Mantêm-se os zeros da previsão solar causal original e a sua
máscara noturna. O preço permanece `max(OMIE/1000, 0)`, com mercado local a 80%
e exportação a zero; não representa uma fatura residencial completa.

### Reprodução, validação e preparação de publicação

Executar `python scripts/revise_portugal_realism.py --apply` numa cópia do
repositório com o commit de origem `c54b13b62823fb1256397ea98cb37a830a24124d`
disponível. Não é necessário extrair uma segunda árvore de dados. `--seed`
permite repetir a experiência com outra realização e `--config` recebe um JSON
com todos os parâmetros explícitos. O inventário atual continua a ser o das
quatro bases Portugal: ainda não é uma interface geral para construir qualquer
comunidade a partir de zero.

Executar os testes Portugal e `python scripts/audit/audit_portugal_realism.py
--runtime`, além de `python scripts/audit/audit_portugal_heldout_2024.py --runtime`.
As evidências correntes ficam em `results/portugal_rec_native/realism_validation.json`
e `heldout_2024/validation.json`. As saídas de cada família têm hashes e a
configuração efetivamente usada em `realism_revision.json`. Figuras antigas
não constituem validação desta revisão; a figura corrente é
`figures/realism_validation.png`.

### Validação empírica e incerteza

`scripts/audit/portugal_empirical_validation.py` compara os ficheiros `raw` de
13 casas SHEERM. A separação foi declarada antes da inspeção das métricas:
IDs ímpares (sete casas) para desenvolvimento, IDs pares (seis) para avaliação.
Usam-se apenas dias civis completos com 96 observações válidas, sem preencher
lacunas; excluem-se duplicações e dias de mudança da hora. Potência média em
kW converte-se em energia multiplicando por 0,25 h. Não se usam os ficheiros
imputados publicados. A etiqueta `raw` não permite excluir todo o tratamento
anterior feito pelo equipamento. A inspeção entre casas não encontrou clones
que atravessassem a separação de desenvolvimento/avaliação.

O ajuste minimizou a média dos valores absolutos dos logaritmos das razões
entre medianas domésticas de cinco métricas: variabilidade diária dentro do
mês, P95 das variações de potência, desvio da forma diária, P99/potência média
e autocorrelação a um dia. A primeira grelha de 24 combinações atingiu uma
fronteira; foi ampliada para 54 combinações, sempre usando apenas as sete casas
de desenvolvimento. A ampliação incluiu o calendário de atividade revisto.
Doze casas sintéticas, duas de cada rotina residencial Premium, representam
o inventário durante o ajuste; os restantes usos e pedidos de aparelhos ficam
fixos nessa etapa. A avaliação final usa os pedidos e os consumos totalmente
regenerados de todas as casas residenciais das quatro bases.

As métricas dão o mesmo peso a cada casa; os intervalos de confiança da mediana
usam 2 000 reamostragens de casas, não intervalos de 15 minutos tratados como
observações independentes. O consumo sintético soma carga não deslocável e
ciclos controláveis iniciados no primeiro instante permitido. Exclui EV, cujo
inventário nas casas medidas é desconhecido. Não é uma avaliação de controlo.
Sazonalidade, clima, ocupantes e equipamentos diferem; três casas reais têm
aquecimento não elétrico. A amostra é pequena e não aleatória. A energia média
e a distribuição dos picos continuam a diferir: a aproximação de algumas
estatísticas não estabelece equivalência entre populações.

Para EV usa-se apenas `Dataset1_charging_reports.csv` do
[SINTEF, Noruega](https://doi.org/10.5281/zenodo.13896176): 35 377 sessões,
267 utilizadores, energia medida e instantes de ligação/desligação. Os ficheiros
com SOC e capacidades inferidos são excluídos. A separação de utilizadores usa
a paridade SHA-256 do identificador. Comparam-se apenas sessões domésticas
sintéticas de residentes em 2024, não censuradas, com filtros comuns de energia
superior a 0,5 kWh e duração entre 2 minutos e 120 horas. A concentração das
durações é comparada depois de discretizar ambas as populações a 15 minutos.
A frequência usa o intervalo observado entre primeira e última sessão, com
pelo menos dez sessões e 60 dias. A energia sintética é convertida para entrada
AC usando as perdas declaradas; a fronteira do contador real pode diferir.
As sessões sintéticas são maiores e menos frequentes que as norueguesas.
Mantém-se essa diferença explícita, dada a frota desconhecida, o período
2018–2021 e as preferências de cenário; não se apresenta como um ajuste do SOC
português nem se altera a energia das viagens para forçar concordância.

Os dois conjuntos compactos em `data/portugal_validation_reference` ocupam
cerca de 5 MiB. `sources.json` regista URLs, licenças CC-BY-4.0 e hashes dos
originais e dos Parquet derivados. A aquisição lê os CSV em memória, sem
guardar ZIPs ou segundas cópias. A geração corrente não requer rede.

`scripts/audit/audit_portugal_ensembles.py` executa 20 sementes em todas as
famílias, representando cada rotina residencial (80 realizações de componentes).
Os programas de sensores e DR são também regenerados nas 20 sementes e quatro
bases, verificando identidades, limites temporais, não sobreposição dos pedidos
e preservação do stress. As três bases Micro/Core partilham fluxos por identidade;
as suas realizações não são amostras independentes para inferência populacional.
Os outros usos permanecem fixos neste diagnóstico. Uma segunda experiência
regenera toda a cadeia física/aparelhos/EV em memória: cinco sementes em Micro
e Core-15 e duas em Core-30 e Premium (14 realizações coerentes, dois anos).
A sensibilidade varia um parâmetro de cada vez em ±25%, com três sementes
comuns: oito parâmetros de procura/calendário/falhas e quatro de carregamento/
aparelhos, totalizando 72 realizações. Estas experiências verificam geração e
viabilidade; não executam controladores anuais nem quantificam incerteza global.
Resultados e limitações ficam em `empirical_validation.json` e
`ensemble_validation.json`, e a comparação visual em
`figures/empirical_validation.png`.

Para um paper que alegue realismo populacional, continuam necessários uma
amostra portuguesa mais ampla com inventário conhecido, validação das
transferências por uso/tipo de estabelecimento e de máquinas profissionais,
dados de mobilidade/SOC medidos, ablações do calendário
partilhado e das causas comuns, e análise global das incertezas. Os casos
impostos de stress e os parâmetros fixos devem permanecer identificados.
Em particular, duração e associação energia/duração dos carregamentos ainda
diferem da referência norueguesa, além da energia e frequência marginais.
Os testes de software e esta comparação exploratória não substituem esses passos.

Outras referências identificadas, **não usadas para ajustar esta revisão**:

- [HEMStoEC, Ruano e Ruano](https://doi.org/10.1038/s41597-024-03184-5), com
  [dados no Zenodo](https://doi.org/10.5281/zenodo.8096648): quatro casas no sul
  de Portugal, janeiro de 2020–fevereiro de 2023, aquisições a 1 segundo/1 minuto
  e versões sincronizadas a 5 minutos. Permite comparar perfis e dependências,
  respeitando as lacunas e interpolações identificadas pelos autores. Quatro
  casas e o período pandémico não constituem uma amostra nacional representativa.
- [SustDataED2, Pereira et al.](https://doi.org/10.1038/s41597-022-01252-2):
  96 dias numa casa portuguesa, com 18 cargas medidas aproximadamente a cada
  2 segundos. É útil para transições e cargas de fundo; não se deve presumir
  submedição completa: os autores identificam cargas não monitorizadas,
  incluindo uma máquina de lavar e um termoacumulador.

O [artigo SHEERM](https://doi.org/10.1038/s41597-025-04750-1) descreve também
versões interpoladas e preenchidas por previsão. Essas versões não entram na
avaliação acima. A geração PV publicada é modelada, pelo que também não foi
tratada como medição independente de produção solar.

Uma sensibilidade adicional de qualidade encontrou três leituras `raw` da
casa 4 acima de 40 kW, equivalentes ao limite de 10 kWh por intervalo descrito
no artigo. Excluindo os três dias completos afetados, sem interpolar, a mediana
do indicador de variações rápidas passa de 1,245 para 1,248 e a do desvio da
forma diária de 0,522 para 0,520; as restantes cinco medianas verificadas não
mudam. A comparação principal conserva os dados `raw`. Esta inspeção é posterior
à primeira avaliação e está identificada como sensibilidade, sem reajuste do
gerador nem alteração das fontes guardadas.

Uma comparação publicável deve alinhar unidades, clima, composição doméstica,
usos elétricos, cobertura e tratamento de falhas antes de comparar distribuições.
Não se devem confundir os canais não monitorizados de um dataset real com a
fração não deslocável: falta de submedição não determina flexibilidade física.

Temperatura interior 21 °C e humidade interior 50% continuam a ser campos
constantes de compatibilidade, identificados como não térmicos; não suportam
conclusões de conforto. Os dados continuam a 15 minutos; não validam ainda uma
dinâmica física ou um controlo realista a 15 segundos.

### Referências comerciais e de aparelhos da revisão 3

A evidência reprodutível está em `commercial_appliance_validation.json` e em
`data/portugal_validation_reference/commercial_appliance_sources.json`.
Retêm-se apenas os canais comerciais selecionados, resumos de candidatos e
cobertura dos aparelhos e agregações das cozinhas a 15 minutos. Os CSV originais
são lidos em memória, verificados e descartados; não foram criados ZIPs ou
árvores duplicadas de datasets.

**Comerciais — BDG2.** [Miller et al., Building Data Genome 2](https://doi.org/10.1038/s41597-020-00712-x),
[repositório e licença CC-BY-SA-4.0](https://github.com/buds-lab/building-data-genome-project-2),
commit `9b97ccbe90096aff42ed4fd6493bf7ae692d7118`. Antes de ler os consumos,
selecionaram-se 63 edifícios por categoria, área e ordenação SHA-256 da identidade;
a separação alternada reserva edifícios físicos distintos. Os escritórios têm
100–3000 m²; os restantes setores usam os rótulos publicados. Os três edifícios
industriais são apenas contexto: **o inventário Portugal tem oficinas, não fábricas**.
O supermercado tem uma única referência, reservada, e nenhuma para ajuste.
Essa referência está rotulada `Market` no BDG2; é um proxy imperfeito, sem
inventário de refrigeração que permita confirmar equivalência a um supermercado.
A separação é por edifício, não por local/clima; persistem dependências entre
edifícios do mesmo campus. As áreas, países, horários e serviços energéticos
podem diferir substancialmente dos estabelecimentos fictícios portugueses.

Os valores publicados são kWh por hora e timestamps locais. Comparam-se dias
com 24 horas distintas e válidas, excluindo lacunas, zeros sinalizados como
suspeitos e dias de mudança da hora; não se interpola. Tratar zeros como
suspeitos também pode excluir encerramentos reais: a cobertura é reportada.
As estatísticas são por edifício, sem dar maior peso aos edifícios com mais
observações. Não se alega validação de dinâmica a 15 minutos por dados horários.
Uma referência desportiva não tem dias completos válidos e não entra nas
medianas. A sensibilidade posterior de cobertura exige 180 dias completos e
dez meses do calendário: exclui um escritório e um centro desportivo de
desenvolvimento. A intensidade mediana dos escritórios passa de 154,56 para
184,04 kWh/m²/ano; a desportiva, de 73,42 para 79,15. As medianas reservadas
não mudam. Mantêm-se as âncoras congeladas e reporta-se esta incerteza, sem
reajustar o gerador ou apagar a comparação principal.

Na revisão 3, as medianas de desenvolvimento forneceram âncoras de intensidade anual, relação
noite/dia, dispersão diária e persistência. A intensidade é uma média geométrica
entre o orçamento original por m² e a âncora do setor (peso 0,5), excluindo EV.
O peso do rácio noturno no consumo de fundo é 0,65; a heterogeneidade entre
edifícios tem sigma lognormal 0,16, limitada a 0,7–1,3. Os pesos de transferência
são pressupostos e não parâmetros estimados em Portugal; a sensibilidade inclui
transferência energética nula/completa e pesos de forma 0,4/0,9. O supermercado
na revisão 3 mantinha âncoras próprias de cenário. As intensidades anuais
portuguesas e a dispersão entre locais são revistas na secção da revisão 4 abaixo.

Escritórios e oficinas têm atividade de trabalho e redução à hora de almoço;
lojas e supermercado têm procura sobretudo diurna/tarde; cafés têm picos de
refeições; centros desportivos têm pico vespertino; lares operam 24/7. Estas
formas são pressupostos por atividade, não uma identificação causal das medições.
O nível diário é correlacionado e os picos variam no tempo; a flutuação entre
intervalos também é persistente. Preparação e limpeza prolongam a atividade
antes/depois da abertura ao público. Os 26 calendários públicos originais são
preservados, incluindo feriados, férias em data civil e ano bissexto.
`commercial_activity.parquet` regista a atividade diária; `operations` acrescenta
atividade e presença esperada. A presença usa densidades-prior de pessoas por
área e não significa que existam medições de pessoal/clientes. `members.occupants`
continua a contar residentes domésticos, enquanto `occupant_count` representa
presença esperada. Os campos de dimensionamento `native_peak_import_kw`, `native_peak_export_kw`
e `native_feasibility_margin` descrevem a construção histórica, não uma nova
medição do pico desta revisão; os máximos correntes são auditados contra os
limites efetivamente usados e reportados na evidência comercial.
Os membros suplementares conservam a sua condição explícita
de clones de um arquétipo: recebem a realização atual do dador, não dados antigos.

**Aparelhos — REFIT.** [Murray, Stankovic e Stankovic](https://doi.org/10.1038/sdata.2016.122),
[dados originais](https://doi.org/10.15129/9ab14b0e-19ac-4279-938f-27f643078cec) e
[depósito público usado, CC-BY-4.0](https://doi.org/10.5281/zenodo.5063428).
As casas 3/5 foram destinadas ao desenvolvimento e 2/11 à avaliação antes de
inspecionar potência. Os canais foram confirmados pelos metadados do conversor
NILMTK, commit `8967bd1fde24d0c8d89408179dd2499d8e60b4c3`.
São máquinas britânicas de 2013–2015, não uma amostra portuguesa atual. Não se
estimou a proporção de modos eco nem a distribuição de idade/capacidade do
parque português de 2024. A aproximação às métricas REFIT não prova essa
representatividade. Não se
usam os agregados das casas com PV para identificar programas dos aparelhos.

Detetam-se candidatos com limiar 10 W e intervalo inativo máximo de 15 minutos;
testam-se limiares 5/20 W e intervalos 5/30 minutos. Excluem-se candidatos
censurados, com falhas de amostragem, duração observada inferior a 98% ou
energia/duração implausíveis. A energia integra os intervalos observados,
limitando retenção da última amostra a 16 segundos. O campo `Issues` indica
soma dos medidores individuais acima do agregado, podendo refletir assincronia
ou PV; não é uma etiqueta de avaria daquele aparelho. A comparação principal
não elimina um programa inteiro por uma amostra sinalizada. Apresenta-se uma
sensibilidade separada restringindo a fração sinalizada a 10%. A limpeza
publicada preencheu lacunas curtas e zerou algumas longas, que não podem ser
reconstituídas. Candidatos detetados não são programas anotados. As frequências
usam apenas dias com pelo menos 90% de cobertura, incluindo dias sem utilização;
a sensibilidade de qualidade não deve ser interpretada como frequência real.

**Cozinhas profissionais — FIKElectricity.** [Pereira et al.](https://doi.org/10.1038/s41597-023-02698-8),
[dados OSF, CC-BY-4.0](https://doi.org/10.17605/OSF.IO/K3G8N). As três cozinhas
portuguesas e a máquina de loiça/lava-copos da cozinha 1 são um diagnóstico
estrutural externo. Não existe aqui uma segunda máquina profissional reservada
para validar parâmetros. Os timestamps são UTC, convertidos para hora civil;
usam-se watts ativos, não VA. A integração limita a retenção a duas vezes o
intervalo mediano e guarda cobertura, sem preencher lacunas. As comparações
diárias exigem pelo menos 90% de intervalos com 80% de cobertura; resultam em
9/21/22 dias para os agregados e 21/24 dias para os dois aparelhos. Não se
extrapolam essas semanas para um consumo anual medido nem se equipara cozinha
a edifício completo. Os consumos não submedidos não são classificados como
não deslocáveis apenas por falta de medição.

A auditoria adicional usa 20 sementes em oito setores (160 realizações de dois
anos), 30 configurações de sensibilidade com oito setores (240 realizações) e
20 sementes para cinco classes de utilização de aparelhos. Verifica diversidade,
limites físicos, conservação de energia e identidade dos calendários públicos.
A cadeia física/aparelhos/EV completa é testada separadamente no ensemble comum.
As gamas Monte Carlo não são intervalos de confiança populacionais; a análise
um-parâmetro-de-cada-vez não explora todas as interações.

### Referências portuguesas e correções da revisão 4

A revisão 4 acrescenta referências **para informar pressupostos**, não uma
nova validação independente. Os resultados BDG2/REFIT/FIKElectricity já tinham
sido examinados na revisão 3; as novas comparações são diagnósticos/regressões.
O protocolo anterior e a situação imediatamente anterior à revisão ficam
preservados no relatório existente, sem cópias dos datasets ou PDFs.

- **Escritórios:** [Costa, IST, 2021](https://fenix.tecnico.ulisboa.pt/departamentos/decivil/dissertacao/846778572212503), faturação de 2019. A amostra total contém 104 edifícios; só 58 têm área, 59 têm trabalhadores e 96 têm faturação de 2019. Usa-se a média publicada de 94,4 kWh/m²/ano como centro de cenário, com dispersão lognormal entre edifícios de sigma 0,25, assumida. Não se divide a média de 2500 kWh/trabalhador pela média por área para inferir ocupação: os subconjuntos diferem. Faturação, autoconsumo fotovoltaico e EV não têm fronteiras integralmente reconciliadas com o consumo bruto sem EV do gerador; esta é uma limitação explícita.
- **Pequeno comércio:** [Sequeira, NOVA, 2016](https://run.unl.pt/bitstream/10362/19555/1/Sequeira_2016.pdf), tabelas 4.1/4.3: médias publicadas de 39 kWh/m²/ano para cinco lojas gerais, 271 para dois estabelecimentos de restauração e 531 para dois com produtos refrigerados. São auditorias anuais calculadas por usos; oito dos treze estabelecimentos foram confrontados com faturas, não treze séries anuais medidas. Os 37 inquéritos adicionais e a extrapolação para 107 estabelecimentos não aumentam a amostra medida. A transferência é parcial: média geométrica entre a intensidade original do inventário e a referência. A dispersão entre locais usa o desvio-padrão descritivo dos logaritmos dos valores publicados (0,307 nas lojas, 0,260 na restauração e 0,213 nos refrigerados); a transferência parcial atenua-a. Nos restantes setores, sigma 0,30 é um prior explícito. O fator individual é persistente entre anos; não se força cada estabelecimento à mesma intensidade. As lojas refrigeradas de aproximadamente 36/80 m² são muito menores que a mercearia sintética de 500 m². Esta não fica validada como supermercado típico.
- **Ocupação:** trabalhadores em turnos (jornadas individuais de 7,5–9 h para escritório/oficina, distintas do período de abertura), clientes com chegadas/saídas e residentes permanentes do lar têm processos separados, inteiros e limitados pela capacidade assumida. Clientes respeitam a abertura; no lar, visitas ocorrem apenas entre 10 e 20 h, um prior de cenário separado do serviço contínuo. Preparação/limpeza podem incluir pessoal fora da abertura. O ruído elétrico não modifica pessoas. A procura de aparelhos comerciais acompanha horas-pessoa do grupo pertinente, conservando uma taxa de utilização média assumida. Densidades, assiduidade e permanência continuam priors: dados portugueses de salas de aula não foram transferidos para lojas/escritórios.
- **Equipamentos:** a [ficha Miele portuguesa](https://media.miele.com/downloads/t-/pt/FS_11496730_PTP_PT-pt-PT.pdf) informa a energia da lavandaria; não valida cargas parciais ou produção diária de roupa. A lavandaria modelada é apenas a carga interna do inventário: não representa automaticamente toda a roupa gerada pelo lar/centro desportivo, nem inclui secagem, engomadoria ou serviços externos. A máquina de loiça profissional mantém lotes agregados e apoio térmico separado. O primeiro intervalo de abertura é reservado ao aquecimento; testa-se a potência combinada em todos os inícios admissíveis.

O objetivo anual comercial é agora o consumo de todo o edifício, com cada
programa de aparelho contado **uma única vez**. As energias dos programas fixos
e controláveis e do apoio profissional saem do orçamento de base; os fixos são
somados ao não deslocável e os controláveis entram apenas na execução. O campo
`building_budget_excluding_ev_kwh` permite testar diretamente este balanço.
Carga de EV, perdas de armazenamento e trocas de mercado não pertencem a esse
orçamento de procura do edifício.

Mantêm-se os horários públicos, contratos, carregadores, catálogos físicos e
pares de cenários. Os escritórios conservam os seus carregadores, sem inferir
que o número de carregadores determina o número de trabalhadores. Não se usa
saúde/beleza como amostra de clínicas, uma gráfica como fábrica, nem energia
primária/padronizada SCE como consumo elétrico medido. Clínicas, oficinas,
centros desportivos e lar ainda carecem de referências portuguesas comparáveis.
As hipóteses e a rastreabilidade estão em `commercial_appliance_sources.json`;
o orçamento e as distribuições correntes em `commercial_appliance_validation.json`.

## Preços correntes — v3.2

Desde 29/10/2026, o preço ativo é `max(OMIE português em EUR/MWh / 1000, 0)`.
O mercado comunitário está ativo a 80% desse preço; exportação à rede a zero.
As previsões usam apenas OMIE, com publicação causal e persistência diária.
Não se aplicam componentes Coopérnico/ERSE, MIBEL, perdas ou impostos.
Ver [Tarifas e mercado](Tarifas_e_mercado.md).

## Fontes e metodologia da edição histórica de retalho

As descrições tarifárias seguintes documentam os arquivos de origem. As suas
fórmulas e séries de retalho não são selecionadas pelos schemas correntes.

Reference year: **2023**. Retrieved 25 September 2026. Individual buildings,
occupancy and equipment are synthetic; the files here are independent inputs.

- [E-REDES consumption profiles](https://www.e-redes.pt/sites/eredes/files/2023-01/E-REDES_Perfil_Consumo_2023.xlsx): original workbook and chronological CSV, including 92/100 intervals on Lisbon DST dates. BTN-C is a shape prior, not measured individual households.
- [PVGIS API](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/api-non-interactive-service_en): Lisbon 38.72,-9.14, SARAH3, 2023; archived response includes all parameters. Hourly satellite irradiance and reanalysis temperature/wind are interpolated to 15 minutes. No claim of observed quarter-hour clouds.
- [OMIE day-ahead prices](https://www.omie.es/pt/market-results/daily/daily-market/day-ahead-price): `omie_2023_market_periods.parquet` preserves the raw market days. `omie_2023_daily_provenance.csv` records source URLs/hashes. `omie_2023_hourly.parquet` now assigns each period to **Europe/Madrid**, then converts to UTC. The first 2024 market period supplies the final UTC hour of 2023. The former one-hour UTC misalignment is corrected.
- [OMIE definitive adjustment](https://www.omie.es/es/market-results/daily/average-final-prices/hourly-price-consumers?date=2023-01-01&scope=daily): 365 daily snapshots in `raw_2023/omie_adjustment_*.json.gz`, with source URLs and HTTP payload hashes. Includes signed, nonzero adjustments. [DGEG expiry](https://www.dgeg.gov.pt/pt/estatistica/energia/precos-de-energia/mecanismo-iberico/): 31 December 2023.
- [Coopérnico tariff, January 2023](https://www.coopernico.org/files/documents/Coopernico_Tarifario_Indexado.pdf): BASE supply formula uses CGS 0.004, margin 0.010 EUR/kWh and a 15% loss uplift. This historical PDF is archived; the modern 2.0 formula is not substituted for 2023. BTE uses the same **benchmark** supply convention, not a quoted Coopérnico BTE contract. The adjustment is applied to procurement energy before losses.
- [ERSE H1 2023](https://www.erse.pt/media/vueeumz4/tarifas-e-pre%C3%A7os-2023.pdf) and [H2 2023](https://www.erse.pt/media/xjeks20k/tarifas-e-pre%C3%A7os-se-2sem2023.pdf): regulated access energy and contracted-power rates. H2 begins at 00:00 Lisbon on 1 July. Negative access terms are preserved. [2023 tariff regulation](https://www.erse.pt/media/y02kc00v/regulamento-9-2023_rt_site.pdf), article 43: peak power is Ep/Hp; daily BTE has four peak hours per day, so its 0.4854 EUR/kW/day becomes 0.12135 EUR/kWh in peak intervals. Winter peak 09:00–10:30 and 18:00–20:30; summer 10:30–13:00 and 19:30–21:00. Daily off-peak 22:00–08:00, BTE super-off-peak 02:00–06:00. [ERSE period tables](https://www.erse.pt/atividade/regulacao/tarifas-e-precos-eletricidade).
- [REN API documentation](https://datahub.ren.pt/en/api-instructions): daily `ElectricityProductionBreakdownDaily` snapshots under `raw_2023/ren_*.json.gz`, MW values at Lisbon quarter-hour intervals. Clock labels are checked against physical DST-aware timestamps.
- [APA emission factors](https://apambiente.pt/clima/fator-de-emissao-de-gases-de-efeito-de-estufa-para-eletricidade-produzida-em-portugal): archived 2025 report, mainland **2023 annual production 0.092 kg CO2e/kWh**. This is not the all-Portugal value or a five-year average.
- [REE API](https://www.ree.es/en/datos/apidata): Spanish peninsular daily operational production intensity, `no-renovables-detalle-emisiones-CO2`; exact request URLs inside the archived JSON. Daily values are held within their physical Spanish day.

## Derived inputs and scientific scope

`python scripts/prepare_portugal_reference.py` reconstructs prices and carbon
offline, without requesting any website. `aligned_reference_metadata.json`
describes formulas and exclusions. `price_components_*.parquet` decomposes the
five tariffs; `pricing_*.parquet` supplies actual and causal forecast columns.
All prices are EUR/kWh, before VAT. IEC 0.001 EUR/kWh is included, without tax
exemptions. Fixed power charges, audiovisual/DGEG fees, VAT and supplier fixed
margins are not part of the marginal control KPI; fixed access rates are
reported separately in each dataset. This does not reproduce a full invoice.

OMIE forecasts use publication at a conservative 14:00 Madrid. The definitive
MIBEL adjustment is estimated from past observations, not revealed from the
future. Access rates and their known calendar are used for the target period.
The archived E-REDES loss profile is background material; this version uses
the historical Coopérnico flat 15% convention consistently.

`carbon_components_2023.parquet` contains the REN mix, REE imports proxy,
production estimate, delivered estimate and balance residual. A common fossil
coefficient is calibrated so generation-weighted Portuguese production equals
the APA annual factor. Import blending assumes Spain's daily production mix;
the delivery factor is 1.15. **Derived average operational emissions, not
measured 15-minute emissions, marginal emissions, lifecycle emissions or a
full cross-border flow tracing result.** Biomass biogenic CO2 and stored-hydro
carbon attribution are not independently modelled. The annual calibration
reconciles different accounting boundaries and is not independent validation.

SHA-256 hashes of all source files, including daily archives, are in the
generated dataset manifests. Raw provider data remain attributed; public
access does not establish unrestricted redistribution rights.


## Native-runtime recovery

The physical inputs and historical retail references above are retained. Current schemas select `shared/pricing_omie.parquet`, enable native community trading at 0.8 of the nonnegative OMIE purchase price and retain zero grid-export remuneration. Source provenance is not evidence of longitudinal EV simulation or calibrated DR.


## Extensão 2024

As fontes seguintes suportam exclusivamente o ano acrescentado. Os originais e os pedidos de aquisição estão em `data/portugal_reference_2024`.

- [E-REDES, perfis de consumo 2024](https://www.e-redes.pt/sites/eredes/files/2023-12/E-REDES_Perfil_Consumo_2024.xlsx).
- [OMIE, descarga de preços marginais](https://www.omie.es/en/file-download?parents=marginalpdbcpt&filename=marginalpdbcpt_20240101.1): 366 dias de mercado e a primeira hora relevante do dia seguinte ao ano.
- [ERSE, tarifas 2024](https://www.erse.pt/media/kqyhsv1v/tarifas-e-pre%C3%A7os-2024.pdf) e [revisão extraordinária de junho](https://www.erse.pt/media/3khpzpvk/tarifas-e-pre%C3%A7os-se-extra-2024.pdf).
- [Coopérnico, fórmula e encargo TSE de 2024](https://www.coopernico.org/artigo/366): convenção de perda indicativa de 16%, não perdas horárias reais de uma instalação. O encargo retroativo de abril não é revelado às previsões antes da publicação; adota-se conservadoramente 1 de maio.
- [PVGIS6](https://photovoltaic-geographic-information-system.ec.europa.eu/en/about-pvgis-6), SARAH-3 até 2024; [API](https://photovoltaic-geographic-information-system.ec.europa.eu/api/v6/redoc).
- [Open-Meteo, arquivo ERA5](https://open-meteo.com/en/docs/historical-weather-api): temperatura, vento, humidade e apenas os preenchimentos solares documentados. Dados de reanálise, não medições residenciais.
- REN Data Hub: séries diárias oficiais de produção e intercâmbios, em `raw_2024/ren_*.json.gz`, incluindo o URL de cada pedido.
- REE: intensidade operacional diária peninsular, conservada com o URL de aquisição nos ficheiros `ree_emissions_*`.
- [Calendário escolar 2024/25](https://www.dgeste.mec.pt/wp-content/uploads/2024/07/despacho8368-2024-07-25-calendario-escolar-2024-25-2027-28.pdf); para 2023/24, Despacho 8356/2022. A data de início dentro do intervalo permitido é uma escolha da escola fictícia.

Ver [Extensão de 2024](Extensao_2024.md) para diferenças de método, lacunas, incerteza e limites de validação. Os dados são cenários sintéticos informados por fontes públicas; não são registos das famílias narradas.

## Apoio às previsões no fim de 2024

As referências históricas de apoio cobrem apenas 01/01/2025 UTC e não entram no período simulado. OMIE e anúncios ERSE de dezembro de 2024, com ficheiros, URLs e hashes, estão em `data/portugal_reference_2024/forecast_support`. TAR BTN bi-horária: 0,0149/0,0830 €/kWh; TSE: 0,0016574 €/kWh. Restantes componentes da referência comum mantidas. Ver Previsoes_e_historias.md para as fontes, causalidade e métricas.

Na edição histórica, a humidade de 2023 era uma aproximação sintética a partir
da temperatura, enquanto a de 2024 vinha de ERA5. A análise histórica das
previsões inclui essa diferença de origem. A revisão `portugal-realism-1`
substitui a humidade de 2023 por ERA5, como descrito no início deste documento;
as métricas históricas não validam as previsões correntes.

Na edição v3.2, os ficheiros OMIE brutos desse dia de apoio são usados diretamente, sem TAR, TSE ou outras componentes.
