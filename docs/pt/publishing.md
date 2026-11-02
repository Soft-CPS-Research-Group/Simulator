# Guia de publicação

Este fork publica **`softcpsrecsimulator`** a partir de
[Soft-CPS-Research-Group/Simulator](https://github.com/Soft-CPS-Research-Group/Simulator).
O import Python continua a ser `citylearn`. Versão inglesa:
[../publishing.md](../publishing.md). Responsável por defeito:
[@calofonseca](https://github.com/calofonseca).

A publicação é uma ação separada do mantenedor. Testar ou limpar documentação
localmente não implica alterar versão, criar tag, fazer push ou publicar.
A versão no código não prova que todas as alterações locais já foram publicadas.

## Preparar e validar

1. Rever o diff. Atualizar `citylearn/__init__.py`, a versão em `README.md` e
   `README_PYPI.md`, e [releases.md](releases.md). Atualizar os contratos afetados,
   a compatibilidade e as migrações necessárias.
2. Executar a suite completa e os auditores obrigatórios do
   [guia de desenvolvimento](developer_guide.md). Evidência de execução fica fora
   do checkout ou em artefactos de CI/releases, não em JSON de cada execução em `docs/`.
3. Construir e verificar as distribuições numa pasta de saída nova para a versão.
4. Validar a wheel instalada fora do código-fonte. O
   [workflow CI](../../.github/workflows/run_tests.yml) exemplifica
   `scripts/ci/installed_export_smoke.py` com Python isolado e caminhos explícitos.

```console
.venv/bin/python -m pip install build wheel twine
.venv/bin/python -m pytest -q
.venv/bin/python scripts/audit/run_full_audit.py --output-dir /tmp/softcps-release-audit
.venv/bin/python -m build --outdir /tmp/softcps-dist-X.Y.Z
.venv/bin/python -m twine check /tmp/softcps-dist-X.Y.Z/*
```

## Commit, CI e tag

Numa publicação autorizada, fazer commit com a identidade habitual do mantenedor
e enviar o commit revisto para `origin/master` uma vez. Aguardar a sua CI antes
de publicar. A CI atual cobre Python 3.9/3.10, lint, testes, auditorias
contabilísticas, desempenho e exportação do pacote instalado. Depois, criar uma
tag ainda não utilizada para o commit aprovado:

```console
git push origin master
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

São passos de publicação, não de manutenção local. O push da tag não repete a CI
configurada apenas para branches. Não acrescentar um disparo manual sem necessidade.
Conservar tags herdadas: a versão 1.7.0 do fork usa `softcpsrecsimulator-v1.7.0`,
porque `v1.7.0` já pertencia ao upstream.

## Publicar e confirmar

O [workflow Publish Python Package](../../.github/workflows/pypi_deploy.yml) é
iniciado por uma **GitHub Release publicada** ou `workflow_dispatch` explícito;
o push da tag, por si só, não publica. Escolher um único disparo por versão,
sem repetir manualmente depois da GitHub Release.

- O build usa Python 3.11, cria wheel/sdist, executa `twine check`, instala a wheel
  e verifica exportações fora do checkout antes de guardar os artefactos.
- A publicação depende desse build, usa o ambiente `pypi` e envia os artefactos
  ao projeto do fork com o `PYPI_API_TOKEN` configurado.
- Num disparo manual, selecionar a tag aprovada. Não enviar tags para `upstream`
  nem publicar este fork como o projeto `CityLearn` original.

O mantenedor confirma a CI da principal antes de criar a GitHub Release.
A dependência automática da publicação é o próprio job de build/exportação.
Este guia descreve os workflows existentes sem modificar os seus disparos.

Confirmar os dois jobs para a referência aprovada e a versão exata no
[PyPI](https://pypi.org/project/softcpsrecsimulator/). Instalar essa versão fixa num
ambiente limpo, fora do checkout, e verificar `citylearn.__version__`, o import de
`CityLearnEnv` e a exportação. Upload aprovado não substitui o teste do pacote instalado.

## Política e modelo de notas

| Tipo | Quando usar |
|---|---|
| Patch | Correções compatíveis, documentação e alterações aditivas. |
| Minor | Capacidade substancial ou mudança de schema/API. |
| Major | Alterações amplas incompatíveis. |

Manter uma entrada por versão no changelog central, com âmbito e limites reais
da validação. Uma iteração sem commit não cria automaticamente uma versão.

```markdown
## vX.Y.Z — YYYY-MM-DD — resumo

Responsável: mantenedor.

- Alterações e impacto para o utilizador.
- Impacto nos datasets/schemas e compatibilidade.
- Validação, cobertura e limitações.
- Migração, quando necessária.
```
