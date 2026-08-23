# Análise do WackyWacky

Pacote independente para caracterizar cópias imutáveis de `pages.tsv` e `domain.tsv` sem carregar o corpus na memória. A execução é manual na processadora; o projeto não automatiza SSH e nunca modifica os TSVs.

## Visões do corpus

- `R_valid`: páginas `done` com texto hexadecimal/Zstandard/UTF-8 válido e não vazio.
- `E_exact`: `R_valid` após deduplicação exata normalizada.
- `B_clean`: `E_exact` após remoção intradomínio aprovada por revisão e nova deduplicação. É a visão lexical principal.
- `N_near`: sensibilidade opcional por MinHash e Jaccard; não altera `B_clean`.

## Instalação

Requer Python 3.12 e [`uv`](https://docs.astral.sh/uv/). As dependências, inclusive `pt_core_news_sm==3.8.0`, estão em `uv.lock`.

```bash
uv sync --extra dev
```

Copie `configs/full.toml`, ajuste caminhos, data de corte e orçamento da máquina, e versione a configuração usada no artigo.

## Prévia local com dados reais

O perfil `tiny` cria uma amostra privada determinística de até 10 mil páginas. Ele lê janelas distribuídas da fonte, não os 52 GB completos, e serve apenas para validar o pipeline e os gráficos; seus agregados não estimam o corpus.

```bash
export WACKYWACKY_DATA_ROOT=/caminho/para/wacky
uv run wackywacky sample --config configs/tiny.toml
uv run wackywacky verify --config configs/tiny.toml
uv run wackywacky run --config configs/tiny.toml
```

A amostra fica em `work/`, sem URLs de páginas, HTML ou campos não analíticos. Se houver candidatos a boilerplate, use o mesmo fluxo de revisão descrito abaixo e retome com `--resume`.

## Execução

```bash
uv run wackywacky verify --config configs/full.toml
uv run wackywacky run --config configs/full.toml
```

Em terminal interativo, o comando usa barras `tqdm` com velocidade e ETA; quando redirecionado, grava o mesmo progresso periodicamente no `stderr`. O JSON final permanece isolado no `stdout`.

No perfil `full`, a etapa lexical usa `runtime.workers` processos persistentes e confirma um checkpoint a cada `runtime.chunk_bytes`. `--resume` reaproveita o último offset confirmado tanto na tokenização quanto na recontagem de bigramas; um estado lexical antigo é descartado sem afetar D1, D2, revisão, limpeza ou D3.

Após o léxico, `run` caracteriza somente `B_clean`: estrutura de frases e parágrafos, diversidade lexical, classes gramaticais, repetição interna, sinais textuais, colocações e variação entre domínios. A etapa possui checkpoint próprio; `--resume` complementa snapshots antigos sem refazer inventário, deduplicação ou revisão.

Quando houver candidatos intradomínio, `run` termina com código 2 e grava a amostra privada em `work/`. Rotule cada item como `boilerplate`, `conteúdo` ou `incerto`:

O CSV mantém um item por linha física, mostra quebras internas como `[QUEBRA]` e começa por `sample_id` e `frequencia`. Todo `label` começa como `boilerplate`; altere somente as exceções para `conteúdo` ou `incerto`. Prévias longas são reduzidas de forma explícita.

```bash
uv run wackywacky review export --config configs/full.toml
uv run wackywacky review import --config configs/full.toml --input /caminho/revisao.csv
uv run wackywacky run --config configs/full.toml --resume
```

Comandos independentes:

```bash
uv run wackywacky near-duplicates --config configs/full.toml
uv run wackywacky render --config configs/full.toml --snapshot-id SNAPSHOT_ID
```

`render` lê somente agregados. Scratch, Parquet, bancos e revisão privada ficam ignorados; `results/<snapshot-id>/` contém apenas manifests, tabelas e figuras publicáveis.

Alterações apenas no estilo das figuras continuam exigindo somente `render`. Alterações nas métricas de `[content]` retomam apenas a análise de conteúdo.

## Validação local

```bash
uv run pytest
```

Somente os testes usam dados sintéticos. A execução integral deve ocorrer na processadora após o `verify` confirmar hashes e pelo menos 300 GB livres de scratch.

Detalhes: [dataset](docs/dataset.md) e [metodologia](docs/analysis.md).
