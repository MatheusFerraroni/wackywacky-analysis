# Dataset e estado conhecido

## Natureza da fonte

O WackyWacky é um dataset em geração. A pasta analisada é uma exportação parcial
de um crawler MySQL e não deve ser tratada como versão final ou snapshot
necessariamente transacional.

Estado observado em 2026-08-20:

| Arquivo | Tamanho | Estrutura |
| --- | ---: | --- |
| `pages.tsv` | 56.205.366.282 bytes | 19 colunas |
| `domain.tsv` | 66.912.833 bytes | 9 colunas |

## `pages.tsv`

Campos centrais: `id`, `domain_id`, `same_as`, `status`, `url`, `url_final`,
`title`, `text`, `text_md5` e `recursion_level`.

`text` não é texto puro. O fluxo esperado é:

```text
hexadecimal -> bytes Zstandard -> bytes descompactados -> UTF-8
```

O marcador literal `NULL` representa ausência em vários campos. `same_as`
aponta para outra página quando o crawler identifica duplicação.

Uma amostra limitada e distribuída pelo arquivo encontrou 29.278 linhas com 19
campos válidos. Entre 719 textos elegíveis amostrados, todos foram
descompactados, mas 299 divergiram de `text_md5`. Esses números são
preliminares e não representam a distribuição global.

O total de linhas permanece desconhecido. A fonte aparenta estar ordenada por
estado e possui registros de tamanhos muito diferentes; amostras por offset não
devem ser usadas como contagem definitiva.

## `domain.tsv`

Campos centrais: `id`, `url`, `parent_domain_id`, `recursion_level`,
`request_count` e `last_request_at`.

Os IDs observados são esparsos. Uma estimativa preliminar indica cerca de 615
mil registros, mas a contagem precisa deve ser obtida por uma passagem completa
em streaming.

## Relação e consistência

`pages.domain_id` referencia `domain.id`. Como a exportação ocorreu enquanto o
crawler aparentava continuar ativo, a análise deve medir referências ausentes
e registrar a data de corte usada.
