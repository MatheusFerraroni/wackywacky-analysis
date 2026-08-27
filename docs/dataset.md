# Dataset e estado conhecido

O WackyWacky é uma exportação parcial de um crawler ainda em execução. A análise usa somente cópias locais imutáveis e registra uma data de corte; não presume que a origem tenha sido transacional.

Estado observado em 2026-08-20:

| Arquivo | Tamanho observado | Colunas |
| --- | ---: | ---: |
| `pages.tsv` | 56.205.366.282 bytes | 19 |
| `domain.tsv` | 66.912.833 bytes | 9 |

## `pages.tsv`

Campos analíticos: `id`, `domain_id`, `parent_page_id`, `same_as`, `status_code`, `recursion_level`, `status`, `retry_count`, `text` e `text_md5`. URLs e títulos existem na fonte, mas não são lidos para a análise pública.

`text` segue `hexadecimal → Zstandard → UTF-8`. `NULL`, vazio e `\N` representam ausência. `text_md5` é apenas diagnóstico (`ausente`, `inválido`, `correspondente` ou `divergente`); a identidade científica usa SHA-256 calculado pelo pacote.

`same_as` é medido antes de exclusões e comparado com as duplicações por bytes e texto normalizado. Referências ausentes são reportadas, não corrigidas.

## `domain.tsv`

As nove colunas reais são `id`, `url`, `url_md5`, `parent_domain_id`, `recursion_level`, `request_count`, `last_request_at`, `created_at` e `updated_at`. Não há campo `status`. O parser lê bytes e ignora `url_md5`, que pode conter dados binários. Da URL são derivados somente host e domínio registrável usando uma Public Suffix List offline.

IDs duplicados, pais ausentes, profundidade, requisições, hosts e agrupamento Wikimedia são medidos. `pages.domain_id` é validado contra `domain.id`.

## Limites

Linhas e textos descompactados têm limite padrão de 128 MiB. Linhas excessivas são drenadas até `\n` sem acumulação.

O perfil `tiny` extrai até 10 mil páginas de janelas distribuídas da fonte e então analisa somente essa cópia privada. URLs de páginas, HTML e demais campos sem uso analítico são substituídos por `NULL`. A amostra não é representativa e seus resultados recebem marca explícita de prévia.

O `tiny` usa um trabalhador e orçamento inferior a 768 MiB. O `full` usa oito trabalhadores, limite DuckDB de 96 GiB e exige 300 GB livres além dos TSVs.

Os números acima descrevem apenas a prévia observada. Totais científicos devem vir de `manifest.json` e `summary.json` do snapshot executado.
