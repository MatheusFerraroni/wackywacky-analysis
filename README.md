# Análise do WackyWacky

Projeto dedicado a caracterizar uma prévia do dataset WackyWacky ainda em
geração. O resultado esperado é um conjunto de métricas, tabelas e figuras para
descrever, em artigo científico, os dados disponíveis em uma data de corte.

## Escopo

- `pages.tsv`: estrutura, estados de coleta, disponibilidade e integridade de
  `text`, distribuição de palavras, vocabulário, tamanho, idioma, repetição e
  outros indicadores de qualidade textual.
- `domain.tsv`: quantidade, níveis de recursão, requisições, relações pai-filho
  e concentração das páginas por domínio.
- Cruzamento por `pages.domain_id = domain.id` para medir cobertura,
  diversidade e possíveis inconsistências.


## Fonte

O dataset fica fora deste repositório e deve ser acessado somente para leitura.
`pages.text` contém um frame Zstandard representado em hexadecimal; a análise
deve decodificá-lo como UTF-8 e validar `text_md5` antes das estatísticas
linguísticas.

Como a coleta continua ativa, todo resultado deve informar a data de corte e
ser descrito como parcial.

## Estado

A base documental está criada. A implementação da análise ainda não foi
iniciada.

## Documentação

- [Dataset e estado conhecido](docs/dataset.md)
- [Escopo da análise](docs/analysis.md)

Os resultados devem ser agregados e reproduzíveis. Textos, URLs e domínios
reais não devem aparecer nos artefatos publicados.
