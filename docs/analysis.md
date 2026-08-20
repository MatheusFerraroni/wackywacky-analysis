# Escopo da análise

## `pages.tsv`

Medir:

- total de registros por `status` e `recursion_level`;
- presença, decodificação e integridade de `text` e `text_md5`;
- quantidade de palavras e caracteres por documento;
- tamanho e crescimento do vocabulário;
- frequências, hapax, n-gramas e distribuição de Zipf;
- idioma, caracteres, repetição, boilerplate e duplicação;
- concentração de texto e páginas por domínio.

Antes de publicar resultados, a definição de palavra, normalização Unicode,
tratamento de caixa e regras de tokenização deve ser explicitada.

## `domain.tsv`

Medir:

- quantidade de domínios e distribuição por profundidade;
- `request_count`, domínios nunca requisitados e relações pai-filho;
- páginas por domínio e medidas de concentração;
- cobertura do cruzamento com `pages.tsv` e referências ausentes.

## Método

- Fazer passagens sequenciais em streaming, com progresso retomável.
- Gerar contagens exatas quando viável e identificar claramente estimativas.
- Registrar tamanho, data e fingerprint dos arquivos analisados.
- Salvar somente métricas agregadas e parâmetros do método.
- Não incluir textos, URLs ou domínios reais nos resultados por padrão.

As saídas previstas são resumos estruturados, tabelas e figuras prontas para o
artigo, sempre identificadas como referentes a um conjunto parcial.
