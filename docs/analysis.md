# Metodologia

## Ordem das passagens

No perfil `tiny`, `sample` lê janelas determinísticas distribuídas por `pages.tsv`, projeta somente campos analíticos e fecha os domínios referenciados e seus pais. `verify` e as passagens seguintes usam exclusivamente os dois TSVs derivados. A origem é identificada por tamanho e mtime; os arquivos amostrados recebem SHA-256 completo.

1. `verify` valida largura/header, espaço, tamanho, mtime e SHA-256 das duas fontes efetivas. No `full`, a identidade usa o hash de páginas; no `tiny`, usa o hash combinado dos dois TSVs amostrados.
2. O inventário converte somente campos analíticos de domínio para Parquet e cria índice DuckDB em disco.
3. A primeira passagem de páginas mede todos os estados e, para `done`, decodifica o texto sem usar `text_md5` como filtro. Features e hashes são persistidos; texto descompactado não.
4. D1 usa SHA-256 dos bytes; D2 usa SHA-256 após normalização NFC e de layout. O menor `pages.id`, depois a linha, é o representante.
5. Fragmentos exatos intradomínio são candidatos quando atingem `max(5, min(100, ceil(0,001 × N)))`. Repetições cross-domain são apenas medidas.
6. Uma amostra determinística privada contém até 100 parágrafos e 100 blocos. A limpeza só é autorizada com precisão ≥95% e limite inferior de Wilson 95% ≥90%; `incerto` é erro.
7. A segunda passagem remove a união das ocorrências aprovadas, sem regra global para linhas curtas. Só textos vazios saem de `B_clean`; D3 deduplica novamente.
8. A passagem lexical calcula formas observadas e lemas com spaCy. Palavras são tokens alfabéticos; números e outros tokens ficam separados. Bigramas não atravessam parágrafos.
9. Frequências de formas/lemas usam spills e 256 partições; bigramas usam Space-Saving e recontagem exata. O top-K só é publicado quando o limite do item omitido fica abaixo do último item publicado.
10. `near-duplicates` é opcional: documentos com ≥50 palavras, shingles de cinco palavras, 112 MinHashes em 14×8 e confirmação por Jaccard ≥0,85.

## Atribuição a domínios

`parent_domain_id` representa origem da descoberta, não propriedade. Wikimedia é uma lista explícita de famílias de domínio, nunca busca pela substring `wiki`.

Conteúdo cross-domain possui duas leituras:

- `page-weighted`: descreve o volume efetivamente coletado;
- `unique-content`: divide cada unidade igualmente entre os domínios presentes no grupo.

Hosts reais podem aparecer apenas nessas tabelas agregadas. Textos e URLs não entram em resultados públicos.

## Léxico e figuras

`B_clean` é a visão principal. `R_valid` e `E_exact` quantificam o efeito de validação, deduplicação e limpeza. Stopwords participam dos bigramas e são removidas apenas da apresentação dos rankings.

As saídas incluem status, funil, quantis de tamanho, duplicação, remoção, rendimento por recursão, concentração por domínio, vocabulário, hapax, Zipf e rankings. Todas as figuras são geradas de CSVs agregados em `results/`, sem reler os TSVs.

Na prévia `tiny`, distribuições descrevem apenas a composição estratificada selecionada. Indicadores por `request_count` ficam como não aplicáveis porque não é válido combinar palavras amostradas com requisições integrais dos domínios.

A primeira versão não analisa tempo, segurança/ética, Heaps ou classificação completa de idioma. `blocked_language` permanece na distribuição de estados.

## Reprodutibilidade e falhas

Chunks são gravados como `.partial`, validados e renomeados antes de atualizar `state.json`. Representantes são listas `u64` ordenadas consultadas por merge. DuckDB recebe limites explícitos de memória e temporários; lotes spaCy são limitados por bytes. Mudança de fonte, configuração ou modelo invalida a retomada.

Parâmetros de near-duplicate seguem a configuração publicada do [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb/blob/v1.4.0/README.md); candidatos são confirmados por similaridade exata, em linha com [Lee et al. (2022)](https://aclanthology.org/2022.acl-long.577/).
