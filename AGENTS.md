# AGENTS.md

## Objetivo

Analisar a prévia em evolução do dataset WackyWacky e produzir estatísticas
agregadas, reproduzíveis e adequadas para um artigo científico.

## Leitura obrigatória

- [README.md](README.md): escopo e estado do projeto.
- [docs/dataset.md](docs/dataset.md): estrutura e fatos conhecidos da fonte.
- [docs/analysis.md](docs/analysis.md): perguntas e princípios da análise.

## Regras

- Foque em `pages.tsv`, `domain.tsv` e na relação `pages.domain_id = domain.id`.
- Trate a fonte externa como somente leitura; não copie nem versione o dataset.
- Processe arquivos grandes por streaming e evite leituras integrais sem necessidade.
- Não exponha textos, URLs ou domínios reais em logs, testes ou documentos.
- Separe fatos medidos de estimativas e registre a data de corte do snapshot.
- Produza apenas resultados agregados e métodos reproduzíveis.
- Mantenha código e documentação diretos e pequenos.
- Atualize o `README.md` e os arquivos em `docs/` quando escopo, fonte, método ou
  resultados mudarem. Registre aqui qualquer novo documento obrigatório.
