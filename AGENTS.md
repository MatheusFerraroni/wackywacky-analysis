# AGENTS.md

Objetivo: caracterizar de forma agregada e reproduzível o snapshot parcial WackyWacky.

Leia `README.md`, `docs/dataset.md` e `docs/analysis.md`.

Regras:

- trate `pages.tsv` e `domain.tsv` como fontes externas imutáveis;
- processe em streaming e respeite os limites de memória e scratch da configuração;
- nunca publique textos ou URLs; hosts reais só podem aparecer em tabelas agregadas;
- separe `R_valid`, `E_exact`, `B_clean` e a sensibilidade opcional `N_near`;
- use apenas fixtures sintéticas nos testes;
- mantenha README e docs curtos e atualizados quando método, interface ou resultado mudar.
