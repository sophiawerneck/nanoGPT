
# Group 7: Exploring Alternative MHA–FFN Arrangements in Decoder-Only Transformers


## Resumo do projeto

Este trabalho investiga o impacto de rearranjos estruturais das camadas de Atenção Multi-Cabeça (MHA) e Redes Feed-Forward (FFN) na arquitetura Transformer decoder-only. Avaliamos empiricamente três configurações: a convencional (MHA → FFN), a invertida (FFN → MHA)  e a paralela (MHA ∥ FFN), paridade de parâmetros (~125M) e dados entre os modelos. Avaliamos o trade-off entre qualidade de modelagem (Perda e Perplexidade) e eficiência computacional (latência e speedup).

## Configuração do ambiente

### Requisitos

<!-- Informar os requisitos necessários para executar o projeto. -->

- Python: Python: 3.13
- GPU NVIDIA com suporte a CUDA. Os experimentos usam bfloat16, disponível a partir da arquitetura Ampere (ex.: A100).
- ~10 GB de disco para os dados tokenizados e ~1,5 GB por checkpoint.
- Acesso à internet na preparação dos dados, que baixa o C4 do Hugging Face em streaming.
- Dependências com versões fixadas em requirements.txt.

### Instalação

Clonar o repositório:

```bash
git clone https://github.com/sophiawerneck/NOME_DO_REPOSITORIO/
cd <NOME_DO_REPOSITORIO>
```

Instalar as dependências:

```bash
pip install -r requirements.txt
```

<!-- Incluir outras instruções necessárias para configurar
     o ambiente e acessar os recursos utilizados. -->

## Reprodução dos resultados

<!-- Apresentar os comandos exatos, na ordem de execução,
     necessários para reproduzir os principais resultados do projeto. -->
Todos os resultados do relatório são reproduzidos por um único comando, que executa as etapas abaixo em ordem:
```bash
bash run_all.sh
```
Por padrão, ele treina as 3 variantes (seq, inv, par) com 3 sementes cada (1337, 1338, 1339), o que leva cerca de 8 h por execução numa A100 (~72 GPU-horas no total). O script pode ser executado de novo após uma interrupção: ele pula as etapas concluídas e retoma as interrompidas. Variações:

### Treinamento

```bash
<comando_exato_de_treinamento>
```

### Avaliação

```bash
<comando_exato_de_avaliacao>
```

<!-- Informar como acessar os resultados gerados, caso necessário. -->
