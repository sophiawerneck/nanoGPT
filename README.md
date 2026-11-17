
# Group 7: Exploring Alternative MHA–FFN Arrangements in Decoder-Only Transformers


## Resumo do projeto

Este trabalho investiga o impacto de rearranjos estruturais das camadas de Atenção Multi-Cabeça (MHA) e Redes Feed-Forward (FFN) na arquitetura Transformer decoder-only. Avaliamos empiricamente quatro configurações: a convencional (MHA → FFN), a invertida (FFN → MHA), a paralela com LayerNorms independentes (MHA ∥ FFN) e a paralela com LayerNorm compartilhada. Os experimentos buscam manter a paridade de parâmetros (~125M) e utilizar os mesmos dados entre os modelos. Avaliamos o trade-off entre qualidade de modelagem (perda e perplexidade) e eficiência computacional (latência e speedup).

## Configuração do ambiente

### Requisitos

<!-- Informar os requisitos necessários para executar o projeto. -->

- Python: 3.13
- GPU NVIDIA com suporte a CUDA.
- Precisão numérica: FP16 para NVIDIA T4; BF16 pode ser utilizado em GPUs compatíveis, como A100.
- Aproximadamente 10 GB de espaço para os dados tokenizados.
- Espaço adicional para checkpoints, métricas e resultados.
- Acesso à internet na preparação dos dados, que baixa o C4 do Hugging Face em streaming.
- Dependências com versões fixadas em `requirements.txt` (incluindo bibliotecas fundamentais como `torch`, `tiktoken` e `simple-parsing` para estruturação de hiperparâmetros).

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
Por padrão, o experimento treina quatro variantes (seq, inv, par e par_shared), com três sementes cada (1337, 1338 e 1339), totalizando 12 execuções.

O pipeline realiza automaticamente a preparação dos dados, os treinamentos, a avaliação final, o benchmark de inferência, a geração de amostras de texto e a agregação dos resultados.

O script pode ser executado novamente após uma interrupção, retomando os treinamentos a partir dos checkpoints disponíveis e ignorando as execuções já concluídas. Variações de execução:

Variações de execução
```bash
SEEDS="1337" bash run_all.sh
# Executa apenas uma semente.

NPROC=1 bash run_all.sh
# Utiliza uma GPU por execução, sem DDP.

NPROC=4 bash run_all.sh
# Utiliza quatro GPUs por execução via DDP.

TRAIN_ARGS="--peak_tflops VALOR" bash run_all.sh
# Define o pico teórico utilizado no cálculo da MFU.

```
Para utilizar outra GPU, ajuste --peak_tflops conforme o modelo e a precisão numérica utilizados. Para utilizar múltiplas GPUs, ajuste NPROC conforme a quantidade de GPUs disponíveis.

### Passo a passo manual
Caso queira executar as etapas individualmente em vez de usar o run_all.sh, siga o fluxo abaixo:

#### Preparação dos dados
```bash
python data/c4/prepare.py
```
Gera train.bin e validation.bin a partir do C4 em inglês, utilizando o tokenizador BPE do GPT-2. A versão do dataset, as contagens e o SHA-256 de cada arquivo ficam armazenados nos respectivos arquivos de metadados.

#### Treinamento e avaliação 
A avaliação final da perplexidade é integrada ao script de treinamento e ocorre automaticamente na última iteração. Ela utiliza a porção holdout da validação, separada do subconjunto empregado no monitoramento periódico.
```bash
python train.py --block_type seq --seed 1337
python train.py --block_type inv --seed 1337
python train.py --block_type par --seed 1337
python train.py --block_type par_shared --seed 1337
```
Repita os quatro comandos com as sementes 1338 e 1339. Os checkpoints, as métricas de treinamento e os resultados da avaliação final são armazenados no diretório de cada execução.

Para retomar um treinamento interrompido a partir do último checkpoint salvo, adicione --init_from resume.
Cada execução grava seus artefatos, métricas e resultados de avaliação em out/c4/<variante>_s<semente>/.

#### Benchmark de Latência
Após o treinamento, meça a latência média de geração e o throughput de inferência para cada modelo salvo:
```bash
python bench.py --out_dir out/c4/seq_s1337
python bench.py --out_dir out/c4/inv_s1337
python bench.py --out_dir out/c4/par_s1337
python bench.py --out_dir out/c4/par_shared_s1337
```

Repita para as demais sementes. Os resultados são armazenados em benchmark.json.

#### Geração de Texto (Análise Qualitativa)
A geração de cinco amostras de texto já é executada automaticamente pelo run_all.sh, sendo armazenada em samples_text.txt no diretório de cada execução. Para realizar a geração manualmente:

```bash
python sample.py --init_from resume --out_dir out/c4/seq_s1337
```
É possível personalizar a geração por meio de argumentos, como --num_samples, --max_new_tokens e --temperature.

#### Agregação de Resultados
Por fim, consolide as métricas de treinamento, avaliação e benchmark das execuções em tabelas e gráficos:
```bash
python summarize.py --out_root out/c4
```
