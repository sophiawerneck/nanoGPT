
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

Após receber os arquivos do projeto, acesse o diretório raiz pelo terminal e instale as dependências:
```bash
pip install -r requirements.txt
```

<!-- Incluir outras instruções necessárias para configurar
     o ambiente e acessar os recursos utilizados. -->

## Reprodução dos resultados

### Execução automatizada
Os resultados do relatório podem ser reproduzidos por meio de um único comando, que executa as etapas do experimento em sequência:

```bash
bash run_all.sh
```
Por padrão, o experimento treina quatro variantes (seq, inv, par e par_shared), com três sementes cada (1337, 1338 e 1339), totalizando 12 execuções.

O pipeline realiza automaticamente a preparação dos dados, o treinamento das variantes para cada semente, a avaliação final, o benchmark de inferência, a geração de 5 amostras de texto e a agregação dos resultados.

Os checkpoints, as métricas e os resultados de cada execução são armazenados em seus respectivos diretórios. As amostras geradas são salvas em samples_text.txt, e os resultados do benchmark, em benchmark.json.

O script pode ser executado novamente após uma interrupção. Quando encontra um summary.json, ignora o treinamento daquela execução. Se encontra um checkpoint, mas não o resumo final, retoma o treinamento a partir do checkpoint disponível. Caso contrário, inicia o treinamento do zero. O benchmark e a geração de amostras são executados novamente, inclusive para treinamentos já concluídos.

### Variações de execução

É possível alterar a configuração do experimento por meio das seguintes variáveis:

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
Caso queira executar as etapas individualmente, sem utilizar o run_all.sh, siga o fluxo abaixo.

#### Preparação dos dados
```bash
python data/c4/prepare.py
```
Gera train.bin e validation.bin a partir do C4 em inglês, utilizando o tokenizador BPE do GPT-2. A versão do dataset, as contagens e o SHA-256 de cada arquivo ficam armazenados nos respectivos arquivos de metadados.

#### Treinamento e avaliação 
A avaliação final da perplexidade é integrada ao script de treinamento e ocorre automaticamente na última iteração. Ela utiliza a porção holdout da validação, separada do subconjunto empregado no monitoramento periódico.

Execute os comandos para cada variante:

```bash
python train.py --block_type seq --seed 1337
python train.py --block_type inv --seed 1337
python train.py --block_type par --seed 1337
python train.py --block_type par_shared --seed 1337
```
Repita os quatro comandos com as sementes 1338 e 1339.

Para retomar manualmente um treinamento interrompido a partir do último checkpoint salvo, adicione --init_from resume. 

Por exemplo:
```bash
python sample.py --init_from resume --out_dir out/c4/seq_s1337 --num_samples 5
```

Cada execução grava seus checkpoints, métricas e resultados de avaliação em out/c4/<variante>_s<semente>/.
Para treinar com múltiplas GPUs em um único nó, utilize torchrun. Por exemplo:
```bash
torchrun --standalone --nproc_per_node=4 train.py --block_type seq --seed 1337
```

#### Benchmark de Latência
Após o treinamento, execute os seguintes comandos:
```bash
python bench.py --out_dir out/c4/seq_s1337
python bench.py --out_dir out/c4/inv_s1337
python bench.py --out_dir out/c4/par_s1337
python bench.py --out_dir out/c4/par_shared_s1337
```

Repita para as demais sementes. Os resultados são armazenados em benchmark.json.

#### Geração de Texto (Análise Qualitativa)
Para gerar amostras manualmente, execute:

```bash
python sample.py --init_from resume --out_dir out/c4/seq_s1337
```
É possível personalizar a geração por meio de argumentos como --num_samples, --max_new_tokens e --temperature. A geração manual precisa ser executada para cada modelo que se deseja analisar.

#### Agregação de Resultados
Por fim, consolide os resultados das execuções:
```bash
python summarize.py --out_root out/c4
```
