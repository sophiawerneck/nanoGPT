
# Group 7: Exploring Alternative MHA–FFN Arrangements in Decoder-Only Transformers


## Resumo do projeto

Este trabalho investiga o impacto de rearranjos estruturais das camadas de Atenção Multi-Cabeça (MHA) e Redes Feed-Forward (FFN) na arquitetura Transformer decoder-only. Avaliamos empiricamente três configurações: a convencional (MHA → FFN), a invertida (FFN → MHA)  e a paralela (MHA ∥ FFN), paridade de parâmetros (~125M) e dados entre os modelos. Avaliamos o trade-off entre qualidade de modelagem (Perda e Perplexidade) e eficiência computacional (latência e speedup).

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
Por padrão, o experimento treina as três variantes ("seq", "inv" e "par") com três sementes cada (1337, 1338 e 1339), totalizando nove execuções.

Em uma GPU NVIDIA A100, estima-se aproximadamente 8 horas por execução, totalizando cerca de 72 GPU-horas. No Google Colab com NVIDIA T4, o tempo de execução será consideravelmente maior, podendo chegar a dezenas de horas por treinamento. O tempo efetivo depende das condições de execução e da disponibilidade dos recursos computacionais.

O script pode ser executado novamente após uma interrupção, retomando os treinamentos a partir dos checkpoints disponíveis e ignorando as etapas já concluídas. Variações:

Variações de execução
```bash
SEEDS="1337" bash run_all.sh
# Executa apenas uma semente.

NPROC=1 bash run_all.sh
# Utiliza uma GPU por execução, desativando o paralelismo via DDP, quando disponível.

TRAIN_ARGS="--peak_tflops 65" bash run_all.sh
# Define o pico teórico de 65 TFLOPS para NVIDIA T4 em FP16.
```
Para utilizar outra GPU, ajuste --peak_tflops conforme o modelo e a precisão numérica utilizados. Para utilizar múltiplas GPUs, ajuste NPROC conforme a quantidade de GPUs disponíveis.

### Passo a passo manual
Caso queira executar as etapas individualmente em vez de usar o run_all.sh, siga o fluxo abaixo:

#### Preparação dos dados
```bash
python data/c4/prepare.py
```
Gera data/c4/train.bin e data/c4/validation.bin a partir do C4 en, com o tokenizador BPE do GPT-2. A versão do dataset, as contagens e o SHA-256 de cada arquivo ficam armazenados em data/c4/*_meta.json.

#### Treinamento e avaliação 
A avaliação final da Perplexidade, calculada na porção holdout da validação (não vista durante o monitoramento), já está integrada e ocorre automaticamente na última iteração do script de treinamento.
```bash
python train.py --block_type seq --seed 1337
python train.py --block_type inv --seed 1337
python train.py --block_type par --seed 1337
```
Repita os três comandos com --seed 1338 e --seed 1339. Para treinar distribuído em várias GPUs de um único nó (DDP):
```bash
torchrun --standalone --nproc_per_node=4 train.py --block_type seq --seed 1337
```

Para retomar um treino interrompido a partir do último checkpoint salvo, adicione --init_from resume. Os hiperparâmetros padrão do train.py (estruturados via simple-parsing) refletem a configuração usada no relatório. Cada execução grava seus artefatos, métricas e o resumo da avaliação no diretório out/<variante>_s<semente>/.

#### Benchmark de Latência
Após o treinamento, meça a latência de inferência (prefill e geração) para cada modelo salvo:
```bash
python benchmark.py --out_dir out/seq_s1337
python benchmark.py --out_dir out/inv_s1337
python benchmark.py --out_dir out/par_s1337
```

(Repita para as demais sementes).

#### Geração de Texto (Análise Qualitativa)
Para ver os modelos em ação gerando texto livre e avaliar qualitativamente o que foi aprendido:

```bash
python sample.py --init_from resume --out_dir out/seq_s1337
```
(Você pode customizar a geração editando argumentos via terminal, ex: --max_new_tokens 200 --temperature 0.8).

#### Agregação de Resultados
​Por fim, consolide todas as métricas de Treino, Avaliação e Benchmark de todas as pastas em tabelas CSV, Markdown e gráficos:
```bash
python summarize.py
```
