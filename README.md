
# Group 7: Exploring Alternative MHA–FFN Arrangements in Decoder-Only Transformers


## Resumo do projeto

Este trabalho investiga o impacto de rearranjos estruturais das camadas de Atenção Multi-Cabeça (MHA) e Redes Feed-Forward (FFN) na arquitetura Transformer decoder-only. Avaliamos empiricamente três configurações: a convencional (MHA → FFN), a invertida (FFN → MHA)  e a paralela (MHA ∥ FFN), paridade de parâmetros (~125M) e dados entre os modelos. Avaliamos o trade-off entre qualidade de modelagem (Perda e Perplexidade) e eficiência computacional (latência e speedup).

## Configuração do ambiente

### Requisitos

<!-- Informar os requisitos necessários para executar o projeto. -->

- Python: Python: 3.13
- GPU NVIDIA com suporte a CUDA.
- Precisão numérica: FP16 para NVIDIA T4; BF16 pode ser utilizado em GPUs compatíveis, como A100.
- Aproximadamente 10 GB de espaço para os dados tokenizados.
- Espaço adicional para checkpoints, métricas e resultados.
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

### Preparação dos dados
```bash
python data/c4/prepare.py
```
Gera data/c4/train.bin e data/c4/validation.bin a partir do C4 en, com o tokenizador BPE do GPT-2. A versão do dataset, as contagens e o SHA-256 de cada arquivo ficam em data/c4/*_meta.json.

### Treinamento

```bash
python train.py --block_type seq --seed 1337
python train.py --block_type inv --seed 1337
python train.py --block_type par --seed 1337
```
Repita os três comandos com --seed 1338 e --seed 1339. Para treinar em várias GPUs:
```bash
torchrun --standalone --nproc_per_node=4 train.py --block_type seq --seed 1337
```
Para retomar um treino interrompido do último checkpoint, adicione --init_from resume. Os hiperparâmetros padrão do train.py são os usados no relatório. Cada execução grava seus artefatos em out/<variante>_s<semente>/.


### Avaliação
Perplexidade final, calculada na porção da validação não usada durante o treino:
```bash
python evaluate.py --out_dir out/seq_s1337
python evaluate.py --out_dir out/inv_s1337
python evaluate.py --out_dir out/par_s1337
```

Latência de inferência (prefill e geração):
```bash
python benchmark.py --out_dir out/seq_s1337
python benchmark.py --out_dir out/inv_s1337
python benchmark.py --out_dir out/par_s1337
```
Repita para as demais sementes. Em seguida, agregue os resultados:
```bash
python summarize.py
```
## Resultados gerados
