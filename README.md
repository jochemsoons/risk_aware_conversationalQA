# Reproduction study: Controlling the Risk of Conversational Search via Reinforcement Learning

### A risk-aware conversational search system consisting of pretrained answer and question rerankers and a decision maker trained by reinforcement learning.

This repository has been used to perform a reproduction study of the following paper:

    Wang, Z., & Ai, Q. (2021, April). Controlling the Risk of Conversational Search via Reinforcement Learning. In Proceedings of the Web Conference 2021 (pp. 1968-1977).

This repository is based on the repository made available by the authors, which can be found [through this link](https://github.com/zhenduow/conversationalQA). The original paper can be found [here](https://dl.acm.org/doi/abs/10.1145/3442381.3449893) and the reproduction study can be found [here](https://github.com/jochemsoons/risk_aware_conversationalQA/blob/main/Reproduction_Study__Controlling_the_Risk_of_Conversational_Search_via_Reinforcement_Learning.pdf).


## Requirements
In the ParlAi folder, a requirements.txt file can be found that can be used to install required packages. We suggest to use an [Anaconda](https://docs.conda.io/en/latest/) environment. That can be installed using the environment.yml file in this repository:

```
$ conda env create -f environment.yml
$ conda activate risk_aware_agent
```

## How to use
There are three steps to reproduce results of the original paper and our reproduction study:

### 1. Preprocess data. 

Here we use [MSDialog dataset](https://ciir.cs.umass.edu/downloads/msdialog/) as example. You can also set dataset_name to be 'UDC' for [Ubuntu Dialog Corpus](http://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/) or 'opendialkg' for [Opendialkg](https://github.com/facebookresearch/opendialkg).
 
 First, download MSDialog-Complete.json into /data.
 
```
$ cd data
$ python3 data_processing.py --dataset_name MSDialog
```

 This will process and filter the data. All conversations that meet the filtering criterion are saved in MSDialog-Complete and will be automatically split into training and testing set. The others are save in MSDialog-Incomplete. The former is used for the main experiments and the latter is used for fine-tuning the rerankers only. The data processing code uses `random.seed(2020)` to fix the result of data generation.
    
### 2. Fine-tune the pretrained reranker 

Fine-tune the rerankers on the answer and question training samples (MSDialog as example). The training of the rerankers is based on [ParlAI]       (https://github.com/facebookresearch/ParlAI)
```
$ cd ParlAI
$ python3 -u examples/train_model.py \
    --init-model zoo:pretrained_transformers/poly_model_huge_reddit/model \
    -t fromfile:parlaiformat --fromfile_datapath ../data/MSDialog-parlai-answer \
    --model transformer/polyencoder --batchsize 4 --eval-batchsize 100 \
    --warmup_updates 100 --lr-scheduler-patience 0 --lr-scheduler-decay 0.4 \
    -lr 5e-05 --data-parallel True --history-size 20 --label-truncate 72 \
    --text-truncate 360 --num-epochs 12.0 --max_train_time 200000 -veps 0.5 \
    -vme 8000 --validation-metric accuracy --validation-metric-mode max \
    --save-after-valid True --log_every_n_secs 20 --candidates batch --fp16 True \
    --dict-tokenizer bpe --dict-lower True --optimizer adamax --output-scaling 0.06 \
    --variant xlm --reduction-type mean --share-encoders False \
    --learn-positional-embeddings True --n-layers 12 --n-heads 12 --ffn-size 3072 \
    --attention-dropout 0.1 --relu-dropout 0.0 --dropout 0.1 --n-positions 1024 \
    --embedding-size 768 --activation gelu --embeddings-scale False --n-segments 2 \
    --learn-embeddings True --polyencoder-type codes --poly-n-codes 64 \
    --poly-attention-type basic --dict-endtoken __start__ \
    --model-file zoo:pretrained_transformers/model_poly/answer \
    --ignore-bad-candidates True  --eval-candidates batch
```
```
$ python3 -u examples/train_model.py \
    --init-model zoo:pretrained_transformers/poly_model_huge_reddit/model \
    -t fromfile:parlaiformat --fromfile_datapath ../data/MSDialog-parlai-question \
    --model transformer/polyencoder --batchsize 4 --eval-batchsize 100 \
    --warmup_updates 100 --lr-scheduler-patience 0 --lr-scheduler-decay 0.4 \
    -lr 5e-05 --data-parallel True --history-size 20 --label-truncate 72 \
    --text-truncate 360 --num-epochs 12.0 --max_train_time 200000 -veps 0.5 \
    -vme 8000 --validation-metric accuracy --validation-metric-mode max \
    --save-after-valid True --log_every_n_secs 20 --candidates batch --fp16 True \
    --dict-tokenizer bpe --dict-lower True --optimizer adamax --output-scaling 0.06 \
    --variant xlm --reduction-type mean --share-encoders False \
    --learn-positional-embeddings True --n-layers 12 --n-heads 12 --ffn-size 3072 \
    --attention-dropout 0.1 --relu-dropout 0.0 --dropout 0.1 --n-positions 1024 \
    --embedding-size 768 --activation gelu --embeddings-scale False --n-segments 2 \
    --learn-embeddings True --polyencoder-type codes --poly-n-codes 64 \
    --poly-attention-type basic --dict-endtoken __start__ \
    --model-file zoo:pretrained_transformers/model_poly/question \
    --ignore-bad-candidates True  --eval-candidates batch
```

This will download the poly-encoder checkpoints pretrained on the huge reddit dataset and fine-tune it on our preprocessed dataset. The fine-tuned model is save in ParlAI/data/models/pretrained_transformers/model_poly/.
    
If you get an error of dictionary size mismatching, this is because that the pretrained model checkpoints has a dictionary that's larger than the fine-tune dataset. To solve this problem, before running the fine-tuning script, copy the downloaded pretrained dict file `ParlAI/data/models/pretrained_transformers/poly_model_huge_reddit/model.dict` to `ParlAI/data/models/pretrained_transformers/model_poly/` twice and rename them to `answer.dict` and `question.dict`. Then run the above fine-tuning script. Perform the same steps for the bi-directional encoder (i.e. copy the model.dict file twice from the ./bi_model_huge_reddit folder to the model_bi folder, and name them again answer.dict and question.dict).

Now you can run the previous and following scripts without getting an error of dictionary size mismatching:
```
$ cd ParlAI
$ python3 -u examples/train_model.py \
    --init-model zoo:pretrained_transformers/bi_model_huge_reddit/model \
    -t fromfile:parlaiformat --fromfile_datapath ../data/MSDialog-parlai-answer \
    --model transformer/biencoder --batchsize 4 --eval-batchsize 100 \
    --warmup_updates 100 --lr-scheduler-patience 0 \
    --lr-scheduler-decay 0.4 -lr 5e-05 --data-parallel True \
    --history-size 20 --label-truncate 72 --text-truncate 360 \
    --num-epochs 12.0 --max_train_time 200000 -veps 0.5 -vme 8000 \
    --validation-metric accuracy --validation-metric-mode max \
    --save-after-valid True --log_every_n_secs 20 --candidates batch \
    --dict-tokenizer bpe --dict-lower True --optimizer adamax \
    --output-scaling 0.06 \
    --variant xlm --reduction-type mean --share-encoders False \
    --learn-positional-embeddings True --n-layers 12 --n-heads 12 \
    --ffn-size 3072 --attention-dropout 0.1 --relu-dropout 0.0 --dropout 0.1 \
    --n-positions 1024 --embedding-size 768 --activation gelu \
    --embeddings-scale False --n-segments 2 --learn-embeddings True \
    --share-word-embeddings False --dict-endtoken __start__ --fp16 True \
    --model-file zoo:pretrained_transformers/model_bi/answer\
    --ignore-bad-candidates True  --eval-candidates batch
```
```
$ python3 -u examples/train_model.py \
    --init-model zoo:pretrained_transformers/bi_model_huge_reddit/model \
    -t fromfile:parlaiformat --fromfile_datapath ../data/MSDialog-parlai-question \
    --model transformer/biencoder --batchsize 4 --eval-batchsize 100 \
    --warmup_updates 100 --lr-scheduler-patience 0 \
    --lr-scheduler-decay 0.4 -lr 5e-05 --data-parallel True \
    --history-size 20 --label-truncate 72 --text-truncate 360 \
    --num-epochs 12.0 --max_train_time 200000 -veps 0.5 -vme 8000 \
    --validation-metric accuracy --validation-metric-mode max \
    --save-after-valid True --log_every_n_secs 20 --candidates batch \
    --dict-tokenizer bpe --dict-lower True --optimizer adamax \
    --output-scaling 0.06 \
    --variant xlm --reduction-type mean --share-encoders False \
    --learn-positional-embeddings True --n-layers 12 --n-heads 12 \
    --ffn-size 3072 --attention-dropout 0.1 --relu-dropout 0.0 --dropout 0.1 \
    --n-positions 1024 --embedding-size 768 --activation gelu \
    --embeddings-scale False --n-segments 2 --learn-embeddings True \
    --share-word-embeddings False --dict-endtoken __start__ --fp16 True \
    --model-file zoo:pretrained_transformers/model_bi/question\
    --ignore-bad-candidates True  --eval-candidates batch
```
    
The fine-tuning code is based on [ParlAI poly-encoder](https://github.com/facebookresearch/ParlAI/tree/master/projects/polyencoder/), but we modify several scripts for our needs. We do not recommended downloading the original ParlAI code and replace the ParlAI folder in this program. The original training of the encoders are done on 8 x GPU 32GB. We decrease the batch size and therefore the code is able to run it on 4 x GPU 11GB (GeForce RTX 2080Ti).
    
  
### 3.1 Run the main experiments. 

To run the experiments, use the following code:

```
$ python3  run_sampling.py --dataset_name MSDialog --reranker_name Poly --topn 1 --cv 0 > your_log_file
```
    
- `--dataset_name` can be 'MSDialog', 'UDC', or 'Opendialkg' currently.
- `--reranker_name` can be 'Poly' or 'Bi' currently.
- `--topn` means the top n reranked candidates are considered correct, i.e. `--topn ` computes recall@1. 
- `--cv` selects the cross validation fold. Because the MSDialog is small it is adviced to use cross validation. `--cv` can be set to 0,1,2,3,4 for the MSDialog dataset. Set `--cv -1` if you want to turn of cross validation. 
- `--n_epochs` sets the amount of epochs. 
- `--batch_size` sets the batch size. 
- `--cq_reward` sets the reward if a correct question is asked. 
- `--user_patience` corresponds to the maximum amount of turns between user and agent before the user leaves the conversation. 
- `--user_tolerance` corresponds to the amount of bad questions that can be asked before the user leaves the conversation. 
- `--seed` is an optional argument that sets the seed to run the experiments. 
- `--path_to_parlai` should correspond to the path to the ParlAI map.
    
The experiment would take a couple of hours to one day. So, it's recommended to save the results to a log file (add `> your_log_file` to your command).

### 3.2 Run the BM25 negative sampling experiments. To run the experiments, use the following code:
To run the extension of our reproduction study that uses globally sampled BM25 negatives, first install bm25 using:

`pip install rank-bm25 (0.2.1)`

Then you can run the code by:

```
$ python3  run_sampling_bm25.py --dataset_name MSDialog --reranker_name Poly --topn 1 --cv 0 > your_log_file
```

Similarly to the main experiments, arguments can be adjusted.
    
    
## Reference

Please cite the work of Wang and Ai if you use this code repository in your work:

```
@misc{wang2021controlling,
      title={Controlling the Risk of Conversational Search via Reinforcement Learning}, 
      author={Zhenduo Wang and Qingyao Ai},
      year={2021},
      eprint={2101.06327},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```
