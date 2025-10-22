## Language Confusion Gate: Language-Aware Decoding Through Model Self-Distillation

- Link to Paper https://arxiv.org/abs/2510.17555

## Data Preparation
```
cd train
python prepare_train_dataset.py
```

## Training
```
cd train
python gate_train.py Qwen/Qwen3-4B-Instruct-2507 qwen3-4b-gate-norm
```

## Evaluation
Install lcg_plugin for inference of lcg in vllm, may also need to make a small change to vllm, instructions will be updated soon
```
cd lcg_plugin
pip install -e .
cd ../train
python prepare_eval_dataset.py
VLLM_USE_V1=0 TOK_PATH='Qwen/Qwen3-8B' python eval_gate.py flores-no-latin [MODEL_PATH]
python score_result flores [OUTPUT_PATH]
```

## TODO
- [ ] upload gate weights
- [ ] improve the code and Instruction
