# MAGIC: Memory-enhanced Emotional Support Conversations with Motivation-driven Strategy Inference

This repository contains the **anonymous codes** for our **ECML-PKDD 2024 submitted paper**: *MAGIC: Memory-enhanced Emotional Support Conversations with Motivation-driven Strategy Inference*.

**Anonymous authors**


# Requirements
- Python 3.9
- CUDA 11.8
- PyTorch 2.2.1
- transformers 4.38.1
- tqdm 4.66.2
- openai 1.13.3
- datasets 2.18.0
- accelerate 0.28.0
- peft 0.9.0
- trl 0.7.11
- gradio 3.38.0
- sentencepiece 0.1.99
- protobuf 4.25.3
- nltk 3.8.1

You can install the required packages by
`pip install -r requirements.txt`

All experiments are conducted on a single NVIDIA Tesla A100-80GB GPU.

# Data Preparation
We follow the original division of the ESConv dataset for training, validation, and testing, with a ratio of 8:1:1.

For strategy annotations, we use the optimized version provided by MultiESC, a typical method for ESC systems.

We have downloaded the `train.txt`, `valid.txt`, and `test.txt` files from the [MultiESC](https://github.com/lwgkzl/MultiESC) repository and place them in the `data/` folder.

# Motivation-driven Support Strategy Inference
Enter `LLM_inference/` directory and follow the following steps to fine-tune LLaMA2 for Motivation-driven strategy inference.

## Step 1: Collection of strategy inference motivations

Run following command to collect strategy inference motivations using ChatGPT API (*``gpt-3.5-turbo-0125``* verison) based on the dialogue context and corresponding response in the training data of ESConv `data/train.txt`. 

```python
python collect_motivations_ChatGPT.py --openai_api_key <your_openai_api_key>
```

Please replace `<your_openai_api_key>` with your OpenAI API key from [OpenAI API](https://platform.openai.com/api-keys)

The generated motivations will be saved in `motivations_4_LLaMA.json`. We have released the generated motivation file for reproducibility.

Create prompts for fine-tuning LLaMA2 using following command.

```python
python create_promots_4_LLaMA.py
```

The generated prompts will be saved in `prompts_4_LLaMA.json`, `prompts_4_LLaMA_valid.json`, and `prompts_4_LLaMA_test.json` files for training, validation and testing data in ESConv, respectively. We have released these three files for reproducibility.



## Step 2: Fine-tuning LLaMA2 for motivation-driven strategy inference

Enter our cloned `LLaMA-Factory/` directory from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) repository, and follow the following steps to fine-tune LLaMA2 for motivation-driven strategy inference.

Place the generated motivation file `prompts_4_LLaMA.json` in the `LLaMA-Factory/data/` directory. 

Add following lines to the `LLaMA-Factory/data/dataset_info.json` file:

```json
"strategy_motivation": {
        "file_name": "prompts_4_LLaMA.json",
        "columns": {
          "prompt": "instruction",
          "response":"output"
        }
}
```

Dowdload LLaMA2 model `llama2-7B-hf` from [huggingface](https://huggingface.co/meta-llama/Llama-2-7b-hf) and place it in `LLaMA-Factory/llama2-7B-hf/` directory.

Run following command to fine-tune LLaMA2 with motivation data:

```python
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path llama2-7B-hf/ \ 
    --dataset strategy_motivation \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir output/ \
    --overwrite_cache \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 5e-5 \
    --num_train_epochs 5.0 \
    --plot_loss \
    --fp16
```

The model fine-tuned by LoRA will be saved in the `LLaMA-Factory/output/` directory. Running following command to merge LoRA weights and export the final model.

```python
python src/export_model.py \
    --model_name_or_path llama2-7B-hf/ \
    --adapter_name_or_path output/checkpoint-200/ \  
    --template default \
    --finetuning_type lora \
    --export_dir  fine_tuned_llama2/ \ 
    --export_size 2 \
    --export_legacy_format False
```

Please set `adapter_name_or_path` to your saved checkpoint directory.

The merged fine-tuned LLaMA2 model for motivation-driven strategy inference will be saved in the `LLaMA-Factory/fine_tuned_llama2/` directory.


## Step 3: Infer support strategies with motivations using fine-tuned LLaMA2

Run following command to infer support strategies with motivations using fine-tuned LLaMA2:

```python
python infer_strategy.py
```

The inferred strategies with motivations will be saved in `LLM_inference/strategy_with_motivation.json`, `LLM_inference/strategy_with_motivation_valid.json`, and `LLM_inference/strategy_with_motivation_test.json` files for training, validation, and testing data in ESConv, respectively.

# MAGIC Model Training and Evaluation

Please return to project root directory.

## Step 1: Data Preprocess and Pre-trained Model Preparation

Run the following command to preprocess the data in ESConv, with the following options:
- Update ``strategy`` and ``motivation`` fields in the original data files with inferred strategies and motivations by LLaMA2.
- Create `response_memory` and `strategy_memory` data fields to store responses and corresponding strategies in the dialogue history for each dialogue sample, for the training of strategy memory store.
- Convert data files with ``txt`` format to ``json`` format.

```python
python data/add_memory_motivation.py
```

The preprocessed data files will be saved in `data/train.json`, `data/valid.json`, and `data/test.json`.

Please download the pre-trained checkpoint of `bart-base` from [huggingface](https://huggingface.co/facebook/bart-base) and place it in the `src/bart_base/` directory.

## Step 2: MAGIC Model Training

Running following command to train the MAGIC model.

```python
CUDA_VISIBLE_DEVICES=0 python main.py \
    --train_data data/train.json \
    --valid_data data/valid.json \
    --output_dir out_model/ \
    --batch_size 16 \
    --num_epochs 20 \
    --learning_rate 5e-5 \
    --warmup_steps 0 \
    --bart_base_dir src/bart_base/ \
    --l_strategy 0.2 \
    --max_length 512 \
    --strategy_memory_max 128 \
    --max_length_res_memory 64
```

The training process will stop when the validation perplexity does not improve for 5 epochs.

After training, the model will be saved in `out_model` directory, organized in the following directory.

```python
├── out_model
│    ├── bs_16_epochs_20_20240213_152303  #16: batch size, 20: epochs, 20240213_152303: timestamp
│    │    ├── epoch_0_val_ppl_18.3 # 0: epoch, 18.3: validation perplexity
│    │    │     ├── dialogue_model.pth # dialogue model
│    │    │     ├── memory_bank.pth # memory bank
│    │    │     ├── merges.txt # the following 4 files for BART tokenizer
│    │    │     ├── special_tokens_map.json
│    │    │     ├── tokenizer_config.json
│    │    │     └── vocab.json
```

## Step3: MAGIC Model Evaluation
Download the evaluation script from [Google Drive](https://drive.google.com/file/d/1AFE2B7dYw9mU4rLEN4k7BMrtOxIlhXYh/view?usp=sharing) and place it in the ``metric/`` directory.

Running following command to evaluate the MAGIC model on the test data.

```python
CUDA_VISIBLE_DEVICES=0 python test_evaluation.py \
    --test_data data/test.json \ 
    --model_path your_model_path \
    --bart_base_dir src/bart_base/ \
    --batch_size 1 \
    --num_beams 5 \
    --max_length 512 \
    --max_length_res 64 \
    --strategy_memory_max 128 \
    --max_length_res_memory 64 \
    --diversity_penalty 1.0 \
    --length_penalty 1.0
```

Set ``model_path`` to your saved model directory. e.g., `out_model/bs_16_epochs_20_20240213_152303/epoch_0_val_ppl_18.3/`.

The evaluation results will be saved in `eval_result/bs_x_epochs_y_timestamp/diversity_z_length_w.txt` file. e.g., `eval_result/bs_16_epochs_20_20240213_152303/diversity_1.0_length_1.0.txt`.

The automatic evaluation metrics will be printed on the screen, including `BLEU-1`, `BLEU-2`, `BLEU-3`, `BLEU-4`, `METEOR`, `ROUGE-L`, `CIDEr`.
