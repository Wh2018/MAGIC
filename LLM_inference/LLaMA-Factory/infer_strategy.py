from transformers import AutoTokenizer
import transformers
import torch
import json
from tqdm import tqdm
import argparse

strategy_list = [
        "[question]",
        "[restatement or paraphrasing]",
        "[reflection of feelings]",
        "[self-disclosure]",
        "[affirmation and reassurance]",
        "[providing suggestions or information]",
        "[greeting]",
        "[others]"
]

def get_strategy(motivations):
    strategies = []
    for motivation in motivations:
        tag = 0
        for strategy in strategy_list:
            if strategy in motivation.lower():
                tag = 1
                strategies.append(strategy)
                break
        if tag == 0:
            strategies.append("[others]")
    return strategies

def inference(model_path, prompt_path, idx=0):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    with open(prompt_path, 'r') as f:
        datas = json.load(f)
        if idx == 0:
            prompts = [data['instruction'] for data in datas]
        else:
            prompts = datas

    motivations = []
    i = 0
    for prompt in tqdm(prompts):
        i = i + 1
        if i > 3:
            break
        sequences = pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            max_length=4096,
        )
        for seq in sequences:
            motivations.append(seq['generated_text'])
    
    strategies = get_strategy(motivations)
    assert len(motivations) == len(strategies)
    

    # save to json, ["strategy", "motivation"]
    json_datas = []
    for i in range(len(motivations)):
        motivation = motivations[i].split('\n')[-1]
        strategy = strategies[i]
        
        json_data = {"strategy": strategy, "motivation": motivation}
        json_datas.append(json_data)
    json_data = json.dumps(json_datas, ensure_ascii=False, indent=2, default=str)

    if idx == 0:
        with open('../strategy_with_motivation.json', 'w') as json_file:
            json_file.write(json_data)
    elif idx == 1:
        with open('../strategy_with_motivation_valid.json', 'w') as json_file:
            json_file.write(json_data)
    elif idx == 2:
        with open('../strategy_with_motivation_test.json', 'w') as json_file:
            json_file.write(json_data)
                
    
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_path', type=str, default='fine_tuned_llama2/')
    argparser.add_argument('--prompt_path', type=str, default='../prompts_4_LLaMA.json')
    argparser.add_argument('--prompt_path_valid', type=str, default='../prompts_4_LLaMA_valid.json')
    argparser.add_argument('--prompt_path_test', type=str, default='../prompts_4_LLaMA_test.json')
    args = argparser.parse_args()
    
    inference(args.model_path, args.prompt_path, 0)
    inference(args.model_path, args.prompt_path_valid, 1)
    inference(args.model_path, args.prompt_path_test, 2)