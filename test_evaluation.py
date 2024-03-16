import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartPretrainedModel, BartModel, BartConfig
from transformers import get_linear_schedule_with_warmup
from torch.nn import functional as F
from argparse import ArgumentParser
from tqdm import tqdm
from src.bart_dialog import DialogueModel
from src.dataset import DialogueDataset, collate_fn, read_data
import datetime
import os
import logging
import numpy as np
import json
from torch.nn.functional import log_softmax
import nltk
import re

def add_spaces_before_punctuation(text):
    # add spaces before punctuation marks
    text = re.sub(r'(\w)([.,!?;:])', r'\1 \2', text)
    return text

def data_clean(result_file):
    with open(result_file, 'r') as file:
        content = file.read()
    
        content = add_spaces_before_punctuation(content)
        # remove multiple .,!?, ;
        content = re.sub(r'\.+', '.', content)
        content = re.sub(r'\!+', '!', content)
        content = re.sub(r'\"', '', content)

    with open(result_file, 'w') as file:
        file.write(content)

def clac_metric(result_file, label_file, no_glove=True):
    """
    Automatic evaluation metrics for dialogue response generation.
    Return: metric_res
        BLEU-1, BLEU-2, BLEU-3, BLEU-4, CIDEr, ROUGE-L, and METEOR
    """
    with open(result_file, 'r') as f:
        hypothesis = f.readlines()

    with open(label_file, 'r') as f:
        references = f.readlines()
    
    ref_list = []
    hyp_list = []
    for ref, hyp in zip(references, hypothesis):
        ref = ' '.join(nltk.word_tokenize(ref.rstrip('\n').lower()))
        hyp = ' '.join(nltk.word_tokenize(hyp.rstrip('\n').lower()))
        if len(hyp) == 0:
            hyp = '&'
        ref_list.append(ref)
        hyp_list.append(hyp)

    from metric import NLGEval
    metric = NLGEval(no_glove=no_glove)
    metric_res, metric_res_list = metric.compute_metrics([ref_list], hyp_list)
    for metric, score in metric_res.items():
        score = score * 100
        print(f"{metric}: {score:.2f}")
    return metric_res

def beam_search(model, input_ids, attention_mask, strategy_ids, response_memory, strategy_memory, beam_size, max_length, tokenizer, device, bos_token_id, eos_token_id, diversity_penalty=1.0, length_penalty=1.0):
    # Initialize the starting input_ids and attention_mask for the decoder
    decoder_input_ids = torch.tensor([[bos_token_id]]).to(device)
    decoder_attention_mask = torch.ones(decoder_input_ids.shape).to(device)

    # Initialize the starting logits for the beam search
    logits =  model(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, strategy_ids, response_memory, strategy_memory, stage='test')
    #logits = logits.squeeze(1)
    logits = logits[0, -1, :]
    logits = logits.softmax(-1)
    top_logits, top_indices = logits.topk(beam_size)


    # Initialize the starting beams
    beams = []
    for i in range(beam_size):
        beams.append(([top_indices[i].item()], top_logits[i].item()))

    # Perform beam search
    for _ in range(max_length):
        new_beams = []
        for beam in beams:
            decoder_input_ids = torch.tensor([beam[0]]).to(device)
            decoder_attention_mask = torch.ones(decoder_input_ids.shape).to(device)
            logits = model(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, strategy_ids, response_memory, strategy_memory, stage='test')
            #logits = logits.squeeze(1)
            logits = logits[0, -1, :]
            logits = logits.softmax(-1)
            top_logits, top_indices = logits.topk(beam_size)
            for i in range(beam_size):
                new_beams.append((beam[0] + [top_indices[i].item()], beam[1] + top_logits[i].item()))
        
        # Apply diversity penalties and length penalties
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = []
        for i in range(beam_size):
            beam = new_beams[i]
            penalty = (diversity_penalty * len(set(beam[0])) / len(beam[0]) + length_penalty * len(beam[0]) / max_length)
            beams.append((beam[0], beam[1] / penalty))
        
        # Prune the beams based on the total log probability
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    # Convert the top beam to a string and return it
    top_beam = beams[0][0]
    top_beam = top_beam[1:]
    print(top_beam)
    response = tokenizer.decode(top_beam, skip_special_tokens=True)
    return response

# test function
def test_epoch(model, tokenizer, dataloader, device, config, num_beams=5, max_length=64, diversity_penalty=1.0, length_penalty=1.0):

    results = []
    i = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            i = i +1 
            if i > 5:
                break
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            strategy_ids = batch['strategy_ids']
            response_memory = batch['response_memory']
            strategy_memory = batch['strategy_memory']
            
            result = beam_search(model, input_ids, attention_mask, strategy_ids, response_memory, strategy_memory, beam_size=num_beams, max_length=max_length, tokenizer=tokenizer, device=device, bos_token_id=config.bos_token_id, eos_token_id=config.eos_token_id, diversity_penalty=diversity_penalty, length_penalty=length_penalty)
            
            results.append(result)
            print(result)

    return results
    
def save_results(results, response_lables, result_dir, diversity_penalty, length_penalty):
    """
    save response labels and test results
    calculate aumomatic evaluation metrics
    """
    
    if not os.path.exists('eval_result'):
        os.makedirs('eval_result')
    
    with open('eval_result/response_lables.txt', 'w') as f:
        for label in response_lables:
            f.write(label + '\n')
    
    result_root_dir = 'eval_result/' + result_dir
    
    if not os.path.exists(result_root_dir):
        os.makedirs(result_root_dir)
    
    result_dir = result_root_dir + '/' + 'diversity_' +  str(diversity_penalty) + '_length_' + str(length_penalty) + '.txt'
    with open(result_dir, 'w') as f:
        for result in results:
            f.write(result + '\n')
    
    
    # calculate aumomatic evaluation metrics
    data_clean(result_dir)
    metric_res = clac_metric(result_dir, 'eval_result/response_lables.txt')

def main():
    parser = ArgumentParser()
    parser.add_argument("--test_data", type=str, default='data/test.json', help="Path to test data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--bart_base_dir", type=str, default='src/bart_base/', help="bart_base_dir")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_beams", type=int, default=5, help="Num beams")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--max_length_res", type=int, default=64, help="Max sequence length for response")
    parser.add_argument("--strategy_memory_max", type=int, default=128, help="Max num in the strategy memory")
    parser.add_argument("--max_length_res_memory", type=int, default=64, help="Max sequence length for response in strategy memory")
    parser.add_argument("--diversity_penalty", type=float, default=1.0, help="Diversity penalty for beam search")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="Length penalty for beam search")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BartTokenizer.from_pretrained(args.model_path)
    
    
    config = BartConfig.from_pretrained(args.bart_base_dir)
    
    test_data = read_data(args.test_data)
    test_dataset = DialogueDataset(test_data, tokenizer, args.max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    bart_model = BartModel.from_pretrained(args.bart_base_dir)
    model = DialogueModel(config, bart_model.config, tokenizer, args.max_length, args.strategy_memory_max, args.max_length_res_memory, device)
    
    
    #load pretrained checkpoints
    model.load_state_dict(torch.load(args.model_path + '/dialogue_model.pth'))
    model.memory_bank = torch.load(args.model_path + '/memory_bank.pth')
    
    model.to(device)
    
    response_lables = []
    # get response lables
    for conversation in test_data:
        response = conversation['response']
        response_lables.append(response)
    
    results = test_epoch(model, tokenizer, test_dataloader, device, config, num_beams=args.num_beams, max_length=args.max_length_res, diversity_penalty=args.diversity_penalty, length_penalty=args.length_penalty)
    
    result_start = args.model_path.find("out_model/")
    if result_start != -1:
        result_path = args.model_path[result_start+10:result_start+10+31]
    else:
        result_path = args.model_path
    
    save_results(results, response_lables, result_path, args.diversity_penalty, args.length_penalty)
 
    print('Test Done!')
    

if __name__ == "__main__":
    main()
