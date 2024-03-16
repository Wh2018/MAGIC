from torch.utils.data import Dataset, DataLoader
import torch
import json
from time import sleep

def collate_fn(batch):
    """
    Convert data in a batch into tensor
    """
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    decoder_input_ids = [item['decoder_input_ids'] for item in batch]
    decoder_attention_mask = [item['decoder_attention_mask'] for item in batch]
    targets = [item['targets'] for item in batch]
    strategy_ids = [item['strategy_id'] for item in batch]
    motivation_ids = [item['motivation_ids'] for item in batch]
    motivation_mask = [item['motivation_mask'] for item in batch]
    
    #bs * []
    response_memory = [item['response_memory'] for item in batch]
    strategy_memory = [item['strategy_memory'] for item in batch]
    

    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'decoder_input_ids': torch.stack(decoder_input_ids),
        'decoder_attention_mask': torch.stack(decoder_attention_mask),
        'targets': torch.stack(targets),
        'strategy_ids': strategy_ids,
        'response_memory': response_memory,
        'strategy_memory': strategy_memory,
        'motivation_ids': torch.stack(motivation_ids),
        'motivation_mask': torch.stack(motivation_mask)
    }

def read_data(file_path, delimiter='\t'):
    """
    Read data from ESCONV, lookahead_version, train:val:test = 910:195:195

    Returns:
        Conversations: []
        {
            context: dialogue context, with n-1 turns
            response: n-th response, from sys
            strategy: n-th strategy, from sys
            response_memory: (n-1) / 2 response from sys
            strategy_memory: (n-1) / 2 strategy from sys of response
        }
    """
    with open(file_path, 'r') as f:
        conversations = json.load(f)
    
    return conversations


def filter_smaller_subsets(arrays):
    """
    Pick all the largest collections from response_memory and return the largest collection and its corresponding ID to retrieve the strategy
    """
    
    result_arrays = []
    result_indices = []

    array_dict = {tuple(array): index for index, array in enumerate(arrays)}

    for array in arrays:
        is_larger_subset = True

        for other_array in arrays:
            if array != other_array and is_subset(array, other_array):
                is_larger_subset = False
                break

        if is_larger_subset:
            result_arrays.append(array)
            result_indices.append(array_dict[tuple(array)])

    return result_arrays, result_indices

def is_subset(arr1, arr2):
    len_arr1 = len(arr1)
    len_arr2 = len(arr2)

    if len_arr1 > len_arr2:
        return False

    i = 0
    j = 0

    while i < len_arr1 and j < len_arr2:
        if arr1[i] == arr2[j]:
            i += 1
        j += 1

    return i == len_arr1


class DialogueDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_length=512):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.strategies = ['Question', 'Restatement or Paraphrasing', 'Reflection of Feelings', 'Self-disclosure', 'Affirmation and Reassurance', 'Providing Suggestions or Information', 'Greeting', 'Others']
    
    def find_strategy_index(self, strategy):
        lower_str = [s.lower() for s in self.strategies]
        strategy = strategy.lower()
        index = lower_str.index(strategy)
        return index
    
    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        context = conversation['context']
        response = conversation['response']
        motivation = conversation['motivation']
        strategy_id = self.find_strategy_index(conversation['strategy'])
        # [res1, res2, ...]
        response_memory = conversation['response_memory']
        # [id1, id2, ...]
        strategy_memory = [self.find_strategy_index(strategy) for strategy in conversation['strategy_memory']]


        # Use [SEP] to join multiple sentences in context list
        context_text = self.tokenizer.sep_token.join(context)
        inputs = self.tokenizer(
            context_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        
        #no padding, <s> x x x </s>
        decoder_inputs = self.tokenizer(
            response,
            max_length=self.max_length,
            #padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        decoder_ids = decoder_inputs['input_ids'].squeeze()
        decoder_attention_mask = decoder_inputs['attention_mask'].squeeze()

        pad_len = max(self.max_length - decoder_ids.size(0), 0) + 1
        pad_tensor = torch.full((pad_len, ), self.tokenizer.pad_token_id)

        #remove <s>, padding to max_length
        targets = torch.cat((decoder_ids[1:], pad_tensor), dim=0)
        #remove </s>, padding to max_length
        decoder_input_ids = torch.cat((decoder_ids[:-1], pad_tensor), dim=0)
        pad_mask_tensor = torch.zeros(pad_len)
        decoder_attention_mask = torch.cat((decoder_attention_mask[:-1], pad_mask_tensor), dim=0)

        assert len(decoder_input_ids) == len(targets)
        
        # tokenize motivation
        if motivation  == "":
            motivation = "No motivation provided"
            
        motivation_inputs = self.tokenizer(
            motivation,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask,
            'targets': targets.contiguous(),  # Shift the labels by one position
            'strategy_id': strategy_id,
            'response_memory': response_memory,
            'strategy_memory': strategy_memory,
            'motivation_ids': motivation_inputs['input_ids'].squeeze(),
            'motivation_mask': motivation_inputs['attention_mask'].squeeze()
        }