import torch
from torch import nn
from transformers import BartTokenizer, BartPreTrainedModel, BartModel, BartConfig
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder
from torch.nn import functional as F
from .dataset import filter_smaller_subsets
from time import sleep
from collections import OrderedDict
import copy

def BuildMemoryDict(response_memory, strategy_memory):
    """
    Group response_memory and strategy_memory by strategy_id and put responses with the same strategy_id together
    """
    
    filter_response, filter_index = filter_smaller_subsets(response_memory)
    filter_strategy = [strategy_memory[i] for i in filter_index]
    
    # Organise responses according to strategy
    response_memory = []
    strategy_memory = []

    for i in range(len(filter_response)):
        for filter_res in filter_response[i]:
            response_memory.append(filter_res)
        for filter_str in filter_strategy[i]:
            strategy_memory.append(filter_str)
    
    # Using dictionary to put together responses that use the same strategy
    memory_dict = {}
    for response, strategy_id in zip(response_memory, strategy_memory):
        # If the strategy ID is already in the dictionary, add the current sentence to the dict corresponding to the strategy ID
        if strategy_id in memory_dict:
            memory_dict[strategy_id].append(response)
        else:
            # If the strategy ID is not in the dictionary, create a new dict and add the current sentence to it
            memory_dict[strategy_id] = [response]
    
    for i in range(8):
        if i not in memory_dict.keys():
            memory_dict[i] = []
    
    memory_dict = OrderedDict(sorted(memory_dict.items()))
    return memory_dict

class StrategyEncoder(BartEncoder):
    """
    Update the memory bank based on the responses and the corresponding strategies in the current batch.
    """
    def __init__(self, config, tokenizer, strategy_memory_max, max_length_res_memory):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.strategy_memory_max = strategy_memory_max
        self.max_length_res_memory = max_length_res_memory
        self.max_pooling = nn.MaxPool1d(kernel_size=self.max_length_res_memory)
        self.init_weights()
    
        
    def forward(self, memory_bank, response_memory, strategy_memory):
        """
        Concatenate all responses with the same strategy_id, and encode them together.
        """
        memory_dict = BuildMemoryDict(response_memory, strategy_memory)
        new_memory_bank = [copy.deepcopy(memory.detach()) for memory in memory_bank]
        
        all_input_ids = []
        all_attention_mask = []
        output_lengths = []
        
        # use for auxiliary loss 
        memory_strategy_ids = []
             
        for strategy_id, response_list in memory_dict.items():
            if len(response_list) == 0:
                output_lengths.append(0)
                continue
            
            for i in range(len(response_list)):
                memory_strategy_ids.append(strategy_id)
            
            inputs = self.tokenizer(
                response_list,
                max_length=self.max_length_res_memory,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            all_input_ids.append(inputs['input_ids'])
            all_attention_mask.append(inputs['attention_mask'])
            output_lengths.append(inputs['input_ids'].size(0))
        
        all_input_ids = torch.cat(all_input_ids, dim=0).to("cuda")
        all_attention_mask = torch.cat(all_attention_mask, dim=0).to("cuda")
        res_encoder_output = super().forward(input_ids=all_input_ids, attention_mask=all_attention_mask).last_hidden_state

        #pooled_encoder_output, _ = torch.max(res_encoder_output, dim=1)
        # max_pooling along sequence length dim
        pooled_encoder_output = self.max_pooling(res_encoder_output.transpose(1, 2)).squeeze(2)

        
        # Split the pooled_encoder_output into multiple tensors according to the strategy_id
        start_idx = 0
        
        for strategy_id, length in zip(memory_dict.keys(), output_lengths):
            if length == 0:
                continue
            
            end_idx = start_idx + length
            # retrieve the pooled representation of the current strategy_id
            #pooled_res, _ = torch.max(res_encoder_output[start_idx:end_idx], dim=1)
            pooled_res = pooled_encoder_output[start_idx:end_idx]
            # update the corresponding memory bank
            new_memory_bank[strategy_id] = torch.cat([new_memory_bank[strategy_id], pooled_res], dim=0)
            if new_memory_bank[strategy_id].size(0) > self.strategy_memory_max:
                new_memory_bank[strategy_id] = new_memory_bank[strategy_id][-self.strategy_memory_max: ]
            # update the start_idx
            start_idx = end_idx


        # return the updated memory bank. pooled_encoder_output and strategy_ids for auxiliary loss calculation
        return new_memory_bank, pooled_encoder_output, memory_strategy_ids

# The auxiliary classification task would be another model that takes the pooled representation and predicts the strategy.
class StrategyClassifier(nn.Module):
    def __init__(self, input_dim, num_strategies):
        super(StrategyClassifier, self).__init__()
        self.classifier = nn.Linear(input_dim, num_strategies)
    
    def forward(self, strategy_pattern_representation):
        return self.classifier(strategy_pattern_representation)


class MemoryEnhanceModule(nn.Module):
    def __init__(self, d_model, num_heads, max_length, device):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads)
        self.max_pooling = nn.MaxPool1d(kernel_size=max_length)
        self.device = device
    
    def forward(self, strategy_ids, memory_bank, encoder_output, motivation_output):
        motivation_output = motivation_output
        # retrieve the memory bank corresponding to the strategy_id
        selected_memories = [memory_bank[strategy_id] for strategy_id in strategy_ids]

        max_memory_length = max([tensor.size(0) for tensor in selected_memories])
        
        padded_memories = []
        attention_masks = []
        for tensor in selected_memories:
            padding_length = max_memory_length - tensor.size(0)
            padding = torch.zeros(padding_length, tensor.size(1)).to("cuda")
            padded_tensor = torch.cat([tensor, padding], dim=0)
            padded_memories.append(padded_tensor)
            
            attention_mask = torch.ones(tensor.size(0), dtype=torch.bool)
            attention_mask = F.pad(attention_mask, (0, padding_length), value=False)
            attention_masks.append(attention_mask)
        
        padded_memories = torch.stack(padded_memories, dim=0).permute(1, 0, 2).to(self.device)  # max_length x batch_size x hidden_size
        attention_masks = torch.stack(attention_masks, dim=0).to(self.device)  # batch_size x sequence_length
    

        attn_output, _ = self.multihead_attn(query=encoder_output.permute(1, 0, 2), key=padded_memories, value=padded_memories, key_padding_mask=~attention_masks)
        attn_output = attn_output.permute(1, 0, 2)

    
        #pooled_output, _ = torch.max(attn_output, dim=1)
        pooled_output = self.max_pooling(attn_output.transpose(1, 2)).squeeze(2)
        pooled_output = pooled_output.unsqueeze(1)
    
        return pooled_output


class DialogueEncoder(BartEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.init_weights()
        
    def forward(self, input_ids, attention_mask, motivation_ids, motivation_mask):
        encoder_output = super().forward(input_ids=input_ids, attention_mask=attention_mask)      
        motivation_output  = super().forward(input_ids=motivation_ids, attention_mask=motivation_mask)  
        return encoder_output.last_hidden_state, motivation_output.last_hidden_state

class DialogueDecoder(BartDecoder):
    def __init__(self, config):
        super().__init__(config)
        self.init_weights()
        
    def forward(self, decoder_input_ids, encoder_hidden_states, encoder_attention_mask, decoder_attention_mask, memory_enhanced_context):

        cat_encoder_hidden_states = F.pad(encoder_hidden_states, (0, 0, 0, 1, 0, 0), value=0)
        cat_encoder_hidden_states[:, -1, :] = memory_enhanced_context[:, 0, :]

        cat_encoder_attention_mask = torch.cat((encoder_attention_mask, torch.ones(encoder_attention_mask.size(0), 1).to(encoder_attention_mask.device)), dim=1)
    
        decoder_output = super().forward(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=cat_encoder_hidden_states,
            encoder_attention_mask=cat_encoder_attention_mask
        )
        return decoder_output.last_hidden_state

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.proj(x)
    
class DialogueModel(nn.Module):
    def __init__(self, config, bart_config, tokenizer, max_length, strategy_memory_max, max_length_res_memory, device):
        super().__init__()
        self.encoder = DialogueEncoder(bart_config)
        self.decoder = DialogueDecoder(bart_config)
        self.strategy_encoder = StrategyEncoder(bart_config, tokenizer, strategy_memory_max, max_length_res_memory)
        self.memory_enhance_module = MemoryEnhanceModule(config.d_model, config.encoder_attention_heads, max_length, device)
        self.generator = Generator(config.d_model, config.vocab_size)
        self.strategy_cls = StrategyClassifier(config.d_model, 8)
        
        #initial memory bank with empty, 8 strategy category
        self.memory_bank =[torch.empty((0, config.d_model)).to(device) for _ in range(8)]
      
        
    def forward(self, input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask, strategy_ids, response_memory, strategy_memory, motivation_ids, motivation_mask, stage='train'):
        if stage == 'train':
            self.memory_bank, pooled_encoder_output, memory_strategy_ids = self.strategy_encoder(self.memory_bank, response_memory, strategy_memory)
            
            # forward auxiliary classification task
            strategy_logits = self.strategy_cls(pooled_encoder_output.to(input_ids.device))
        
        encoder_output, motivation_output = self.encoder(input_ids, encoder_attention_mask, motivation_ids, motivation_mask)
        memory_enhanced_context = self.memory_enhance_module(strategy_ids, self.memory_bank, encoder_output, motivation_output)
       
        decoder_output = self.decoder(decoder_input_ids, encoder_output, encoder_attention_mask, decoder_attention_mask, memory_enhanced_context)
        logits = self.generator(decoder_output)
        
        if stage == 'train':
            return logits, strategy_logits, memory_strategy_ids
        else:
            return logits


