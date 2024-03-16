"""
Convert ESConv data in text format to json format

read_data_train_val: read train and validation data from ESCONV, 
read_data_test: read test data from ESCONV

The output is a list of conversations, each conversation is a dictionary with keys 'context', 'response', 'strategy','response_memory','strategy_memory', 'motivation'.
"""

import json
import re
from tqdm import tqdm
import os
import random

abbreviations_lables = {
    "You ' re": "You are",
    "I ' m": "I am",
    "He ' s": "He is",
    "She ' s": "She is",
    "It ' s": "It is",
    "We ' re": "We are",
    "They ' re": "They are",
    "Can ' t": "Cannot",
    "Don ' t": "Do not",
    "Doesn ' t": "Does not",
    "Didn ' t": "Did not",
    "Won ' t": "Will not",
    "Isn ' t": "Is not",
    "Aren ' t": "Are not",
    "Wasn ' t": "Was not",
    "Weren ' t": "Were not",
    "Haven ' t": "Have not",
    "Hasn ' t": "Has not",
    "Hadn ' t": "Had not",
    "Let ' s": "Let us",
    "That ' s": "That is",
    "Who ' s": "Who is",
    "What ' s": "What is",
    "Where ' s": "Where is",
    "When ' s": "When is",
    "Why ' s": "Why is",
    "How ' s": "How is",
    "I ' ll": "I will",
    "You ' ll": "You will",
    "He ' ll": "He will",
    "She ' ll": "She will",
    "It ' ll": "It will",
    "We ' ll": "We will",
    "They ' ll": "They will",
    "I ' ve": "I have",
    "You ' ve": "You have",
    "We ' ve": "We have",
    "They ' ve": "They have",
    "I ' d": "I would",
    "You ' d": "You would",
    "He ' d": "He would",
    "She ' d": "She would",
    "It ' d": "It would",
    "We ' d": "We would",
    "They ' d": "They would",
}


def read_data_train_val(file_path):
    """
    Read train and validation data from ESCONV

    Returns:
        Conversations: []
        {
            context: user situation and dialogue context with n-1 turns
            response: n-th response, from sys
            strategy: n-th strategy, from sys
            response_memory: (n-1) / 2 response from sys
            strategy_memory: (n-1) / 2 strategy from sys of response
        }
    """
    conversations = []

    with open(file_path, 'r') as file:
        for line in file:
            json_data = json.loads(line)
            len_dia = len(json_data['dialog'])
            situation = json_data['situation']
            dialog = json_data['dialog']
            
            if len_dia % 2 == 0:
                if dialog[0]['speaker'] == 'usr':
                    for i in range(0, len_dia, 2):
                        context = dialog[:i+1]
                        context = [item['text'] for item in context]
                        context.insert(0, situation)
                        response = dialog[i+1]['text']
                        strategy = dialog[i+1]['strategy']
                        
                        #add response with strategy for strategy memory
                        if i != 0:
                            memory_slice = dialog[:i+1][1::2]
                            response_memory = [item['text'] for item in memory_slice]
                            strategy_memory = [item['strategy'] for item in memory_slice]
                                                  
                        conversations.append({'context': context, 'response': response, 'strategy':strategy, 'response_memory':response_memory if i!=0 else [], 'strategy_memory':strategy_memory if i!=0 else []})

                if dialog[0]['speaker'] == 'sys':
                    for i in range(1, len_dia-1, 2):
                        context = dialog[:i+1]
                        context = [item['text'] for item in context]
                        context.insert(0, situation)
                        response = dialog[i+1]['text']
                        strategy = dialog[i+1]['strategy']
                        
                        # add response with strategy for strategy memory
                        memory_slice = dialog[:i+1][0::2]
                        response_memory = [item['text'] for item in memory_slice]
                        strategy_memory = [item['strategy'] for item in memory_slice] 
                        
                        conversations.append({'context': context, 'response': response, 'strategy':strategy, 'response_memory':response_memory if i!=0 else [], 'strategy_memory':strategy_memory if i!=0 else []})
                
            else:
                if dialog[0]['speaker'] == 'usr':
                    for i in range(0, len_dia-1, 2):
                        context = dialog[:i+1]
                        context = [item['text'] for item in context]
                        context.insert(0, situation)
                        response = dialog[i+1]['text']
                        strategy = dialog[i+1]['strategy']
                        
                        # add response with strategy for strategy memory
                        if i != 0:
                            memory_slice = dialog[:i+1][1::2]
                            response_memory = [item['text'] for item in memory_slice]
                            strategy_memory = [item['strategy'] for item in memory_slice] 
                        
                        conversations.append({'context': context, 'response': response, 'strategy':strategy, 'response_memory':response_memory if i!=0 else [], 'strategy_memory':strategy_memory if i!=0 else []})
                        
                if dialog[0]['speaker'] == 'sys':
                    for i in range(1, len_dia, 2):
                        context = dialog[:i+1]
                        context = [item['text'] for item in context]
                        context.insert(0, situation)
                        response = dialog[i+1]['text']
                        strategy = dialog[i+1]['strategy']
                        
                        # add response with strategy for strategy memory
                        memory_slice = dialog[:i+1][0::2]
                        response_memory = [item['text'] for item in memory_slice]
                        strategy_memory = [item['strategy'] for item in memory_slice] 
                        
                        conversations.append({'context': context, 'response': response, 'strategy':strategy, 'response_memory':response_memory if i!=0 else [], 'strategy_memory':strategy_memory if i!=0 else []})

    return conversations

def read_data_test(file_path):
    """
    Read test data from ESCONV

    Returns:
        Conversations: []
        {
            context: user situation and dialogue context with n-1 turns
            response: n-th response, from sys
            strategy: n-th strategy, from sys
            response_memory: (n-1) / 2 response from sys
            strategy_memory: (n-1) / 2 strategy from sys of response
        }
    """
    conversations = []

    with open(file_path, 'r') as file:
        for line in file:
            json_data = json.loads(line)
            len_dia = len(json_data['dialog'])
            situation = json_data['situation']
            dialog = json_data['dialog']
            
            if dialog[-1]['speaker'] == 'usr':
                context = dialog[:-2]
                context = [item['text'] for item in context]
                context.insert(0, situation)
                response = dialog[-2]['text']
                strategy = dialog[-2]['strategy']
                
                response_memory = []
                strategy_memory = []
                
                conversations.append({'context': context, 'response': response, 'strategy':strategy, 'response_memory':response_memory, 'strategy_memory':strategy_memory})
            else:
                context = dialog[:-1]
                context = [item['text'] for item in context]
                context.insert(0, situation)
                response = dialog[-1]['text']
                strategy = dialog[-1]['strategy']
                
                response_memory = []
                strategy_memory = []
                
                conversations.append({'context': context, 'response': response, 'strategy':strategy, 'response_memory':response_memory, 'strategy_memory':strategy_memory})

    return conversations
    

def update_strategy_motivation_2_json(train_conversations, val_conversations, test_conversations):
    """
    Update strategies and motivations inferred by from fine-tuned LLaMA2 in train, validation, and test data.
    """
    
    train_conversations = train_conversations
    val_conversations = val_conversations
    test_conversations = test_conversations
    
    #Reducing abbreviations to original words in original data and save data to json files.
    #e.g., "You ' re" -> "You are"    
    for conversations in [train_conversations, val_conversations, test_conversations]:
        for conversation in tqdm(conversations):
            context = conversation['context']
            response = conversation['response']
            response_memory = conversation['response_memory']
            strategy_memory = conversation['strategy_memory']
            
            new_context = []
            for turn in context:
                turn = turn.strip()

                for abbreviation, full_form in abbreviations_lables.items():
                    turn = re.sub(r'\b' + re.escape(abbreviation) + r'\b', full_form, turn, flags=re.IGNORECASE)
                new_context.append(turn)
            conversation['context'] = new_context
            
            for abbreviation, full_form in abbreviations_lables.items():
                response = re.sub(r'\b' + re.escape(abbreviation) + r'\b', full_form, response, flags=re.IGNORECASE)
            conversation['response'] = response
            
            new_response_memory = []
            for turn in response_memory:
                for abbreviation, full_form in abbreviations_lables.items():
                    turn = re.sub(r'\b' + re.escape(abbreviation) + r'\b', full_form, turn, flags=re.IGNORECASE)
                new_response_memory.append(turn)
            conversation['response_memory'] = new_response_memory
            
            new_strategy_memory = []
            for turn in strategy_memory:
                for abbreviation, full_form in abbreviations_lables.items():
                    turn = re.sub(r'\b' + re.escape(abbreviation) + r'\b', full_form, turn, flags=re.IGNORECASE)
                new_strategy_memory.append(turn)
            conversation['strategy_memory'] = new_strategy_memory   

    # read strategies and motivations inferred by fine-tuned LLaMA2, in json file
    if os.path.exists('LLM_inference/strategy_with_motivation.json') and os.path.exists('LLM_inference/strategy_with_motivation_valid.json') and os.path.exists('LLM_inference/strategy_with_motivation_test.json'):
        with open('LLM_inference/strategy_with_motivation.json') as f:
            strategy_motivation_train = json.load(f)
        with open('LLM_inference/strategy_with_motivation_valid.json') as f:
            strategy_motivation_valid = json.load(f)
        with open('LLM_inference/strategy_with_motivation_test.json') as f:
            strategy_motivation_test = json.load(f)
        
        idx = 0
        for conversations in [train_conversations, val_conversations, test_conversations]:
            for i in range(len(conversations)):
                conversation = conversations[i]
                if idx == 0:
                    strategy = strategy_motivation_train[i]['strategy']
                    motivation = strategy_motivation_train[i]['motivation']
                elif idx == 1:
                    strategy = strategy_motivation_valid[i]['strategy']
                    motivation = strategy_motivation_valid[i]['motivation']
                else:
                    strategy = strategy_motivation_test[i]['strategy']
                    motivation = strategy_motivation_test[i]['motivation']
                
                conversation.update({'strategy': strategy})   
                conversation.update({'motivation': motivation})
            
            idx = idx + 1
    
    else:
        for conversations in [train_conversations, val_conversations, test_conversations]:
            for conversation in conversations:
                conversation.update({'motivation': ''})


    # save data to json files
    with open('data/train.json', 'w') as f:
        json.dump(train_conversations, f, indent=4, ensure_ascii=False)

    with open('data/valid.json', 'w') as f:
        json.dump(val_conversations, f, indent=4, ensure_ascii=False)

    with open('data/test.json', 'w') as f:
        json.dump(test_conversations, f, indent=4, ensure_ascii=False)



if __name__ == '__main__':
    train_conversations = read_data_train_val('data/train.txt')
    val_conversations = read_data_train_val('data/valid.txt')
    test_conversations = read_data_test('data/test.txt')
    
    update_strategy_motivation_2_json(train_conversations, val_conversations, test_conversations)