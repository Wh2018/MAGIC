# Create prompts with motivations in the self-instruct format for fine-tune llama2
# {"prompt:xxx, completion:xxx"}

import json
from tqdm import tqdm
def _norm(x):
    return ' '.join(x.strip().split())

def gen_prompt_bein_end():
    prompt_begin = """1) Task definition and instruction:
You are an expert in the theory of dialogue contextual understanding and strategy inference. Given a recent emotional support dialogue with labeled support strategies, you need to predict an appropriate support strategy for the last utterance of the supporter with motivations, to reach the long-term emotional support goal. There are 8 kinds of support strategies, including [Question], [Restatement or Paraphrasing], [Reflection of Feelings], [Self-disclosure], [Affirmation and Reassurance], [Providing Suggestions or Information], [Greeting], and [Others].

2) Example and Answers:
The following is a dialogue example and corresponding strategy inference with motivation:
(1) Seeker: Hi can you help me with my problem?
(2) Supporter: Hello. What is on your mind? [Question]
(3) Seeker: I am disgusted with my friend for cheating on her boyfriend. Am I right to feel this way?

What is the last appropriate support strategy to be used by the supporter? Please make inferences with a succinct motivation (max 40 words) based on the dialogue history like this:
Motivation: The motivation of the supporter in choosing the strategy [Affirmation and Reassurance] is to validate the seeker's feelings and provide reassurance, aiming to build trust and a supportive relationship.

3) Dialogue context to be inferred:
Now, infer the supporter's choice of the last strategy with a succinct motivation (max 40 words). The dialogue clip is:
"""

    prompt_end = """
Motivation:
"""

    return prompt_begin, prompt_end


def read_traindata_4_prompt_medium(file_path):
    with open(file_path, 'r') as f:
        datas = f.readlines()
    
    prompts_medium = []
    
    for data in tqdm(datas):
        data = json.loads(data)
        dialogue = data['dialog']
        
        if dialogue[0]['speaker'] == 'usr':
            if len(dialogue) % 2 == 0:
                dia_len = len(dialogue)
            else:
                dia_len = len(dialogue) - 1 
                
            for i in range(0, dia_len, 2):
                if i > 0:
                    prompt_medium = ''
                    for j in range(0, i+2):
                        if j % 2 == 0:
                            prompt_medium += """(""" + str(j+1) + """) Seeker: """ + _norm(dialogue[j]['text']) + """
"""
                        else:
                            if j == i+1:
                                continue
                                prompt_medium += """(""" + str(j+1) + """) Supporter: """ + _norm(dialogue[j]['text']) + """ [""" + dialogue[j]['strategy'] + """]
"""
                            else:
                                prompt_medium += """(""" + str(j+1) + """) Supporter: """ + _norm(dialogue[j]['text']) + """ [""" + dialogue[j]['strategy'] + """]
"""

                else:
                    prompt_medium = """(1) Seeker: """ + _norm(dialogue[i]['text'])
                
                prompts_medium.append(prompt_medium)
        
        else:
            if len(dialogue) % 2 == 0:
                dia_len = len(dialogue) - 1
            else:
                dia_len = len(dialogue)
            
            for i in range(1, dia_len, 2):
                if i > 1:
                    prompt_medium = ''
                    for j in range(0, i+2):
                        if j % 2 == 0:
                            if j == i+1:
                                continue
                                prompt_medium += """(""" + str(j+1) + """) Supporter: """ + _norm(dialogue[j]['text']) + """ [""" + dialogue[j]['strategy'] + """]
"""
                            else:
                                prompt_medium += """(""" + str(j+1) + """) Supporter: """ + _norm(dialogue[j]['text']) + """ [""" + dialogue[j]['strategy'] + """]
"""
                        else:
                            prompt_medium += """(""" + str(j+1) + """) Seeker: """ + _norm(dialogue[j]['text']) + """
"""

                else:
                    prompt_medium = """(1) Supporter: """ + _norm(dialogue[0]['text']) + """ [""" + dialogue[0]['strategy'] + """]
""" + """(2) Seeker: """ + _norm(dialogue[1]['text'])
            
                prompts_medium.append(prompt_medium)
        
    return prompts_medium

def read_testdata_4_prompt_medium(file_path):
    with open(file_path, 'r') as f:
        datas = f.readlines()
    
    prompts_medium = []
    
    for data in tqdm(datas):
        data = json.loads(data)
        dialogue = data['dialog']
        
        if dialogue[-1]['speaker'] == 'usr':
            prompt_medium = ''
            for i in range(0, len(dialogue)-2):
                if dialogue[i]['speaker'] == 'usr':
                    prompt_medium += """(""" + str(i+1) + """) Seeker: """ + _norm(dialogue[i]['text']) + """
"""
                else:
                    prompt_medium += """(""" + str(i+1) + """) Supporter: """ + _norm(dialogue[i]['text']) + """ [""" + dialogue[i]['strategy'] + """]
""" 
            prompts_medium.append(prompt_medium)
        
        else:
            prompt_medium = ''
            for i in range(0, len(dialogue)-1):
                if dialogue[i]['speaker'] == 'usr':
                    prompt_medium += """(""" + str(i+1) + """) Seeker: """ + _norm(dialogue[i]['text']) + """
"""
                else:
                    prompt_medium += """(""" + str(i+1) + """) Supporter: """ + _norm(dialogue[i]['text']) + """ [""" + dialogue[i]['strategy'] + """]
"""
            prompts_medium.append(prompt_medium)
            
    return prompts_medium

def create_prompts_4_LLaMA_train(prompts_medium_train):
    """
    Create prompts for fine-tune llama2 with motivations in the self-instruct format with training data
    """
    # load generated motivations generated by chatgpt    
    with open('motivations_4_LLaMA.json') as f:
        motivations = json.load(f)
    
    prompt_begin, prompt_end = gen_prompt_bein_end()
    prompts_4_LLaMA = []
    
    for i in range(len(motivations)):
        prompt = {}
        instruction = prompt_begin + prompts_medium_train[i] + prompt_end
        output = motivations[i]
        
        prompt['instruction'] = instruction
        prompt['output'] = output
        prompts_4_LLaMA.append(prompt)
    
    json_data = json.dumps(prompts_4_LLaMA, ensure_ascii=False, indent=2, default=str)
    
    with open('prompts_4_LLaMA.json', 'w') as json_file:
        json_file.write(json_data)
    
        

def create_prompts_4_LLaMA_test(prompts_medium_test, idx=0):
    """
    create prompts for inferring motivations using fined-tuned llama2 with test data
    """
    prompt_begin, prompt_end = gen_prompt_bein_end()

    prompts_4_LLaMA = []

    for i in range(len(prompts_medium_test)):
        prompt = prompt_begin + prompts_medium_test[i] + prompt_end
        prompts_4_LLaMA.append(prompt)
    
    json_data = json.dumps(prompts_4_LLaMA, ensure_ascii=False, indent=2, default=str)

    if idx == 0:
        with open('prompts_4_LLaMA_valid.json', 'w') as json_file:
            json_file.write(json_data)
    else:
        with open('prompts_4_LLaMA_test.json', 'w') as json_file:
            json_file.write(json_data)
    

if __name__ == '__main__':
    prompts_medium_train = read_traindata_4_prompt_medium('../data/train.txt')
    prompts_medium_valid = read_traindata_4_prompt_medium('../data/valid.txt')
    prompts_medium_test = read_testdata_4_prompt_medium('../data/test.txt')
    
    create_prompts_4_LLaMA_train(prompts_medium_train)
    create_prompts_4_LLaMA_test(prompts_medium_valid, 0)
    create_prompts_4_LLaMA_test(prompts_medium_test, 1)
    
    