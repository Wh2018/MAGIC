# Collecting motivations of strategy inference from ChatGPT API

import json
from tqdm import tqdm
import argparse
from openai import OpenAI
from random import choice
import os
from time import sleep

def _norm(x):
    return ' '.join(x.strip().split())


def gen_prompt_bein_end():
    """
    Generate the begin and the end par of prompts for ChatGPT
    """
    
    prompt_begin = """Given a recent emotional support dialogue with labeled support strategies, infer the supporter's motivation for choosing the last strategy in the last response. There are 8 kinds of support strategies, including [Question], [Restatement or Paraphrasing], [Reflection of Feelings], [Self-disclosure], [Affirmation and Reassurance], [Providing Suggestions or Information], [Greeting], and [Others]
    
The following is an example of a dialogue clip and corresponding motivation:
(1) Seeker: Hi can you help me with my problem?
(2) Supporter: Hello. What is on your mind? [Question]
(3) Seeker: I am disgusted with my friend for cheating on her boyfriend. Am I right to feel this way?
(4) Supporter: Your feelings are valid, no matter what they are. Why do you think you are disgusted with her? [Affirmation and Reassurance]

What is the motivation of the supporter to choose the last strategy [Affirmation and Reassurance]? Please provide a succinct motivation (max 40 words) based on the dialogue history before the last response like this:
Motivation: The motivation of the supporter in choosing the strategy [Affirmation and Reassurance] is to validate the seeker's feelings and provide reassurance, aiming to build trust and a supportive relationship.

Now, generate a succinct motivation (max 40 words) based on the dialogue history for the supporter's choice of the last strategy. The dialogue clip is:
"""

    prompt_end = """What is the motivation of the supporter to choose the last strategy in his last utterance?
Motivation:
    """
    
    return prompt_begin, prompt_end
    
    
def read_traindata_4_prompt_medium(train_file_path):
    """
    Read training data of ESConv to generate the medium part of prompts for ChatGPT
    """
    
    with open(train_file_path, 'r') as f:
        datas = f.readlines()
    
    prompts_medium = []
    
    for data in tqdm(datas):
        data = json.loads(data)
        dialogue = data['dialog']
        # One set of multi-turn dialogue can generate many prompts with different turns, [0,1], [0,1,2,3], [0,1,2,3,4,5,6],...
        
        # usr (seeker) begin first
        if dialogue[0]['speaker'] == 'usr':
            # Even-numbered turns of dialogue, begin with usr, end with sys (supporter)
            if len(dialogue) % 2 == 0:
                dia_len = len(dialogue)
            # Odd-numbered turns of dialogue, begin with usr, end with usr
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
                            # The final turn of the supporter
                            if j == i+1:
                                prompt_medium += """(""" + str(j+1) + """) Supporter: """ + _norm(dialogue[j]['text']) + """ ***[""" + dialogue[j]['strategy'] + """]***
"""
                            else:
                                prompt_medium += """(""" + str(j+1) + """) Supporter: """ + _norm(dialogue[j]['text']) + """ [""" + dialogue[j]['strategy'] + """]
"""
                
                else:
                    prompt_medium = """(1) Seeker: """ + _norm(dialogue[i]['text']) + """
(2) Supporter: """ + _norm(dialogue[i+1]['text']) + """ ***[""" + dialogue[i+1]['strategy'] + """]***
"""
                
                prompts_medium.append(prompt_medium)
        
        # sys (supporter) begin first
        else:
            # Even-numbered turns of dialogue, begin with sys, end with usr
            if len(dialogue) % 2 == 0:
                dia_len = len(dialogue) - 1
            # Odd-numbered turns of dialogue, begin with sys, end with sys
            else:
                dia_len = len(dialogue)
            
            for i in range(1, dia_len, 2):
                if i > 1:
                    prompt_medium = ''
                    for j in range(0, i+2):
                        if j % 2 == 0:
                            # The final turn of the supporter
                            if j == i+1:
                                prompt_medium += """(""" + str(j+1) + """) Supporter: """ + _norm(dialogue[j]['text']) + """ ***[""" + dialogue[j]['strategy'] + """]***
"""
                            else:
                                prompt_medium += """(""" + str(j+1) + """) Supporter: """ + _norm(dialogue[j]['text']) + """ [""" + dialogue[j]['strategy'] + """]
"""
                        else:
                            prompt_medium += """(""" + str(j+1) + """) Seeker: """ + _norm(dialogue[j]['text']) + """
"""
            
                else:
                    prompt_medium = """(1) Supporter: """ + _norm(dialogue[0]['text']) + """ ***[""" + dialogue[0]['strategy'] + """]***
""" + """(2) Seeker: """ + _norm(dialogue[1]['text']) + """
""" + """(3) Supporter: """ + _norm(dialogue[2]['text']) + """ ***[""" + dialogue[2]['strategy'] + """]***
"""
            
                prompts_medium.append(prompt_medium)
    
    return prompts_medium


def create_prompts_4_chatgpt(prompts_medium):
    prompt_bein, prompt_end = gen_prompt_bein_end()
    prompts_4_chatgpt = []
    for prompt_medium in prompts_medium:
        prompt = prompt_bein + prompt_medium + prompt_end
        prompts_4_chatgpt.append(prompt)
    
    # json_data = json.dumps(prompts_4_chatgpt, ensure_ascii=False, indent=2)

    # with open('prompts_4_chatgpt.json', 'w') as json_file:
    #     json_file.write(json_data)
    
    return prompts_4_chatgpt
    
def collect_motivations_by_chatgpt_API(prompts_medium, openai_client):
    """
    This function collects motivations from chatgpt API using the designed prompts.
    """
    
    prompts_4_chatgpt = create_prompts_4_chatgpt(prompts_medium)
    Motivations = []

    for prompt in tqdm(prompts_4_chatgpt):
        try:
            completion = openai_client.chat.completions.create(
            model = "gpt-3.5-turbo-0125",
            messages= [
                #{"role": "system", "content": "You will be provided with context, requirement and format. Generate a dialogue as required."},
                {"role": "user", "content": prompt}
            ]
            )

            motivation = completion.choices[0].message.content
            print(motivation)
            Motivations.append(motivation)
            
            # sleep for 3 seconds to avoid rate limit
            sleep(3)
            
        except Exception as e:
            print(e)
            print(prompt)
    
    json_data = json.dumps(Motivations, ensure_ascii=False, indent=2)

    with open('motivations_4_LLaMA.json', 'w') as json_file:
        json_file.write(json_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file_path', type=str, default='../data/train.txt')
    # Set up your own OpenAI API kEY!!!
    parser.add_argument('--openai_api_key', type=str, required=True)
    args = parser.parse_args()
    
    
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=args.openai_api_key,
    )
        
    prompts_medium = read_traindata_4_prompt_medium(args.train_file_path)
    collect_motivations_by_chatgpt_API(prompts_medium, client)
