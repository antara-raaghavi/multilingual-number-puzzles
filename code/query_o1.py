from datasets import load_dataset
from mathify import *
from mathify_new import *
from tqdm import tqdm
import load_questions
import os
from openai import OpenAI
import csv
import pandas as pd
import json
import tiktoken
import random
from datetime import datetime
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import string
import requests


#-------------------------------------------------------------------------------------------------
#                                MODEL ACCESS + QUERYING
#-------------------------------------------------------------------------------------------------

#Set HuggingFace token for gated access.
os.environ["HF_TOKEN"] = ''

#OpenAI API key.
my_key = 'sk-proj--'

HUIT_API_KEY = ''

client = OpenAI(api_key=my_key)


# If using Harvard HUIT-granted API key -- with a personal key, just use query_gpt().

def huit_query_gpt(prompt, variant):
    url = "https://go.apis.huit.harvard.edu/ais-openai-direct-limited-schools/v1/chat/completions"

    payload = {
        "model": "o1-mini-2024-09-12",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 1 if variant == "o1-mini" else 0  
    }
    headers = {
        "Content-Type": "application/json",
        "api-key": HUIT_API_KEY
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"] 
        return content
    
    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        return None
    



def query_gpt(prompt, variant):
    ''' NOTE: temperature is not manipulable for o1-mini. 
        For all other GPT models, we use temperature = 0 (see commented line.)'''
    try:
        
        chat_completion = client.chat.completions.create(messages=[{"role": "user", "content": prompt}],
                                                         model=variant
                                                        #temperature= 1 if variant == "o1-mini" else 0 
                                                         )
        return chat_completion.choices[0].message.content.strip()
    except Exception as exception:
        print(str(exception))








#-------------------------------------------------------------------------------------------------
#                                   MINIMAL PAIR PROBLEMS
#-------------------------------------------------------------------------------------------------

# First-pass, one-off prompts. 


is_lang = """AB = big bird

AC = big fish

DB = small bird

DC = ??

"""

is_num = """AB = fifty-one

AC = fifty-seven 

DB = forty-one

DC = ?? 

"""


num_decimal = """AB = 51

AC = 57

DB = 41

DC = ?? 

"""
num_vigesimal = """AB = 31

AC = 24

DB = 51

DC = ??

"""

ordering_LR = """AB = 51

AC = 57

DB = 41

DC = ??

"""

ordering_RL = """BA = 51

CA = 57

BD = 41

CD= ??

"""

additive = """AB = 27

DC = 33

DB = 37

AC = ??

"""

subtractive = """AB = 27 

AC = 33

DB = 37

DC = ??

"""

    
paradigms = [is_lang, is_num, 
           num_decimal, num_vigesimal, 
           ordering_LR, ordering_RL, 
           additive, subtractive]


#-------------------------------------------------------------------------------------------------
#                               RANDOM TOKENIZATION FOR TEMPLATE LHS
#-------------------------------------------------------------------------------------------------



def get_random_tokens(n_tokens = 4, tokenizer_model = "gpt-4o"):
    # cf. https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

    # load the tokenizer GPT models use.
    tokenizer = tiktoken.encoding_for_model(tokenizer_model)


    # seed the random generator (if you want to get the same 4 numbers every time).
    # random.seed(4242)

    i = 0
    tokens = []
    while i < n_tokens:
        # generate a random token. 
        max_tok = 100000
        random_token_id = random.randint(0, max_tok)
        tok = tokenizer.decode([random_token_id])

        #the token can't itself be a number, and for ease of suffering no non-eng scripts.
        if not tok.isnumeric() and re.match(r'^[a-zA-Z]+$', tok):
            tokens.append(tok)
            i += 1 

    return tokens


def get_random_tokens_sampled(n_tokens = 4, single_toks = False, tokenizer_model = "gpt-3.5-turbo", max_tok_len = 3, ):

    if single_toks:
        lowercase_letters = list(string.ascii_lowercase)
        return random.sample(lowercase_letters, n_tokens)

    # seed the random generator (if you want to get the same 4 numbers every time).
    # random.seed(4242)

    
    # load the tokenizer GPT models use.
    tokenizer = tiktoken.encoding_for_model(tokenizer_model)

    decoded = []
    for i in range(10000):
        random_token_id = random.randint(0, 100000)
        tok = tokenizer.decode([random_token_id])
        decoded.append(tok)
        i+=1


    # regex only for alphabetic and smallish length
    valid_toks = [tok.lower() for tok in decoded if len(tok) <= max_tok_len
                                            and re.match(r'^[a-zA-Z]+$', tok)
                                            and re.search(r'[aeiouAEIOU]', tok)]

    # pick some small tokens and concatenate them, n_tokens times
    tokens = []
    for _ in range(n_tokens):
        random.shuffle(valid_toks) #in case rng seeded.
        pick_2 = random.sample(valid_toks, 2)  
        concat = "".join(pick_2) 
        tokens.append(concat)
        valid_toks = [item for item in valid_toks if item not in pick_2]


    return tokens






#-------------------------------------------------------------------------------------------------
#                               TEMPLATES FOR ALL BASES 4 - 20
#-------------------------------------------------------------------------------------------------




def all_bases_test(random_flag):

    base_4, base_5, base_6, base_7, base_8, base_9 = [], [], [], [], [], []
    base_11, base_12, base_13, base_14, base_15, base_16, base_17, base_18, base_19 = [], [], [], [], [], [], [], [], []
    baseprompts = [base_4, base_5, base_6, base_7, base_8, base_9, 
                   base_11, base_12, base_13, base_14, base_15, base_16, base_17, base_18, base_19]


    # BASES < 10

    base_4_dict = {'ab': ('7', '1x4+3'), 'ac': ('5', '1x4+1'), 'db': ('11', '2x4+3'), 'dc' : ('9', '2x4+1')}
    base_5_dict = {'ab': ('9', '1x5+4'), 'ac': ('8', '1x5+3'), 'db': ('14', '2x5+4'), 'dc' : ('13', '2x5+3')}
    base_6_dict = {'ab': ('11', '1x6+5'), 'ac': ('9', '1x6+3'), 'db': ('17', '2x6+5'), 'dc' : ('15', '2x6+3')}
    base_7_dict = {'ab': ('13', '1x7+6'), 'ac': ('11', '1x7+4'), 'db': ('20', '2x7+6'), 'dc' : ('18', '2x7+4')}
    base_8_dict = {'ab': ('15', '1x8+7'), 'ac': ('13', '1x8+5'), 'db': ('23', '2x8+7'), 'dc' : ('21', '2x8+5')}
    base_9_dict = {'ab': ('17', '1x9+8'), 'ac': ('15', '1x9+6'), 'db': ('26', '2x9+8'), 'dc' : ('24', '2x9+6')}

    # BASES >10

    base_11_dict = {'ab': ('20', '1x11+9'), 'ac': ('13', '1x11+2'), 'db': ('31', '2x11+9'), 'dc' : ('24', '2x11+2')}
    base_12_dict = {'ab': ('21', '1x12+9'), 'ac': ('16', '1x12+4'), 'db': ('33', '2x12+9'), 'dc' : ('28', '2x12+4')}
    base_13_dict = {'ab': ('22', '1x13+9'), 'ac': ('18', '1x13+5'), 'db': ('35', '2x13+9'), 'dc' : ('31', '2x13+5')}
    base_14_dict = {'ab': ('23', '1x14+9'), 'ac': ('17', '1x14+3'), 'db': ('37', '2x14+9'), 'dc' : ('31', '2x14+3')}
    base_15_dict = {'ab': ('19', '1x15+4'), 'ac': ('23', '1x15+8'), 'db': ('34', '2x15+4'), 'dc' : ('38', '2x15+8')}
    base_16_dict = {'ab': ('19', '1x16+3'), 'ac': ('25', '1x16+9'), 'db': ('35', '2x16+3'), 'dc' : ('41', '2x16+9')}
    base_17_dict = {'ab': ('25', '1x17+8'), 'ac': ('22', '1x17+5'), 'db': ('42', '2x17+8'), 'dc' : ('39', '2x17+5')}
    base_18_dict = {'ab': ('20', '1x18+2'), 'ac': ('27', '1x18+9'), 'db': ('38', '2x18+2'), 'dc' : ('45', '2x18+9')}
    base_19_dict = {'ab': ('26', '1x19+7'), 'ac': ('22', '1x19+3'), 'db': ('45', '2x19+7'), 'dc' : ('41', '2x19+3')}



    basepairs = [base_4_dict, base_5_dict, base_6_dict, base_7_dict, base_8_dict, base_9_dict, 
                 base_11_dict, base_12_dict, base_13_dict, base_14_dict, base_15_dict, base_16_dict, base_17_dict, base_18_dict, base_19_dict]

    
    

    for i, numdict in enumerate(basepairs):

        #get random tokens for LHS (new random tokens each time). 
        #if you want random tokens from tokenizer use this 

        if random_flag:
            tokens = get_random_tokens(n_tokens=4, tokenizer_model="gpt-4")

            print(tokens, "\n")

            a, b, c, d = tokens[0], tokens[1], tokens[2], tokens[3]

            ab, ac, db = numdict['ab'][0], numdict['ac'][0], numdict['db'][0]
            temp = f"""\n {a} {b} = {ab} \n {a} {c} = {ac} \n {d} {b} = {db} \n {d} {c} = ?? \n """
            
            baseprompts[i].append(temp)

        else:             
            # just format the problems and add to template list.
            ab, ac, db = numdict['ab'][0], numdict['ac'][0], numdict['db'][0]
            temp = f"""
                A B = {ab}

                A C = {ac} 

                D B = {db} 

                D C = ??

                """
            baseprompts[i].append(temp)

    
    return baseprompts, basepairs




#-------------------------------------------------------------------------------------------------
#                        MATH-IFYING LO PROBLEMS + TESTING OPERATIONS
#-------------------------------------------------------------------------------------------------



def get_unfamiliar_ops(n_ops, op_type):

    # get different representations for explicitly marked but unfamiliar operations

    if op_type == "unfamiliar_math":
        # type 1: unfamiliar and math-y (removed o)
        mathy_symbols = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ','ν', 'ξ', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω']
        ops = random.sample(mathy_symbols, n_ops)
    
    elif op_type == "unfamiliar_random": 
        # type 2: unfamiliar and random
        ops = get_random_tokens_sampled(n_ops, max_tok_len=2)

    else:
        raise ValueError("Not using current unfamiliar operations!")

    return ops







#-------------------------------------------------------------------------------------------------
#                               PROMPT MODELS, COLLECT RESPONSES
#-------------------------------------------------------------------------------------------------

def prompt_model_minpair():
    """
    format questions to prompt model with, pass to model, format output to CSV
    """
    prompts = [] 
    responses = {}
    models = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "o1-mini"]
    models = []

    for template in paradigms: 
        prompt_temp = f"""Here is a puzzle. Can you solve it? \n Please output only the answer (in place of the ??) and nothing else!
        {template}
        """

        prompts.append(prompt_temp)

    for prompt in prompts:
        for model in models:
            # response = query_gpt(prompt, model)
            response = huit_query_gpt(prompt, model)
            responses[(model, prompt)] = response

    # Get current date and time (so as to log when expts happened).
    
    time_now = datetime.now().strftime('%m-%d_%H-%M')

    with open(f'minpair_{time_now}.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'Prompt', 'Response']) 
        for (model, prompt), response in responses.items():
            writer.writerow([model, prompt, response])

def prompt_model_bases():
    """
    format base-4-to-19 questions to prompt model with, pass to model, format output to CSV
    """
    prompts = [] 
    responses = {}
    # models = ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]
    models = ["o1-mini"]


    random_flag = True
    b_temps, bp = all_bases_test(random_flag=random_flag)

 
    for template in b_temps: 
        prompt_temp = f"""Here is a puzzle. Can you solve it? \n Please output only the answer (in place of the ??) and nothing else!
        {template[0]}
        """

        prompts.append(prompt_temp)


    bases = [i for i in range (4, 20)]
    bases.remove(10)
        
    for prompt in prompts:
        for model in models:
            # response = query_gpt(prompt, model)
            response = huit_query_gpt(prompt, model)
            responses[(model, prompt)] = response

    # Get current date and time (so as to log when expts happened).
    
    time_now = datetime.now().strftime('%m-%d_%H-%M')
    exp_type = "RANDOM" if random_flag else "ABCD"

    with open(f'bases_4to20_{exp_type}_ws_o1m_{time_now}.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        m = len(models)
        writer.writerow(['Model', 'Base', 'Response', 'Correct Response', 'Prompt']) 
        for i, ((model, prompt), response) in enumerate(responses.items()):
            actual_res = bp[i//m]['dc'][0] 
            writer.writerow([model, bases[i//m], response, actual_res, prompt])



def prompt_model_mathified(dataset, expt_type, sing_bool, n_iters):
    """
    NOTE: would recommend using prompt_model_standardized() with the default prompt_temp, much cleaner. 
    Get the problems in the 4-operator-variations, prompt model, get response.
    Default dataset is "both". UKLO and IOL problems can also be queried separately by passing "uklo" or "iol".
    """

    if dataset not in ["uklo", "iol", "both"]:
        raise ValueError("Pick a valid dataset.")
    prompts = [] 
    responses = {}
    models = ["o1-mini"]


    prob_langs_uklo = ["waorani", "gumatj", "ngkolmpu", "yoruba", "georgian", "northern-pame"]
    prob_langs_iol = ["drehu", "ndom", "birom", "umbu-ungu"]

    prob_langs_both = ["waorani", "gumatj", "ngkolmpu", "yoruba", "georgian", "northern-pame", "yup'ik", "drehu", "ndom", "umbu-ungu"]

    prob_tag = "lo" if dataset == "uklo" else "iol" if dataset == "iol" else "both"
    
    prob_langs = prob_langs_both
    prob_langs = [lang for lang in prob_langs for _ in range(n_iters)]


    
    mathified_probs, mathified_ans = [], []

    temp_probs, temp_ans = [], []
    
    for i in range(n_iters):
        if dataset == "uklo":
            mprob, msol = mathify_uklo(expt_type=expt_type, sing_bool=sing_bool)
        elif dataset == "fix":
            mprob, msol = mathify_uklo(expt_type=expt_type, sing_bool=sing_bool)
        elif dataset == "both":
            mprob_u, msol_u = mathify_uklo(expt_type=expt_type, sing_bool=sing_bool)
            mprob_i, msol_i = mathify_iol(expt_type=expt_type, sing_bool=sing_bool)
            mprob = mprob_u + mprob_i
            msol = msol_u + msol_i 
        else:
            mprob, msol = mathify_iol(expt_type=expt_type, sing_bool=sing_bool)


        if not temp_probs:  
            temp_probs = [[] for _ in range(len(mprob))]
            temp_ans = [[] for _ in range(len(msol))]

        for idx, (p, s) in enumerate(zip(mprob, msol)):
            temp_probs[idx].append(p)
            temp_ans[idx].append(s)

    # flatten to group by type of problem (so all the explicit token ones go together, etc)
    mathified_probs.extend(sum(temp_probs, []))
    mathified_ans.extend(sum(temp_ans, []))
 

    for template in mathified_probs: 
        prompt_temp = f"""Here is a puzzle. Can you solve it? \n All numbers in the problem are positive integers. Please output only the answer, which is a positive integer, in place of the ?? and output nothing else! \n 
        
        
        {template} """ 


        prompts.append(prompt_temp)
        
    for prompt in tqdm(prompts):
        for model in models:
            response = query_gpt(prompt, model)
            responses[(model, prompt)] = response

    # Get current date and time (so as to log when expts happened).
    
    time_now = datetime.now().strftime('%m-%d_%H-%M')
    exp_type = f"{prob_tag}probs_o1m-1TOK-{expt_type}"

    with open(f'{exp_type}.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Model', 'Response', 'Correct Response', 'Language', 'Prompt']) 
        for i, ((model, prompt), response) in enumerate(tqdm(responses.items())):
            m = len(models)
            actual_res = mathified_ans[i//m]
            lang = prob_langs[i]
            writer.writerow([model, response, actual_res, lang, prompt])

    


def prompt_model_standardized(mprob, msol, expt_type, sing_bool, n_iters):
    """
    10 problems (UKLO + IOL), 5 iterations per problem. Default + different variations of context. 
    NOTE: Pick prompt out of implicit_prompt_temp, context_prompt_temp, base_prompt_temp, and (default) prompt_temp.
    """
    prompts = [] 
    responses = {}
    models = ["o1-mini"]

    prob_langs = ["waorani", "gumatj", "ngkolmpu", "yoruba", "georgian", "northern-pame", "yup'ik", "drehu", "ndom", "umbu-ungu"]
    
    base_dict = {"waorani" : 5,
                  "gumatj" : 5, 
                  "ngkolmpu" : 6, 
                  "yoruba" : 20, 
                  "georgian" : 20, 
                  "northern-pame" : 8, 
                  "yup'ik" : 20, 
                  "drehu" : 20, 
                  "ndom" : 6, 
                  "umbu-ungu" : 24}
    

    prob_tag = "both"
 

    
    mathified_probs, mathified_ans = [], []


    for p in mprob:
        mathified_probs.extend([p] * n_iters)

    mathified_ans = msol


    for (i, template) in enumerate(mathified_probs): 
        lang = prob_langs[i//n_iters]
        base = base_dict[lang]
        prob_lang = prob_langs[i//n_iters].capitalize()

        implicit_prompt_temp = f"""Here is a puzzle based on numbers in a language. In this language, numbers may be constructed using operations that are implicit - for example, in the number twenty-nine (20 + 9) contains an implicit addition, and the number five hundred (5 * 100) contains an implicit multiplication operation. Can you solve it? \n All numbers in the problem are positive integers. Please output only the answer, which is a positive integer, in place of the ?? and output nothing else! \n               
        {template} """ 


        context_prompt_temp = f"""Here is a puzzle based on numbers in the {prob_lang} language. Can you solve it? \n All numbers in the problem are positive integers. Please output only the answer, which is a positive integer, in place of the ?? and output nothing else! \n               
        {template} """ 

        base_prompt_temp = f"""Here is a puzzle based on numbers in a language that uses a base-{base} numeral system. Can you solve it? \n All numbers in the problem are positive integers. Please output only the answer, which is a positive integer, in place of the ?? and output nothing else! \n               
        {template} """ 

        prompt_temp = f"""Here is a puzzle. Can you solve it? \n All numbers in the problem are positive integers. Please output only the answer, which is a positive integer, in place of the ?? and output nothing else! \n               
        {template} """ 

        prompts.append(prompt_temp)

    responses = {}

    for prompt in tqdm(prompts):
        for model in models:
            # response = query_gpt(prompt, model)
            response = huit_query_gpt(prompt, model)
            key = (model, prompt)
            if key not in responses:
                responses[key] = []
            responses[key].append(response)
    
        
    # NOTE: for our results, we use "BASE", "CONTEXT", and "IMPLCTXT" as flags in the result CSVs.

    exp_type = f"{prob_tag}probs-o1m-MultiTOK-{expt_type}"

    with open(f'{exp_type}.csv', 'a', newline='', encoding='utf-8') as file: 
        writer = csv.writer(file)
        writer.writerow(['Model', 'Response', 'Correct Response', 'Language', 'Prompt']) 
        for i, ((model, prompt), response_list) in enumerate(tqdm(responses.items())):
            for response in response_list:
                actual_res = mathified_ans[i]
                lang = prob_langs[i]
                writer.writerow([model, response, actual_res, lang, prompt])
                print(f"Writing {lang} to CSV!")





def main():

    """
    Set sing_bool = False for multi-token, True for single_token responses.

    """


   # STANDARDIZED PROMPTING

    mprob_explfam, mprob_expl_unfam_m, mprob_expl_unfam_r, mprob_impl, msol = mathify_standardized(sing_bool=False)

    mprob_list = [mprob_explfam, mprob_expl_unfam_m, mprob_expl_unfam_r, mprob_impl]



    for i, expt in enumerate(["expl_fam", "expl_unfam_m", "expl_unfam_r", "impl_random"]): 
        print(f"\nExperiment {i+1} of 4\n")
        prompt_model_standardized(mprob=mprob_list[i], 
                                  msol=msol, 
                                expt_type=expt,
                                sing_bool=False,
                                n_iters=5)
        print(f"\nDone with experiment {i+1}!\n")

# ----------------------------

# You can also use the previous prompting function if you want to split UKLO / IOL problems.

    # for i, expt in enumerate(["impl_random"]): #"impl_random", "expl_fam", "expl_unfam_m", "expl_unfam_r" / "expl_fam", "expl_unfam_m", 
    #     print(f"\nExperiment {i+1} of 4\n")
    #     prompt_model_mathified(dataset="both", 
    #                             expt_type=expt,
    #                             sing_bool=False,
    #                             n_iters=1)
    #     print(f"\nDone with experiment {i+1}!\n")


    



if __name__ == "__main__":
    main()

