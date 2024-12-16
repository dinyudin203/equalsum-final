#!/usr/bin/env python
# coding: utf-8
# %%

# %%



# <h1>polyglot-ko</h1>

# <h2>환경설정</h2>


# %%




import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# %%




get_ipython().system('pip install -r requirements.txt')
get_ipython().system('pip install multiprocess==0.70.15')


# %%


get_ipython().system('pip install -q -U bitsandbytes accelerate peft tensorboard')


# %%


get_ipython().system('pip uninstall markupsafe --yes')
get_ipython().system('pip install markupsafe==2.0.1')


# %%


import torch

device = torch.device('cuda')


# %%


get_ipython().system('pip install -U bitsandbytes')


# %%


get_ipython().system('export LD_LIBRARY_PATH="~/.local/lib/python3.10/site-packages/bitsandbytes:$LD_LIBRARY_PATH"')


# %%


import pandas as pd
import torch.nn as nn
from tqdm.notebook import tqdm
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# %%


model_id = "beomi/KoAlpaca-Polyglot-12.8B"
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
poly_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")


# %%


from peft import prepare_model_for_kbit_training

poly_model.gradient_checkpointing_enable()
pre_model = prepare_model_for_kbit_training(poly_model)


# %%


#환경설정
max_length =2048
batch_size = 1
lr = 3e-4
num_epochs = 40
save_step = 1000
logging_step = 100


# %%


from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.01,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(pre_model, lora_config)


# %%


# <h2>데이터 전처리</h2>


# %%
from sklearn.model_selection import train_test_split

df = pd.read_csv('240819_dataset_augmented_guideline.csv')
df = df.dropna(subset=["input_sentence", "upper_objective", "type"])
dfObjective = df[df["type"] == "Objective"]
dfKeyresult = df[df["type"] == "Key Result"]
trainObjective, testObjective = train_test_split(dfObjective, test_size=0.2, random_state = 99)
trainKeyresult, testKeyresult = train_test_split(dfKeyresult, test_size=0.2, random_state = 99)
train_df = pd.concat([trainObjective, trainKeyresult], ignore_index=True)
test_df = pd.concat([testObjective, testKeyresult], ignore_index=True)


# %%
keyresult_template = """
###맥락:
상위목표 = {upper_objective}
핵심결과 = {keyresult}
업종 = {field}.
조직명 = {team}.

###답변:
점수:"""

keyresult_reason_template = """
상위목표 = {upper_objective}
핵심결과 = {keyresult}
업종 = {field}.
조직명 = {team}.

###답변:
점수: {score}점
이유:"""

keyresult_output_template = """
###맥락:
상위목표 = {upper_objective}
핵심결과 = {keyresult}
업종 = {field}.
조직명 = {team}.

###답변:
점수: {score}점
이유: {reason}.<|endoftext|>"""

system_connectivity_prompt = """
###질문:
당신은 문장 평가 전문가입니다.
평가기준은 '연결성'입니다. '연관성'이란 상위목표와 핵심결과의 연관성을 의미합니다. 상위목표, 핵심결과, 업종, 팀명을 기반으로 상위목표와 핵심결과의 연관성을 1점에서 5점사이로 평가하세요.

<답변 형식>
답변은 반드시 아래와 같은 형식을 엄격히 지켜주세요.

점수: [1부터 5사이의 점수, 소수점 단위 0.5]
이유: [점수를 준 이유]
"""

system_measurability_prompt = """
###질문:
당신은 문장 평가 전문가입니다.
평가기준은 '측정 가능성'입니다. '측정 가능성'이란 핵심결과가 측정 가능한지를 의미합니다. 상위목표, 핵심결과, 업종, 팀명을 기반으로 핵심결과의 측정 가능성을 1점에서 5점사이로 평가하세요.

<답변 형식>
답변은 반드시 아래와 같은 형식을 엄격히 지켜주세요.

점수: [1부터 5사이의 점수, 소수점 단위 0.5]
이유: [점수를 준 이유]
"""

system_directivity_prompt = """
###질문:
당신은 문장 평가 전문가입니다.
평가기준은 '결과 지향성'입니다. '결과 지향성'이란 핵심결과가 과정보다 결과에 집중하고 있는지를 의미합니다. 상위목표, 핵심결과, 업종, 팀명을 기반으로 핵심결과의 결과 지향성을 1점에서 5점사이로 평가하세요.

<답변 형식>
답변은 반드시 아래와 같은 형식을 엄격히 지켜주세요.

점수: [1부터 5사이의 점수, 소수점 단위 0.5]
이유: [점수를 준 이유]
"""

objective_template = """
###맥락:
상위목표 = {upper_objective}
해당목표 = {objective}
업종 = {field}.
조직명 = {team}.

###답변:
점수:"""

objective_output_template = """
###맥락:
상위목표 = {upper_objective}
해당목표 = {objective}
업종 = {field}.
조직명 = {team}.

###답변:
점수: {score}점
이유: {reason}.<|endoftext|>"""

system_align_prompt = """
###질문:
당신은 문장 평가 전문가입니다.
평가기준은 '상위목표와의 연결성'입니다. '상위목표와의 연결성'이란 해당목표가 상위목표와 관련이 얼마나 있는지를 의미합니다. 상위목표, 해당목표, 업종, 팀명을 기반으로 해당목표의 상위목표와의 연결성을 1점에서 5점사이로 평가하세요. 상위목표가 '최상위 O'면 해당목표가 최상위 목표라는 것을 의미합니다. 이 경우 해당목표와 기업의 비전이 얼마나 관련있는지를 평가하세요.

<답변 형식>
답변은 반드시 아래와 같은 형식을 엄격히 지켜주세요.

점수: [1부터 5사이의 점수, 소수점 단위 0.5]
이유: [점수를 준 이유]
"""

system_value_prompt = """
###질문:
당신은 문장 평가 전문가입니다.
평가기준은 '고객가치'입니다. '고객가치'란 해당목표가 고객가치를 포함하고 있는지를 의미합니다. 상위목표, 해당목표, 업종, 팀명을 기반으로 고객가치를 1점에서 5점사이로 평가하세요.

<답변 형식>
답변은 반드시 아래와 같은 형식을 엄격히 지켜주세요.

점수: [1부터 5사이의 점수, 소수점 단위 0.5]
이유: [점수를 준 이유]
"""


# %%




tokenizer.pad_token = tokenizer.eos_token
model.config.use_cache = False
model.load_state_dict(torch.load('1008_realall_simple_prompt_8bits_weights.pth'))


# %%


##모델 실행


# %%


_ = model.eval()


# %%


def generation(text, 
               max_new_tokens = 512,
               repetition_penalty = 2.0,
               eval_type=0
):
    if(eval_type==0):
        input_text = system_connectivity_prompt + text
    elif(eval_type==1):
        input_text = system_measurability_prompt + text
    elif(eval_type==2):
        input_text = system_directivity_prompt + text
    elif(eval_type==3):
        input_text = system_align_prompt + text
    elif(eval_type==4):
        input_text = system_value_prompt + text
    else:
        return 'error'
    inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False).to(device)
    
    generate_kwargs = dict(
        input_ids=inputs["input_ids"],
        max_length=max_new_tokens,
        do_sample=True,  # Enable sampling
        top_p=0.92,
        top_k=50,
        temperature=0.2,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id= tokenizer.eos_token_id,
        repetition_penalty=repetition_penalty,
    )
    outputs = model.generate(**generate_kwargs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# %%


import re

def score_and_reason(text):
    pattern = r"###답변:\s*점수\s*:\s*(\d+(\.\d+)?)(?:\s*.*?)?\s*이유\s*:\s*(.*?)(?=\n|$)"
    match = re.search(pattern, text, re.DOTALL)
    
    result = {
            "score": None,
            "reason": None
    }
    if match:
        result = {
            "score": float(match.group(1)),
            "reason": match.group(3).strip()
        }
    return result


# %%
key_col = ["connectivity", "measurability", "directivity"]
obj_col = ["align", "value"]
for idx, row in test_df.iterrows():
    text = ""
    col = key_col
    num = 0
    if row["type"] == "Key Result":
        text = keyresult_template.format(upper_objective=row["upper_objective"], keyresult=row["input_sentence"], field = row["field"], team = row["team"])
    else:
        text = objective_template.format(upper_objective=row["upper_objective"], objective=row["input_sentence"], field = row["field"], team = row["team"])
        col = obj_col
        num += 3
    for sidx, name in enumerate(col):
        for i in range(3):
            result_text = generation(text, repetition_penalty=1.2, eval_type=sidx+num)
            result = score_and_reason(result_text)
            torch.cuda.empty_cache()
            print(result_text)
            print(result)
            print()
            print('-'*100)
            print()
            if result["score"] == None and i != 2:
                continue
            test_df.loc[idx, "predict_" + name + "_score"] = result["score"]
            test_df.loc[idx, "predict_"+ name +"_description"] = result["reason"]
            break    


# %%
test_df.to_csv('1008_result_simple.csv', index=False)

