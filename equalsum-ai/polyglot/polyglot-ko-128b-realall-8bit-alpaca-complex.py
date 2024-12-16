#!/usr/bin/env python
# coding: utf-8
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
lr = 5e-5
num_epochs = 20
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





# %%




# <h2>데이터 전처리</h2>

# %%
from sklearn.model_selection import train_test_split

df = pd.read_csv('240819_dataset_augmented_lreason_wg.csv')
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
평가기준은 '연결성'입니다. 평가기준에 대한 설명을 보고, 이를 엄격히 준수해서 평가합니다.
평가 기준에 따라 평가할 때, 상위목표, 업종, 조직명을 반드시 참고해 주십시오.

<평가기준>
## 평가기준에 대한 설명
연결성은 다음과 같은 두 가지 기준으로 평가합니다.
기준 1. 상위목표를 달성하는 데 핵심결과가 기여하는가?
기준 2. 핵심결과 문장이 나타내는 바가 구체적이고 명확한가?
## 평가기준에 따른 점수
평가기준에 대한 설명을 반드시 엄격히 준수합니다. 그리고 평가기준을 만족하는 정도에 따라 점수를 다르게 주세요.
4.5~5점: 기준 1과 기준 2를 모두 만족한다.
3~4점: 기준 1은 만족하지만, 기준 2는 만족하지 못한다.
2~2.5점: 기준2는 만족하지만, 기준 1은 만족하지 못한다.
1~1.5점: 기준 1과 기준 2를 모두 만족하지 못한다.

<출력>
출력은 반드시 아래와 같은 형식을 엄격히 지켜주세요.

점수: [1부터 5사이의 점수, 소수점 단위 0.5]
이유: [점수를 준 이유]
"""

system_measurability_prompt = """
###질문:
당신은 문장 평가 전문가입니다.
평가기준은 '측정 가능성'입니다. 평가기준에 대한 설명을 보고, 이를 엄격히 준수해서 평가합니다.
평가 기준에 따라 평가할 때, 상위목표, 업종, 조직명을 반드시 참고해 주십시오.

<평가기준>
## 평가기준에 대한 설명
측정 가능성은 다음과 같은 세 가지 기준으로 평가합니다.
기준 1. 측정할 대상이 있는가?
기준 2. 측정 대상을 측정할 수 있는가?
기준 3. 측정 대상이 양적으로 평가되는가?
## 평가기준에 따른 점수
평가기준에 대한 설명을 반드시 엄격히 준수합니다. 그리고 평가기준을 만족하는 정도에 따라 점수를 다르게 주세요.
5점: 기준 1과 기준 2, 기준 3을 모두 만족한다.
3.5~4.5점: 기준 1과 기준 2는 만족하지만, 기준 3을 만족하지 못한다. 
2~3점: 기준 1은 만족하지만, 기준 2와 기준 3은 만족하지 못한다.
1~1.5점: 기준 1과 기준 2, 기준 3을 모두 만족하지 못한다.

<출력>
출력은 반드시 아래와 같은 형식을 엄격히 지켜주세요.

점수: [1부터 5사이의 점수, 소수점 단위 0.5]
이유: [점수를 준 이유]
"""

system_directivity_prompt = """
###질문:
당신은 문장 평가 전문가입니다.
평가기준은 '결과 지향성'입니다. 평가기준에 대한 설명을 보고, 이를 엄격히 준수해서 평가합니다.
평가 기준에 따라 평가할 때, 상위목표, 업종, 조직명을 반드시 참고해 주십시오.

<평가기준>
## 평가기준에 대한 설명
결과 지향성은 다음과 같은 두 가지 기준으로 평가합니다.
기준 1. 핵심결과 문장이 방향, 행동이 아닌 결과를 평가하는가?
기준 2. 핵심결과 문장에 '무엇이 달라지는가'와 '얼마나 달라지는가'가 명시되어있는가?
## 평가기준에 따른 점수
평가기준에 대한 설명을 반드시 엄격히 준수합니다. 그리고 평가기준을 만족하는 정도에 따라 점수를 다르게 주세요.
4.5~5점: 기준 1과 기준 2를 모두 만족한다.
3.5~4점: 기준 1은 만족하지만, 기준 2는 만족하지 못한다.
2.5~3점: 기준2는 만족하지만, 기준 1은 만족하지 못한다.
1~2점: 기준 1과 기준 2를 모두 만족하지 못한다.

<출력>
출력은 반드시 아래와 같은 형식을 엄격히 지켜주세요.

점수: [1부터 5사이의 점수, 소수점 단위 0.5]
이유: [점수를 준 이유]
"""

objective_template = """
###맥락:
상위목표 = {upper_objective}
현재목표 = {objective}
업종 = {field}.
조직명 = {team}.

###답변:
점수:"""

objective_output_template = """
###맥락:
상위목표 = {upper_objective}
현재목표 = {objective}
업종 = {field}.
조직명 = {team}.

###답변:
점수: {score}점
이유: {reason}.<|endoftext|>"""

system_align_prompt = """
###질문:
당신은 문장 평가 전문가입니다.
평가기준은 '얼라인'입니다. 평가기준에 대한 설명을 보고, 이를 엄격히 준수해서 평가합니다. 상위목표가 '최상위 O'라면 현재목표가 최상위 목표라는 것을 의미합니다.
평가 기준에 따라 평가할 때, 상위목표, 업종, 조직명을 반드시 참고해 주십시오.

<평가기준>
## 평가기준에 대한 설명
얼라인은 다음과 같은 두 가지 기준으로 평가합니다.
기준 1. 현재 목표가 최상위 목표가 아니라면 상위목표와 현재목표의 전략적 연결이 있는가?
기준 2. 상위목표와 현재목표의 전략적 연결이 직접적인가?
기준 3. 현재 목표의 초점이 명확한가?
## 평가기준에 따른 점수
평가기준에 대한 설명을 반드시 엄격히 준수합니다. 그리고 평가기준을 만족하는 정도에 따라 점수를 다르게 주세요.
4.5~5점: 기준 1과 기준 2, 기준 3을 모두 만족한다.
3~4점: 기준 1과 기준 2는 만족하지만, 기준 3을 만족하지 못한다.
2~2.5점: 기준 1은 만족하지만, 기준 2와 기준 3을 만족하지 못한다.
1~1.5점: 기준 1과 기준 2, 기준 3을 모두 만족하지 못한다.

<출력>
출력은 반드시 아래와 같은 형식을 엄격히 지켜주세요.

점수: [1부터 5사이의 점수, 소수점 단위 0.5]
이유: [점수를 준 이유]
"""

system_value_prompt = """
###질문:
당신은 문장 평가 전문가입니다.
평가기준은 '고객가치'입니다. 평가기준에 대한 설명을 보고, 이를 엄격히 준수해서 평가합니다.
평가 기준에 따라 평가할 때, 상위목표, 업종, 조직명을 반드시 참고해 주십시오.

<평가기준>
## 평가기준에 대한 설명
고객가치는 다음과 같은 두 가지 기준으로 평가합니다.
기준 1. 고객에게 제공하는 가치가 현재 고객에게 필요한가?
기준 2. 고객가치를 명확히 표현했는가?
기준 3. 고객에 관한 문제가 있는가?
## 평가기준에 따른 점수
평가기준에 대한 설명을 반드시 엄격히 준수합니다. 그리고 평가기준을 만족하는 정도에 따라 점수를 다르게 주세요.
4.5~5점: 기준 1과 기준 2를 만족한다.
3~4점: 기준2는 만족하지만, 기준 1은 만족하지 못한다.
2~2.5점: 기준 3은 만족하지만, 기준 1, 기준 2는 만족하지 못한다.
1~1.5점: 기준 1과 기준 2, 기준 3을 모두 만족하지 못한다.

<출력>
출력은 반드시 아래와 같은 형식을 엄격히 지켜주세요.

점수: [1부터 5사이의 점수, 소수점 단위 0.5]
이유: [점수를 준 이유]
"""


# %%


import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding_text = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        input_ids = encoding_text['input_ids'].squeeze(0)
        attention_mask = encoding_text['attention_mask'].squeeze(0)
        encoding_label = self.tokenizer(label, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        lastlabel = encoding_label['input_ids'].squeeze(0)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': lastlabel
        }


# %%
def processing_data(data_df):
    texts = []
    labels = []

    for idx, row in data_df.iterrows():
        if row["type"] == "Key Result":
            text = system_connectivity_prompt + keyresult_template.format(upper_objective=row["upper_objective"], keyresult=row["input_sentence"], field = row["field"], team = row["team"])
            texts.append(text)
            text = system_measurability_prompt + keyresult_template.format(upper_objective=row["upper_objective"], keyresult=row["input_sentence"], field = row["field"], team = row["team"])
            texts.append(text)
            text = system_directivity_prompt + keyresult_template.format(upper_objective=row["upper_objective"], keyresult=row["input_sentence"], field = row["field"], team = row["team"])
            texts.append(text)
            output = system_connectivity_prompt + keyresult_output_template.format(upper_objective=row["upper_objective"], keyresult=row["input_sentence"], field = row["field"], team = row["team"], reason = row["l_connectivity_description"], score = (row["connectivity"]))
            labels.append(output)
            output = system_measurability_prompt + keyresult_output_template.format(upper_objective=row["upper_objective"], keyresult=row["input_sentence"], field = row["field"], team = row["team"], reason = row["l_measurability_description"], score = (row["measurability"]))
            labels.append(output)
            output = system_directivity_prompt + keyresult_output_template.format(upper_objective=row["upper_objective"], keyresult=row["input_sentence"], field = row["field"], team = row["team"], reason = row["l_directivity_description"], score = (row["directivity"]))
            labels.append(output)
        else:
            text = system_align_prompt + objective_template.format(upper_objective=row["upper_objective"], objective=row["input_sentence"], field = row["field"], team = row["team"])
            texts.append(text)
            text = system_value_prompt + objective_template.format(upper_objective=row["upper_objective"], objective=row["input_sentence"], field = row["field"], team = row["team"])
            texts.append(text)
            
            output = system_align_prompt + objective_output_template.format(upper_objective=row["upper_objective"], objective=row["input_sentence"], field = row["field"], team = row["team"], reason = row["l_align_description"], score = row["customer_align"])
            labels.append(output)
            output = system_value_prompt + objective_output_template.format(upper_objective=row["upper_objective"], objective=row["input_sentence"], field = row["field"], team = row["team"], reason = row["l_value_description"], score = row["customer_value"])
            labels.append(output)
        
    return texts, labels


# %%
train_texts, train_labels = processing_data(train_df)
train_dataset = CustomDataset(train_texts, train_labels, tokenizer)

eval_texts, eval_labels = processing_data(test_df)
eval_dataset = CustomDataset(eval_texts, eval_labels, tokenizer)


# %%
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    do_eval=True,
    save_steps=save_step,
    eval_steps=save_step,
    logging_steps=logging_step,
    save_strategy='steps',
    evaluation_strategy='steps',
    learning_rate=lr,
    fp16=False,
    lr_scheduler_type="linear",
    run_name="1112_long_complex_prompt_8bits",
    load_best_model_at_end=True
)

# %%


tokenizer.pad_token = tokenizer.eos_token
model.config.use_cache = False
_ = model.to(device)
#model.load_state_dict(torch.load('0928_realall_complex_prompt_8bits_weights.pth'))


# %%
#점수학습
import mlflow

model.train()


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=1500)]
)
trainer.train()


# %%

torch.save(model.state_dict(), '1112_long_complex_prompt_8bits.pth')

