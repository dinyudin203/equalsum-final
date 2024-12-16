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
num_epochs = 30
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
            output = system_connectivity_prompt + keyresult_output_template.format(upper_objective=row["upper_objective"], keyresult=row["input_sentence"], field = row["field"], team = row["team"], reason = row["connectivity_description"], score = (row["connectivity"]))
            labels.append(output)
            output = system_measurability_prompt + keyresult_output_template.format(upper_objective=row["upper_objective"], keyresult=row["input_sentence"], field = row["field"], team = row["team"], reason = row["measurability_description"], score = (row["measurability"]))
            labels.append(output)
            output = system_directivity_prompt + keyresult_output_template.format(upper_objective=row["upper_objective"], keyresult=row["input_sentence"], field = row["field"], team = row["team"], reason = row["directivity_description"], score = (row["directivity"]))
            labels.append(output)
        else:
            text = system_align_prompt + objective_template.format(upper_objective=row["upper_objective"], objective=row["input_sentence"], field = row["field"], team = row["team"])
            texts.append(text)
            text = system_value_prompt + objective_template.format(upper_objective=row["upper_objective"], objective=row["input_sentence"], field = row["field"], team = row["team"])
            texts.append(text)
            
            output = system_align_prompt + objective_output_template.format(upper_objective=row["upper_objective"], objective=row["input_sentence"], field = row["field"], team = row["team"], reason = row["align_description"], score = row["customer_align"])
            labels.append(output)
            output = system_value_prompt + objective_output_template.format(upper_objective=row["upper_objective"], objective=row["input_sentence"], field = row["field"], team = row["team"], reason = row["value_description"], score = row["customer_value"])
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
    run_name="1008_realall_simple_prompt_8bits",
    load_best_model_at_end=True
)

# %%


tokenizer.pad_token = tokenizer.eos_token
model.config.use_cache = False
#_ = model.to(device)
model.load_state_dict(torch.load('0928_realall_simple_prompt_8bits_weights.pth'))


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

torch.save(model.state_dict(), '1008_realall_simple_prompt_8bits_weights.pth')

