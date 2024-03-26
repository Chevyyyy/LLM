import torch
from trl import SFTTrainer
from datasets import load_dataset
from peft import * 
import textwrap
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments


train_dataset=load_dataset("wikipedia", language="sw", date="20220120")
# train_dataset = load_dataset("tatsu-lab/alpaca", split="train")
print(train_dataset)
pandas_format = train_dataset.to_pandas()
print(pandas_format.head())
for index in range(3):
   print("---"*15)
   print("Instruction:       {}".format(textwrap.fill(pandas_format.iloc[index]["instruction"],    
       width=50)))
   print("Output:       {}".format(textwrap.fill(pandas_format.iloc[index]["output"],  
       width=50)))
   print("Text:        {}".format(textwrap.fill(pandas_format.iloc[index]["text"],  
       width=50)))

       
       
pretrained_model_name = "Salesforce/xgen-7b-8k-base"
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True)