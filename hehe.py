# ------------------------ Code --------------------------------------

## normal import 
import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp
import argparse

import torch
import time

# ## transformers related import
# from transformers import T5Tokenizer,T5ForConditionalGeneration
# from transformers import BertTokenizer
# from transformers import pipeline
import transformers
from transformers import AutoTokenizer, AutoModel

def setup_train_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="cpu",
                        type=str, required=False,)
    return parser.parse_args()


def main():
    args=setup_train_args()
    # tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b",
    #                                           trust_remote_code=True)
    # model = AutoModel.from_pretrained("THUDM/chatglm-6b",
    #                                   trust_remote_code=True).half().cuda()
    if args.device=="cpu":
        device="cpu"
    else:
        device=f"cuda:{args.device}"


    # ## by pretrained model
    # num=5
    # mls=[]
    # for _ in range(num):
    #     # pth="/home/liangzi/models/t5-small/"
    #     pth="/home/liangzi/models/mbart-large-50/"
    #     mls.append(AutoModel.from_pretrained(pth).to(device))

    N=90000 # 32GB
    N=80000 # 25GB
    mat=torch.ones((N,N)).to(device)

    time.sleep(3600*24*365)

## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


