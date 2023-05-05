import time

import transformers
from datasets import load_dataset
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import BertConfig, BertModel
from transformers import BertTokenizer
from tqdm import tqdm


def read_data():
    dataset = load_dataset('text', data_files="small_jTrans_proj.txt")
    a = dataset['train']
    b = dataset['train'][0]
    funcs = dataset['train'][:100]
    target_l = []
    for f in funcs['text']:
        f_p = f[:-1].split('\t')
        target_l.append(f_p)
    print(dataset['train'][0])
    print('done')


if __name__ == '__main__':
    # classifier = pipeline("sentiment-analysis")
    # res = classifier("today is Friday.")
    # print(res)


    # unmasker = pipeline('fill-mask')
    # res = unmasker('today is <mask>.', top_k=5)
    # print(res)

    # checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    # tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    #
    # raw_inputs = [
    #     "I've been waiting for a HuggingFace course my whole life.",
    #     "I hate this so much!",
    #     "endbr64 push rbp mov rbp",
    # ]
    # inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
    # print(inputs)

    # Building the config
    config = BertConfig()
    model = BertModel(config)
    print(config)

    tokenizer_path = '../jTrans/jtrans_tokenizer'
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    print(tokenizer)

    # progress_bar = tqdm(range(100))
    # for i in range(100):
    #     time.sleep(1)
    #     progress_bar.update(1)
