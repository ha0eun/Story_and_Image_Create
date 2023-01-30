import torch
import transformers
from transformers import AutoModelWithLMHead, PreTrainedTokenizerFast
from fastai.text.all import *
import fastai
import re

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',        # 각각의 요소는 파라미터..?
  pad_token='<pad>', mask_token='<mask>')

model = AutoModelWithLMHead.from_pretrained("skt/kogpt2-base-v2")

# test tokenizer
print(tokenizer.tokenize("안녕하세요. 한국어 GPT-2 입니다.😤:)l^o"))

text = """벼랑 끝 외딴 호텔에 갇힌 용의자들."""
input_ids = tokenizer.encode(text)
gen_ids = model.generate(torch.tensor([input_ids]),
                               max_length=100,
                               repetition_penalty=2.0,
                               pad_token_id=tokenizer.pad_token_id,
                               eos_token_id=tokenizer.eos_token_id,
                               bos_token_id=tokenizer.bos_token_id,
                               use_cache=True
                            )

generated = tokenizer.decode(gen_ids[0,:].tolist())
print(generated)


with open('crawling.csv', encoding='UTF8') as f:
    lines = f.read()
    lines = " ".join(lines.split())
print(len(lines))

#%% [code]
lines=re.sub('\(계속\).*?[●○]', '', lines)
lines=re.sub('[●○]', '', lines)
print(len(lines))

class TransformersTokenizer(Transform):
    def __init__(self, tokenizer): self.tokenizer = tokenizer
    def encodes(self, x):
        toks = self.tokenizer.tokenize(x)
        return tensor(self.tokenizer.convert_tokens_to_ids(toks))
    def decodes(self, x): return TitledStr(self.tokenizer.decode(x.cpu().numpy()))


train=lines[:int(len(lines)*0.9)]
test=lines[int(len(lines)*0.9):]
splits = [[0],[1]]
print('1')
tls = TfmdLists([train,test], TransformersTokenizer(tokenizer), splits=splits, dl_type=LMDataLoader)
print('2')
batch,seq_len = 16, 512
print('3')
dls = tls.dataloaders(bs=batch, seq_len=seq_len)
print('4')


class DropOutput(Callback):
    def after_pred(self): self.learn.pred = self.pred[0]


learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), cbs=[DropOutput], metrics=Perplexity()).to_fp16()
lr = learn.lr_find()
print(lr)
learn.fit_one_cycle(4, lr)


prompt='원하는 문장을 입력하기'
prompt_ids = tokenizer.encode(prompt)
inp = tensor(prompt_ids)[None]
preds = learn.model.generate(inp,
                           max_length=30,
                           pad_token_id=tokenizer.pad_token_id,
                           eos_token_id=tokenizer.eos_token_id,
                           bos_token_id=tokenizer.bos_token_id,
                           repetition_penalty=2.0,
                           use_cache=True
                          )
tokenizer.decode(preds[0].cpu().numpy())


learn.model.save_pretrained("kogpt2novel_backup")
model.save_pretrained("kogpt2novel")
tokenizer.save_pretrained("kogpt2novel")

