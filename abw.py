import numpy as np
from tqdm.auto import tqdm
from time import time
import json, pickle, os, string, kenlm, json
from collections import defaultdict, Counter
from itertools import groupby
import Levenshtein as Lev
import math


def lse(*args):
  """
  Stable log sum exp.
  """
  if all(a == -float('inf') for a in args):
      return -float('inf')
  a_max = max(args)
  lsp = math.log(sum(math.exp(a - a_max)
                      for a in args))
  return a_max + lsp

  #s1 = True text
#s2 = predicted text

def wer_(s1, s2):
    """
    Computes the Word Error Rate, defined as the edit distance between the
    two provided sentences after tokenizing to words.
    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """

    # build mapping of words to integers
    b = set(s1.split() + s2.split())
    word2char = dict(zip(b, range(len(b))))

    # map the words to a char array (Levenshtein packages only accepts
    # strings)
    w1 = [chr(word2char[w]) for w in s1.split()]
    w2 = [chr(word2char[w]) for w in s2.split()]
    
    return Lev.distance(''.join(w1), ''.join(w2))

def cer_(s1, s2):
    """
    Computes the Character Error Rate, defined as the edit distance.

    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """
    s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')

    return Lev.distance(s1, s2)

labels = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "

def sort_beam(ptot,k):
    if len(ptot) < k:
        return [i for i in ptot.keys()]
    else:
        dict_ = sorted(dict((v,k) for k,v in ptot.items()).items(),reverse=True)[:k]
        return [i[1] for i in dict_]


def ctc_beam_search(out,labels, prune=0.0001, k=20, lm=None,alpha=0.3,beta=12):
    
    bc_i = 0 # blank/special charatcter index 
    F = out.shape[1]
    dummy_ = np.vstack((np.zeros(F), out))
    out = np.log(np.vstack((np.zeros(F), out)))
    steps = out.shape[0]
    
    pb, pnb = defaultdict(Counter), defaultdict(Counter)
    pb[0][''], pnb[0][''] = 0, -float("inf")
    prev_beams = ['']
    for t in range(1,steps):
        pruned_alphabet = [labels[i] for i in np.where(dummy_[t] > prune)[0]]
        for b in prev_beams:
            
            for c_t in pruned_alphabet:
                index = labels.index(c_t)
                if c_t == "_": #Extending with a blank
                    pb[t][b] = lse(pb[t][b],out[t][index]+pb[t-1][b], out[t][index]+pnb[t-1][b])
                    
                    continue
                
                else:
                    i_plus = b + c_t
                    if len(b.replace(' ', '')) > 0  and c_t == b[-1]: #Extending with the same character as the last one
                        pnb[t][b] = lse(pnb[t][b],out[t][index]+pnb[t-1][b])
                        pnb[t][i_plus] = lse(pnb[t][i_plus],out[t][index]+pb[t-1][b])
                        
                    else:
                        pnb[t][i_plus] = lse(out[t][index]+pb[t-1][b], out[t][index]+pnb[t-1][b])

#                     If the new beam is not in the previous beams
                    if i_plus not in prev_beams:
                        pb[t][i_plus] = lse(pb[t][i_plus],out[t][labels.index("_")]+pb[t - 1][i_plus], out[t][labels.index("_")]+ pnb[t - 1][i_plus])
                        pnb[t][i_plus] = lse(pnb[t][i_plus],out[t][index] + pnb[t - 1][i_plus])


        keys = set(pb[t].keys()).union(set(pnb[t].keys()))
        ptot = {i:pb[t][i]+pnb[t][i] for i in keys }
        
        #lm score
        if alpha != 0 or beta != 0:
            for word in ptot.keys():
                if len(word.replace(' ', '')) > 0 and word[-1] == ' ':
                        if word.strip().upper().split()[-1] in lm:
                            prob = 10**[i for i in lm.full_scores(word.strip().upper(),eos=False,bos=False)][-1][0]
                            prob_ = alpha*math.log(prob)
                        else: prob_ = -100

                        words = len(word.split()) + 1
                        word_inser = beta*math.log(words)
#                         print(word,prob_,word_inser,ptot[word],ptot[word]+prob_+word_inser,lse(ptot[word],prob_+word_inser))
                        ptot[word] += prob_ + word_inser
    
        prev_beams = sort_beam(ptot,k)
        
    return prev_beams[0], pb, pnb


import os
import sys

sys.path.append("/home/hemant/E2E_NER-Through-Speech/S2T/")
# os.chdir("/home/hemant/sopi_deep/")
os.chdir("/home/hemant/E2E_NER-Through-Speech/S2T/")
from opts import add_decoder_args, add_inference_args
from utils import load_model
import os
import argparse

import numpy as np
import torch
from tqdm import tqdm
from data.data_loader import SpectrogramParser

from opts import add_decoder_args, add_inference_args
from utils import load_model

alpha, beta = np.linspace(0,2,15), np.linspace(0,6,15)
values = [[i,j] for j in beta for i in alpha]

total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
output = 'alpha\tbeta\twer\n'

prune = 0.00001
beam_width = 75
alpha = 0
beta = 0
lm = kenlm.LanguageModel('/home/hemant/asr_wm/data/ner/align_wav_txt/4_gram.arpa')

with open("/home/hemant/asr_wm/dev.csv","r") as f:
    csv = f.readlines()

torch.set_grad_enabled(False)
device = torch.device("cuda")
model = load_model(device, "/home/hemant/asr_wm/models/deep/final.pth")
spect_parser = SpectrogramParser(model.audio_conf, normalize=True)

torch.cuda.set_device(int(0))

for alpha,beta in tqdm(values):
    
    for i in csv:
        audio_path, reference_path = i.split(",")

        spect = spect_parser.parse_audio(audio_path).contiguous()
        spect = spect.view(1, 1, spect.size(0), spect.size(1))
        spect = spect.to(device)

        input_sizes = torch.IntTensor([spect.size(3)]).int()
        out, output_sizes = model(spect, input_sizes)
        out = out.cpu().detach().numpy()[0]

        transcript = ctc_beam_search(out,labels,prune,beam_width,lm,alpha,beta)[0]
        
        with open(reference_path.replace("\n",""),"r") as f:
            reference = f.readline()
        wer_inst = wer_(transcript,reference)
        cer_inst = cer_(transcript, reference)
        total_wer += wer_inst
        total_cer += cer_inst
        num_tokens += len(reference.split(' '))
        num_chars += len(reference.replace(' ', ''))
    
    wer = (float(total_wer) / num_tokens)*100
    cer = (float(total_cer) / num_chars)*100
    output +=f'{alpha}\t{beta}\t{wer}\n'
    with open("/home/hemant/ctc_decoders/abw.json", "w") as f:
        f.write(output)
    print("aplha: ",alpha,"beta: ",beta)
    print('Test Summary \t'
        'Average WER {wer:.3f}\t'
        'Average CER {cer:.3f}\t'.format(wer=wer, cer=cer))
