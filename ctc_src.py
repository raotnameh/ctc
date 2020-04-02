import numpy as np
import  kenlm, os, math
from collections import defaultdict
from itertools import groupby
import Levenshtein as Lev

def lse(*args):
    """
    Stable log sum exp.
    """
    if all(a == -float('inf') for a in args):
        return -float('inf')
    a_max = max(args)
#     print('--',args)
#     args = [i for i in args] + [-float('inf')]
    lsp = math.log(sum(math.exp(a - a_max)
                      for a in args))
    return a_max + lsp

def wer_(s1, s2):
    b = set(s1.split() + s2.split())
    word2char = dict(zip(b, range(len(b))))
    w1 = [chr(word2char[w]) for w in s1.split()]
    w2 = [chr(word2char[w]) for w in s2.split()]
    
    return Lev.distance(''.join(w1), ''.join(w2))


# greedy decoding
def ctc_best_path(out,labels):
    "implements best path decoding as shown by Graves"
    out = [labels[i] for i in np.argmax(out, axis=1) if i!=labels[-1]]
    o = ""
    for i,j in groupby(out):
        o = o + i
    return o.replace("_","")

# gred_txt = ctc_best_path(out,labels)
# print(gred_txt)
# wer_(gred_txt,reference)/len(reference.split(' '))*100


#SCORER
class Scorer(object):

    def __init__(self, alpha, beta, model_path,oov_weight=1):
        self.alpha = alpha
        self.beta = beta
        self.oov_weight = oov_weight
        if not os.path.isfile(model_path):
            raise IOError("Invaid language model path: %s" % model_path)
        self.lm = kenlm.LanguageModel(model_path)

    # n-gram language model scoring
    def prob(self, sentence):
        return [10**i[0] for i in self.lm.full_scores(sentence,eos=False)][-1]

    # word insertion term
    def sent_len(self, sentence):
        return len(sentence.strip().split(' '))

    # reset alpha and beta
    def reset_params(self, alpha, beta):
        self.alpha, self.beta = alpha, beta

    # execute evaluation
    def __call__(self, sentence, log=False):
        if self.alpha == 0 and self.beta == 0:
            return 1.0
        lm_score = self.prob(sentence)
        word_insert_score = self.sent_len(sentence)
#         print(sentence, lm_score, word_insert_score)
        if log == False:
            if sentence.strip().split(' ')[-1] not in self.lm:
                    score = (np.power(lm_score, self.alpha) * np.power(word_insert_score, self.beta))*((0.1)**self.oov_weight)
            else: score = np.power(lm_score, self.alpha) * np.power(word_insert_score, self.beta)
        else:
            if sentence.strip().split(' ')[-1] not in self.lm:
                    score = self.alpha * np.log(lm_score) + self.beta * np.log(word_insert_score) - (10)**self.oov_weight
            else: score = self.alpha * np.log(lm_score) + self.beta * np.log(word_insert_score)
        return score

#beam search decoding without log sum exponent
def prefix_bsp(out,labels,scorer,log=False,prune=0.00001,beam_size=25,w_t_o_b=50,
               nproc=False):
    
    blank_symbol = '_'
    F = out.shape[1] # length of labels
    steps = out.shape[0] # number of time steps
    
    t_b = [('', (1.0 ,0.0 ))] # beam at every time step gets updated
    t_1 = None
    if nproc is True:
        print('nproc')
        global ext_nproc_scorer
        scorer = ext_nproc_scorer
    
    for t in range(0,steps):
        pruned_alphabet = [labels[i] for i in np.where(out[t]>prune)[0]]
        dummy_beam = defaultdict(lambda: (0,0))
        dummy = t_b
        for prefix, (pb,pnb) in t_b:
            for c in pruned_alphabet:
                p_t = out[t][labels.index(c)]
                
                if c == blank_symbol:
                    dpb,dpnb = dummy_beam[prefix]
                    dpb += p_t*(pb + pnb)
                    dummy_beam[prefix] = (dpb,dpnb)
                    continue
                
                end_t = prefix[-1] if prefix else None
                c_t = prefix + c
                dpb,dpnb = dummy_beam[c_t]
                if c == end_t and len(prefix) > 0:
                    dpb_,dpnb_ = dummy_beam[prefix]
                    dpnb += p_t*pb
                    dpnb_ += p_t*pnb
                    dummy_beam[prefix] = (dpb_,dpnb_)
                    
                elif c == ' ' and len(prefix.strip().split(' ')) > 1:
                    dpnb += p_t*(pb + pnb)*scorer(prefix)
                
                else:
                    dpnb += p_t*(pb + pnb)
                dummy_beam[c_t] = (dpb,dpnb)

                if beam_size < w_t_o_b:
                    if c_t not in t_b and t_1 != None:
                        dpbn,dpnbn = dummy_beam[c_t]
                        for i in t_1:
                            if i[0] == c_t:
                                b_, nb_  = i[1][0], i[1][1]
                            else:
                                b_, nb_  = 0, 0
                        dpbn  += out[t][labels.index("_")]*(b_ + nb_)
                        dpnbn += p_t*nb_
                        dummy_beam[c_t] = (dpbn,dpnbn)

        t_1 = t_b
        t_b = sorted(dummy_beam.items(),
                      key=lambda x:np.sum(x[1]),
                      reverse=True)
        t_b = t_b[:beam_size]
    
    best = sorted([(scorer(i[0]),i[0]) for i in t_b],reverse=True)[0][1]
    return best

# labels = "_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "
# alpha, beta, lm, oov_weight =1.99, 2, '/home/hemant/4_gram.arpa', 1
# scorer = Scorer(alpha,beta,lm,oov_weight)
# beam_txt = prefix_bsp(out,labels,scorer,log=False,prune=0.00001, beam_size=50,w_t_o_b=10)
# print(beam_txt)
# wer_(beam_txt,reference)/len(reference.strip().split(' '))*100

#beam search decoding with log sum exponent
def prefix_bsl(out,labels,scorer,log=False,prune=0.00001,beam_size=20,w_t_o_b=10):
    
    blank_symbol = '_'
    F = out.shape[1] # length of labels
    steps = out.shape[0] # number of time steps
    prob_ = out
    out = np.log(out)
    NEG_INF = -float("inf")
    
    t_b = [('', (0.0, NEG_INF ))] # beam at every time step gets updated
    t_1 = None
    
    for t in range(0,steps):
        pruned_alphabet = [labels[i] for i in np.where(prob_[t]>prune)[0]]
        dummy_beam = defaultdict(lambda: (NEG_INF, NEG_INF))
        dummy = t_b
        for prefix, (pb,pnb) in t_b:
            for c in pruned_alphabet:
                p_t = out[t][labels.index(c)]
                
                if c == blank_symbol:
                    dpb,dpnb = dummy_beam[prefix]
                    dpb = lse(dpb, p_t+pb, p_t+pnb)
                    dummy_beam[prefix] = (dpb,dpnb)
                    continue
                
                end_t = prefix[-1] if prefix else None
                c_t = prefix + c
                dpb,dpnb = dummy_beam[c_t]
                if c == end_t and len(prefix) > 0:
                    dpb_,dpnb_ = dummy_beam[prefix]
                    dpnb = lse(dpnb,p_t+pb)
                    dpnb_ = lse(dpnb_,p_t+pnb)
                    dummy_beam[prefix] = (dpb_,dpnb_)
                    
                elif c == ' ' and len(prefix.strip().split(' ')) > 1:
                    score = scorer(prefix,log=True)
                    if score == 1.0: dpnb = lse(dpnb,p_t+pb, p_t+pnb)
                    else: 
                        dpnb = lse(dpnb,p_t+pb, p_t+pnb, score)

                else:
                    dpnb = lse(dpnb, p_t+pb, p_t+ pnb)
                dummy_beam[c_t] = (dpb,dpnb)
                
                if w_t_o_b < 50:
                    if c_t not in t_b and t_1 != None:
                        dpbn,dpnbn = dummy_beam[c_t]
                        for i in t_1:
                            if i[0] == c_t:
                                b_, nb_  = i[1][0], i[1][1]
                            else:
                                b_, nb_  = NEG_INF, NEG_INF
                        dpbn  = lse(dpbn,out[t][labels.index("_")]+b_, out[t][labels.index("_")]+ nb_)
                        dpnbn = lse(dpnbn, p_t+nb_)
                        dummy_beam[c_t] = (dpbn,dpnbn)

        t_1 = t_b
        t_b = sorted(dummy_beam.items(),
                      key=lambda x:lse(*x[1]),
                      reverse=True)
        t_b = t_b[:beam_size]
        

    best = sorted([(scorer(i[0]),i[0]) for i in t_b],reverse=True)[0][1]
    return best

# alpha, beta, lm, oov_weight =1.96,3, '/home/hemant/asr_wm/lm/lm.binary', 1
# scorer = Scorer(alpha,beta,lm,oov_weight)
# beam_txt = prefix_bsl(out,labels,scorer,log=True,prune=0.00001, beam_size=10,w_t_o_b=100)
# print(beam_txt)
# wer_(beam_txt,reference)/len(reference.strip().split(' '))*100