from backend.utils import *
from pprint import pprint
import random
import stanza

pos_tag = stanza.Pipeline(lang='fr', processors='tokenize,mwt,pos', dir='../backend/')

sent_tags = read_data('../resources/pos_tag.txt', custom=True)
sent_raw = read_data('../resources/raw_text.txt', raw=True)
sent_tags = zip(sent_raw, sent_tags)
sent_tags = [x for x in sent_tags if len(x[1]) < 10]
random.shuffle(sent_tags)
sent_tags = sent_tags[:1500]
sent_raw, sent_tags = zip(*sent_tags)
sent_raw, sent_tags = list(sent_raw), list(sent_tags)

rules = grammar_induction(sent_tags, n=2)
cfg = grammar2cfg(rules)
print(evaluation(cfg, sent_raw, pos_tag, sent_tags))
