from backend.utils import *
from pprint import pprint
import random
import stanza


random.seed(0)
pos_tag = stanza.Pipeline(lang='fr', processors='tokenize,mwt,pos', dir='../backend/')

sent_tags = read_data('../resources/pos_tag.txt', custom=True)
sent_raw = read_data('../resources/raw_text.txt', raw=True)
sent_tags = zip(sent_raw, sent_tags)
sent_tags = [x for x in sent_tags if len(x[1]) < 15]
random.shuffle(sent_tags)
sent_tags, sent_tags_test = sent_tags[:1500], sent_tags[1500:1500 + 150]
sent_raw, sent_tags = zip(*sent_tags)
sent_raw, sent_tags = list(sent_raw), list(sent_tags)

sent_raw_test, sent_tags_test = zip(*sent_tags_test)
sent_raw_test, sent_tags_test = list(sent_raw_test), list(sent_tags_test)

rules = grammar_induction(sent_tags, n=-1)
cfg = grammar2cfg(rules)
print(evaluation(cfg, sent_raw_test, pos_tag, sent_tags_test))
