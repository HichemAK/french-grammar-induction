import random

import stanza

from backend.utils import *
from pprint import pprint

random.seed(100)
pos_tag = stanza.Pipeline(lang='fr', processors='tokenize,mwt,pos', dir='../backend/')

sent_tags = read_data('../resources/pos_tag.txt', custom=True)
sent_raw = read_data('../resources/raw_text.txt', raw=True)

sent_tags = zip(sent_raw, sent_tags)
sent_tags = [x for x in sent_tags if 3 <= len(x[1]) <= 10]
random.shuffle(sent_tags)
print(len(sent_tags))
sent_tags, sent_tags_test = sent_tags[:1], sent_tags[1500:1500 + 200]
print(sent_tags)

sent_raw, sent_tags = zip(*sent_tags)
sent_raw, sent_tags = list(sent_raw), list(sent_tags)

freq = compute_f_unibi(sent_tags)

sent_raw_test, sent_tags_test = zip(*sent_tags_test)
sent_raw_test, sent_tags_test = list(sent_raw_test), list(sent_tags_test)
print(sent_raw_test[0])
print(sent_tags_test[0])

# test = ["Je suis Hichem."]
# test = [[str(x) for x in y] for y in test]
x = None
for x in grammar_induction(sent_tags, n=2, yield_infos=True):
    if isinstance(x, dict):
        print(x['progression'])
    else:
        break
rules = x
print(len(rules) - len(list(filter(lambda x : x[0] == 'S', rules))))
pprint(rules[-20:])
cfg = grammar2cfg(rules)
with open('grammar', 'w') as f:
    f.write(cfg)

for x in evaluation(cfg, sent_tags_test, freq, yield_infos=True):
    if isinstance(x, dict):
        print(x)
    else:
        break
