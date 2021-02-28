from backend.utils import *
from pprint import pprint
import random

sent_tags = read_data('../resources/sent_tags.stanford-pos')
random.shuffle(sent_tags)
sent_tags = sent_tags[:2000]

l = []
for s in sent_tags:
    l += s
pprint(set(l))

grammar_induction(sent_tags, n=2)
