import numpy as np
from collections import Counter
import copy

def read_data(path):
    with open(path, 'r', encoding="utf-8") as f:
        sent_tags = f.readlines()
    sent_tags = [[y.split('_')[1] for y in x.strip('\n').split(' ')] for x in sent_tags]
    return sent_tags


def grammar_induction(sent_tags : list, n=3):
    sent_tags = copy.deepcopy(sent_tags)
    rules = []
    num_rules = 1
    while True:
        # N-gram extraction
        l = []
        for i in range(2, n+1):
            for s in sent_tags:
                l += [tuple(s[j:j+i]) for j in range(len(s) - i)]
        counter = Counter(l)

        # Most common N-gram
        ngram, _ = counter.most_common()

        # Adding new rule
        rules.append(('NT' + str(num_rules), ngram))
        num_rules += 1

        # Substitution
        non_terminal, other = rules[-1]
        other = list(other)
        for s in sent_tags:
            i = 0
            while i < len(s) - len(other):
                if other == s[i:i+len(other)]:
                    i += len(other) - 1
                    s = s[i:]
                    s[i] = non_terminal
                i += 1

    return rules




sent_tags = read_data('../resources/sent_tags.stanford-pos')
print(np.array([len(x) for x in sent_tags]).mean())