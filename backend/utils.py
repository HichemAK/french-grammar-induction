from collections import Counter
import copy
import nltk

def read_data(path):
    with open(path, 'r', encoding="utf-8") as f:
        sent_tags = f.readlines()
    sent_tags = [[y[y.rindex('_')+1:] for y in x.strip('\n').split(' ')] for x in sent_tags]
    return sent_tags


def grammar_induction(sent_tags : list, n=-1):
    sent_tags = copy.deepcopy(sent_tags)
    rules = []
    num_rules = 1
    while True:
        print(sum(len(x) for x in sent_tags))
        # N-gram extraction
        l = []
        for s in sent_tags:
            for i in range(2, n + 1 if n > 0 else len(s)):
                l += [tuple(s[j:j+i]) for j in range(len(s) - i + 1)]
        counter = Counter(l)

        if len(counter) == 0:
            break

        # Most common N-gram
        ngram, _ = counter.most_common(1)[0]

        # Adding new rule
        rules.append(('NT' + str(num_rules), ngram))
        num_rules += 1

        # Substitution
        non_terminal, other = rules[-1]
        other = list(other)
        for k in range(len(sent_tags)):
            s = sent_tags[k]
            i = 0
            while i <= len(s) - len(other):
                if other == s[i:i+len(other)]:
                    j = i + len(other) - 1
                    sent_tags[k] = s[:i] + s[j:]
                    s = sent_tags[k]
                    s[j-1] = non_terminal
                    i = j
                else:
                    i += 1

    # Adding root rule
    rules.append(('S', tuple(x[0] for x in sent_tags)))
    return rules
