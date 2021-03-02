import copy
import os
from collections import Counter

import nltk
from nltk import FreqDist


def create_dataset(path, pos_tag, num_sents=None):
    with open(path, 'r', encoding="utf-8") as f:
        sentences = f.readlines()
    # random.shuffle(sentences)
    if num_sents is not None:
        sentences = sentences[:num_sents]

    def f(s):
        f.i = f.i + 1
        if f.i % 100 == 0:
            print(str(f.i), '/', len(sentences))
        return pos_tag(s)

    f.i = 0
    sentences = [[x.upos for x in f(s).iter_words()] for s in sentences]
    with open(os.path.join(os.path.split(path)[0], 'pos_tag.txt'), 'w', encoding="utf-8") as f:
        for s in sentences:
            f.write(' '.join(s) + '\n')


def grammar2cfg(rules):
    s = ''
    for rule in rules:
        nt, other = rule
        s += nt + ' -> '
        for x in other:
            if isinstance(x, tuple):
                s += ' | '.join(x)
            elif x.startswith('NT'):
                s += x + ' '
            else:
                s += '"' + x + '" '
        s += '\n'
    return s


def evaluation(cfg_string, sentences, pos_tag, sent_tags=None):
    if sent_tags is None:
        sent_tags = [pos_tag(s) for s in sentences]
        sent_tags = [[x.upos for x in s.iter_words()] for s in sent_tags]
    sentences = [nltk.word_tokenize(s) for s in sentences]
    w = weights(sentences)
    cfg = nltk.CFG.fromstring(cfg_string)
    parser = nltk.RecursiveDescentParser(cfg)
    rf_scores = [rf(sent, parser) for sent in sent_tags]
    precision = sum(w[i] * rf_scores[i] for i in range(len(w))) / sum(w)
    return precision


def rf(sent, parser):
    i = len(sent)
    found = False
    while i > 0:
        for ngram in nltk.ngrams(sent, i):
            try:
                if len(list(parser.parse(ngram))) > 0:
                    found = True
                    break
            except (ValueError, AttributeError):
                pass
        if found:
            break
        i -= 1
    print(i / len(sent))
    return i / len(sent)


def read_data(path, custom=False, raw=False):
    with open(path, 'r', encoding="utf-8") as f:
        sent_tags = f.readlines()
    if raw:
        return [x.strip('\n') for x in sent_tags]

    if custom:
        sent_tags = [x.strip('\n').split(' ') for x in sent_tags]
    else:
        sent_tags = [[y[y.rindex('_') + 1:] for y in x.strip('\n').split(' ')] for x in sent_tags]
    return sent_tags


def weights(sentence):
    l = []
    for s in sentence:
        l += [tuple(s[j:j + 2]) for j in range(len(s) - 1)]
    f = FreqDist(l)
    weights = []
    for s in sentence:
        prod = 1
        for bg in nltk.bigrams(s):
            prod *= f.freq(bg)
        weights.append(prod)
    return weights


def grammar_induction(sent_tags: list, n=-1):
    sent_tags = copy.deepcopy(sent_tags)
    rules = []
    num_rules = 1
    while True:
        print(sum(len(x) for x in sent_tags))
        # N-gram extraction
        l = []
        for s in sent_tags:
            for i in range(2, n + 1 if n > 0 else len(s)):
                l += [tuple(s[j:j + i]) for j in range(len(s) - i + 1) if
                      any(not x.startswith('NT') for x in s[j:j + i])]
        counter = Counter(l)

        if len(counter) == 0:
            break

        # Most common N-gram
        c = counter.most_common()
        ngram, m = c[0]
        for x, y in c:
            if y < m:
                break
            ngram = x

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
                if other == s[i:i + len(other)]:
                    j = i + len(other) - 1
                    sent_tags[k] = s[:i] + s[j:]
                    s = sent_tags[k]
                    s[j - 1] = non_terminal
                    i = j
                else:
                    i += 1

    # Adding root rule
    for x in sent_tags:
        rules.append(('S', tuple(x)))
    return rules[::-1]
