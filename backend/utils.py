import copy
import os
from collections import Counter

import nltk
from nltk import FreqDist
import numpy as np

import re
import lark


def grammar_cfg_string_to_lark(cfg_string):
    s = ""
    terms = set(re.findall(r'("(.+?)")', cfg_string))
    print(terms)
    for x, y in terms:
        s += y + " : " + x + '\n'

    cfg_string = cfg_string.replace('S ->', 's ->')
    cfg_string = re.sub(r'->', ':', cfg_string)
    cfg_string = re.sub(r'NT(\d+)', r'nt\1', cfg_string)
    cfg_string = re.sub(r'"(.+?)"', r'\1', cfg_string)
    cfg_string = re.sub(r'(?<=[\w"]) (?=["\w])', r' " " ', cfg_string)

    return s + cfg_string


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
    """rules : output of grammar_induction"""
    s = ''
    for rule in rules:
        nt, other = rule
        s += nt + ' -> '
        if nt == 'S':
            s += ' | '.join(' '.join(x) for x in other) + '\n'
            continue
        for x in other:
            if x.startswith('NT'):
                s += x + ' '
            else:
                s += '"' + x + '" '
        s += '\n'
    return s


def evaluation(cfg_string, sentences, pos_tag, sent_tags=None):
    """cfg_string : output of grammar2cfg
    sentences : list of Strings
    pos_tag : stanza model
    sent_tags (Optional) : list of tagged sentences in 'sentences' (if you already have them)"""
    if sent_tags is None:
        sent_tags = [pos_tag(s) for s in sentences]
        sent_tags = [[x.upos for x in s.iter_words()] for s in sent_tags]
    sentences = [nltk.word_tokenize(s) for s in sentences]
    w = weights(sentences)
    cfg_string_lark = grammar_cfg_string_to_lark(cfg_string)
    parser = lark.Lark(cfg_string_lark, start='s', lexer="dynamic_complete")
    # tree = parser.parse(' '.join(sent_tags[0]))
    # lark.tree.pydot__tree_to_png(tree, "example.png")
    rf_scores = [rf_fast(sent, parser) for sent in sent_tags]
    print(len(list(filter(lambda x : x == 1, rf_scores))))
    precision = sum(w[i] * rf_scores[i] for i in range(len(w))) / sum(w)
    std = np.array(rf_scores).std()
    return precision, std

def load_grammar(path):
    with open(path, 'r') as f:
        raw = f.read()

    cfg = nltk.CFG.fromstring(raw)
    parser = nltk.RecursiveDescentParser(raw)
    return raw, cfg, parser

def parse(sent, pos_tag, grammar):
    """sent : String
    pos_tag : stanza model
    grammar : output of load_grammar"""
    sent = [x.upos for x in pos_tag(sent).iter_words()]
    tree = list(grammar.parse(sent))
    return tree

def save_grammar(path, grammar):
    """grammar : the output of grammar_induction"""
    with open(path, 'w') as f:
        f.write(grammar2cfg(grammar))


def rf(sent, parser):
    i = len(sent)
    found = False
    while i > 0:
        for ngram in nltk.ngrams(sent, i):
            try:
                parser.parse(' '.join(ngram))
                found = True
                break
            except (lark.UnexpectedCharacters, lark.UnexpectedEOF):
                pass
        if found:
            break
        i -= 1
    print(i / len(sent))
    return i / len(sent)

def rf_fast(sent, parser):
    i = 0
    for i in range(len(sent), -1, -1):
        try:
            parser.parse(' '.join(sent[:i]))
            break
        except (lark.UnexpectedCharacters, lark.UnexpectedEOF, lark.UnexpectedToken, lark.ParseError):
            pass
    print(i / len(sent))
    return i / len(sent)



def read_data(path, custom=False, raw=False):
    """See test.py to see how to load data"""
    with open(path, 'r', encoding="utf-8") as f:
        data = f.readlines()
    if raw:
        return [x.strip('\n') for x in data]

    if custom:
        data = [x.strip('\n').split(' ') for x in data]
    else:
        data = [[y[y.rindex('_') + 1:] for y in x.strip('\n').split(' ')] for x in data]
    return data


def weights(sentences):
    l = []
    for s in sentences:
        l += [tuple(s[j:j + 2]) for j in range(len(s) - 1)]
    f_bigram = FreqDist(l)
    l = [y for x in sentences for y in x]
    f_unigram = FreqDist(l)
    weights = []
    for s in sentences:
        s = tuple(s)
        prod = f_unigram.freq(s[0])
        for i in range(1, len(s)):
            prod *= f_bigram.freq(s[i-1:i+1])/f_unigram.freq(s[i])
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
            for i in range(2, (n + 1 if n > 0 else len(s))):
                l += [tuple(s[j:j + i]) for j in range(len(s) - i + 1) if any(not x.startswith('NT') for x in s[j:j + i])]
        counter = Counter(l)

        if len(counter) == 0:
            break

        # Most common N-gram
        # c = counter.most_common()
        # ngram, m = c[0]
        # for x, y in c:
        #     if y < m:
        #         break
        #     ngram = x
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
                if other == s[i:i + len(other)]:
                    j = i + len(other) - 1
                    sent_tags[k][j] = non_terminal
                    sent_tags[k] = s[:i] + s[j:]
                    s = sent_tags[k]
                    i = j
                else:
                    i += 1

    sent_tags = list(set(tuple(x) for x in sent_tags))
    sent_tags.sort(key=lambda x: len(x))

    # Remove duplicates
    to_remove = set()
    for i in range(len(sent_tags)):
        for j in range(i + 1, len(sent_tags)):
            r1, r2 = sent_tags[i], sent_tags[j]
            if r1 == r2[:len(r1)]:
                to_remove.add(r1)
    sent_tags = set(sent_tags)
    sent_tags = sent_tags.difference(to_remove)

    # Creating the source non-terminal S
    rules.append(('S', tuple(sent_tags)))
    return rules[::-1]
