import re
from io import StringIO

import lark
import nltk
from lark import Token
from pptree import Node

from backend.utils import grammar_induction, grammar2cfg, grammar_cfg_string_to_lark


def tag_text(text, pos_tagger):
    sentences = nltk.sent_tokenize(text, language="french")
    sentences = [[x.upos for x in pos_tagger(s).iter_words()] for s in sentences]
    return sentences


def get_grammar_tuples(sentences):
    """sentences : output of tag_text"""
    return grammar_induction(sentences, n=2)


def get_grammar_cfg_string(grammar_tuples):
    return grammar2cfg(grammar_tuples)


def get_grammar_cfg(grammar_cfg_string):
    return nltk.CFG.fromstring(grammar_cfg_string)


def get_parser(grammar_cfg_string):
    grammar_lark = grammar_cfg_string_to_lark(grammar_cfg_string)
    parser = lark.Lark(grammar_lark, start='s', lexer="dynamic_complete")
    return parser


def pprint_tree(node, file=None, _prefix="", _last=True):
    print(_prefix, "`- " if _last else "|- ",
          node.name if len(node.children) else f'<span style="color: red;">{node.name}</span>', sep="",
          file=file)
    _prefix += "   " if _last else "|  "
    child_count = len(node.children)
    for i, child in enumerate(node.children):
        _last = i == (child_count - 1)
        pprint_tree(child, file, _prefix, _last)


def parse(parser, sent):
    def to_pptree(tree, parent):
        if isinstance(tree, Token):
            Node(str(tree), parent)
            return
        root = Node(tree.data, parent)
        for x in tree.children:
            to_pptree(x, root)
        return root

    try:
        tree = parser.parse(' '.join(sent))
        result = StringIO()
        pprint_tree(to_pptree(tree, None), file=result)
        result = result.getvalue()
        result = re.sub(r'nt([0-9]+)', r'NT\1', result)
        result = re.sub(r's(?!\w)', r'S', result)
        return result
    except Exception:
        return ''


def get_infos_grammar(grammar_cfg_string: str):
    infos = {}
    infos['number_of_rules'] = grammar_cfg_string.count('->')
    infos['number_of_non_terminals'] = len(set(re.findall(r'NT\d+', grammar_cfg_string))) + 1
    infos['number_of_terminals'] = len(set(re.findall(r'".+?"', grammar_cfg_string)))
    infos['number_of_starting_nodes'] = re.findall(r'S -> .+', grammar_cfg_string)[0].split('->').count('|') + 1
    return infos
