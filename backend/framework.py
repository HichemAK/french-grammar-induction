from lark import Token

from backend.utils import *
import nltk
from pptree import Node

def tag_text(text, pos_tagger):
    sentences = nltk.sent_tokenize(text)
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
        return to_pptree(tree, None)
    except (lark.UnexpectedToken, lark.UnexpectedEOF, lark.UnexpectedCharacters, lark.UnexpectedInput):
        return ''
