#!/usr/bin/python
# -*- coding: iso-8859-15 -*-
import sys
import re
from pynlp.de.gwn_old import germanet

adverb_classes={}
# nicht: erst, gerade (Fokusinteraktion)
# nicht: zugleich (parallel)
for k in '''anfangs bald beizeiten früh
nun jetzt gerade derzeit inzwischen soeben
gestern vorgestern neulich bisher bislang früher damals kürzlich letztens seinerzeit
morgen übermorgen seither später nachher demnächst'''.split():
    adverb_classes[k]='tmp'
for k in '''daher darum deshalb deswegen'''.split():
    adverb_classes[k]='causal'
for k in '''auch ebenfalls ebenso gleichfalls'''.split():
    adverb_classes[k]='focus-koord-incl'
for k in '''nur bloß lediglich allein ausschließlich einzig'''.split():
    adverb_classes[k]='focus-restr-excl'
for k in '''erst schon bereits noch'''.split():
    adverb_classes[k]='focus-tmp'
for k in '''zweifellos zweifelsohne fraglos tatsächlich
sicher bestimmt gewiß vermutlich wahrscheinlich
angeblich vorgeblich
leider'''.split():
    adverb_classes[k]='comment'

def classify_adverb(n):
    lem=n.head.lemma
    if lem in adverb_classes:
        return adverb_classes[lem]
    if lem.endswith('weise'):
        return 'comment'
    return None

def classify_px(n):
    if n.children[0].edge_label=='APP' and n.children[0].cat=='PX':
        n=n.children[0]
    lem=n.head.lemma
    if n.children[-1].cat not in ['NX','NCX']:
        return None
    case=n.head.morph[0]
    lem2=n.children[-1].head.lemma
    cls_lem2=set()
    for syn in germanet.synsets_for_word(lem2):
        cls_lem2.update(germanet.classify_synset(syn))
    if lem=='wegen':
        return 'causal'
    elif lem=='trotz':
        return 'concessive'
    if case=='d':
        if lem=='in' and 'zeiteinheit' in cls_lem2:
            return 'tmp'
        elif lem=='an' and 'tag' in cls_lem2:
            return 'tmp'
        elif lem=='nach' and 'information' in cls_lem2:
            return 'source'
        elif lem=='nach' and 'ereignis' in cls_lem2:
            return 'tmp'
        elif lem=='vor':
            if 'ereignis' in cls_lem2:
                # vor dem Unfall
                return 'tmp'
            if 'zeiteinheit' in cls_lem2:
                # vor zwei Wochen, vor dem 1. Mai
                return 'tmp'
        elif lem in ['während','seit']:
            return 'tmp'



def get_productions(n,exclude,lst):
    if n.isTerminal():
        lst.append('%s=%s'%(n.cat,n.word))
    else:
        lst.append('%s=%s'%(n.cat,'-'.join([n1.cat for n1 in n.children])))
        for n1 in n.children:
            if n1 not in exclude:
                get_productions(n1,exclude,lst)

