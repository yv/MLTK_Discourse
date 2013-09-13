#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
import sys
import math
from pkg_resources import resource_stream
from itertools import izip
from collections import defaultdict
import pytree.export as export
from dist_sim.fcomb import InfoNode, InfoTree
from pytree import deps
import simplejson as json
import optparse

from pynlp.de import smor_pos, tueba_heads, pydeps
from pynlp.de.gwn_old import germanet, wordsenses
from pynlp.de.gwn_old.semclass import semclass_for_node
from pynlp.de.gwn_old.gwn_word_features import get_verb_features

from mltk_discourse.sem_features import classify_adverb, classify_px
from mltk_discourse.tree_util import tree_yield, get_productions


stupid_head_finder=deps.SimpleDepExtractor(tueba_heads.hr_table+[(None,[(None,'HD','r'),(None,'l')])],['$,','$.'])

def null_warning_handler(w,args):
    if w in ['nolabel','nohead']:
        return
    sys.stderr.write(deps.messages[w]%args)

deps.warning_handler=null_warning_handler

def make_hier_schema(lines):
    prefixes=['']
    h_map={}
    for l in lines:
        l=l.rstrip()
        if l[0]=='%':
            continue
        if ' ' in l:
            l=l[:l.index(' ')]
        level=1
        while l[0]=='+':
            l=l[1:]
            level+=1
        del prefixes[level:]
        h_map[l]=prefixes[-1]+l
        prefixes.append('%s%s.'%(prefixes[-1],l))
    return h_map

hier_map=make_hier_schema(resource_stream(__name__,'konn2_schema.txt'))

def find_args(n):
    if n.cat=='KOUS':
        if n.parent.cat=='C':
            pp=n.parent.parent
            assert pp.cat in ['SIMPX','FKONJ']
            sub_cl=pp
            while pp.edge_label=='KONJ':
                pp=pp.parent
            if pp.parent:
                pp=pp.parent
                while pp.parent and pp.cat not in ['SIMPX','R-SIMPX','FKONJ']:
                    pp=pp.parent
                main_cl=pp
            else:
                main_cl=None
            return sub_cl,main_cl
        else:
            print "weird: %s"%(n,)
            return None,None
    elif n.cat=='KON':
        pp=n.parent
        seen_kon=False
        main_cl=None
        sub_cl=None
        for n1 in pp.children:
            if n1.cat in ['SIMPX','FKONJ']:
                if seen_kon:
                    sub_cl=n1
                else:
                    main_cl=n1
            if n1.cat=='KON':
                seen_kon=True
        if main_cl is not None and main_cl.cat=='FKONJ':
            main_cl=pp.parent
        return sub_cl,main_cl
    else:
        print >>sys.stderr, "Unknown connective type: %s/%s"%(n.word,n.cat)
        return None,None

def find_negation(n):
    result=[]
    gather_negation(n.children,result)
    return result

def gather_negation(nodes,result):
    for n in nodes:
        if n.cat=='PTKNEG':
            result.append(n)
        elif n.isTerminal() and n.word.startswith('kein'):
            result.append(n)
        elif n.isTerminal() and n.word == 'nichts':
            result.append(n)
        if n.cat in ['SIMPX','R-SIMPX'] and n.edge_label!='OS':
            pass
        elif n.cat=='FKOORD':
            for n1 in n.children:
                if n1.cat=='FKONJ':
                    gather_negation(n1.children,result)
                    break
        else:
            gather_negation(n.children,result)

def find_nomargs(n):
    result=[]
    gather_args(n.children,result)
    return result

def gather_args(nodes,result):
    for n in nodes:
        if n.cat in ['VF','MF']:
            for n1 in n.children:
                if n1.edge_label in ['ON','OA','OD','OG','OPP','FOPP','PRED']:
                    result.append((n1.edge_label,n1))
        elif n.cat=='FKOORD':
            for n1 in n.children:
                if n1.cat=='FKONJ':
                    gather_args(n1.children,result)
                    break

def gather_adjuncts(nodes,result,exclude):
    for n in nodes:
        if n.cat in ['VF','MF','NF']:
            for n1 in n.children:
                if n1 in exclude: continue
                if n1.cat in ['NX','NCX']:
                    if n1.edge_label=='MOD' and n.cat!='NF':
                        result['tmp'].append(n1)
                elif n1.cat=='PX':
                    result[classify_px(n1)].append(n1)
                elif n1.cat=='ADVX':
                    result[classify_adverb(n1)].append(n1)
        elif n.cat=='FKOORD':
            for n1 in n.children:
                if n1.cat=='FKONJ':
                    gather_adjuncts(n1.children,result,exclude)
                    break
def find_adjuncts(n,exclude=[]):
    result=defaultdict(list)
    gather_adjuncts(n.children,result,exclude)
    return result

def enumerate_field_nodes(node):
    for n in node.children:
        if n.cat in ['LK','VC','C','VF','MF','NF']:
            yield n
        elif n.cat in ['SIMPX', 'R-SIMPX'] and n.edge_label in ['KONJ','HD']:
            for n1 in enumerate_field_nodes(n):
                yield n1
                return            
        elif n.cat in ['FKOORD','FKONJ']:
            for n1 in enumerate_field_nodes(n):
                yield n1
                return

def get_clause_type(node):
    if node.cat=='R-SIMPX':
        return 'REL'
    elif node.cat not in ['SIMPX','FKOORD','FKONJ']:
        return node.cat
    # LK - possibly in FKOORD
    # VC - possibly in FKOORD, possibly in VF ( "Einen Kuchen gebacken hat Peter nicht." )
    fd_lk=None
    fd_vc=None
    fd_c=None
    fd_vf=None
    fd_mf=None
    for n in enumerate_field_nodes(node):
        if n.cat=='LK':
            fd_lk=n
        elif n.cat=='VC':
            fd_vc=n
        elif n.cat=='C':
            fd_c=n
        elif n.cat=='VF':
            fd_vf=n
        elif n.cat=='MF':
            fd_mf=n
    if fd_lk:
        if fd_vf or node.edge_label=='KONJ':
            return "V2"
        else:
            return "V1"
    elif fd_vf:
        return "V2"
    elif fd_c:
        if len(fd_c.children)==2 and (fd_c.children[0].word.lower()=='so' and
                                      fd_c.children[1].word.lower() in ['dass','daß']):
            return 'KOUS'
        assert len(fd_c.children)==1, fd_c.children
        n1=fd_c.children[-1]
        if n1.cat == 'KOUI':
            return 'KOUI'
        elif n1.cat in ['ADVX','ADJX','PX','NX']:
            return 'W'
        elif n1.cat == 'KOUS':
            if n1.word.lower() in ['dass','daß']:
                return 'dass'
            elif n1.word.lower() == 'ob':
                return 'ob'
            else:
                return 'KOUS'
        else:
            assert False, n1.to_penn()
    elif fd_vc:
        vx_izu=None
        vx_fin=None
        last_cat=None
        for n in fd_vc.children:
            if n.cat == 'VXFIN':
                vx_fin=n.children[0]
            elif n.cat == 'VXINF':
                if len(n.children)==2 and n.children[0].cat=='PTKZU':
                    vx_izu=n.children[1]
                    last_cat='IZU'
                    continue
                elif len(n.children)>1:
                    konjs=[n1 for n1 in n.children if n1.edge_label=='KONJ']
                    if konjs:
                        assert len(konjs[0].children)==1
                    else:
                        assert False, n.to_penn()
                    n1=konjs[0]
                else:
                    n1=n.children[0]
                if n1.cat=='VVIZU':
                    vx_izu=n1
                    last_cat='IZU'
                else:
                    last_cat=n1.cat[2:]
            elif n.cat == 'PTKVZ' and last_cat is None:
                last_cat='PTKVZ'
        if vx_fin:
            return "Vfin"
        elif vx_izu:
            return "Vizu"
        elif last_cat=='PP':
            return "Vpp"
        elif last_cat=='INF':
            return "Vinf"
        elif last_cat=='PTKVZ':
            if node.edge_label=='KONJ':
                return 'V2'
            else:
                assert False, '%s: %s'%(last_cat, n.to_penn())
        else:
            assert False, '%s: %s'%(last_cat, n.to_penn())
    elif fd_mf:
        return 'null'
    else:
        assert False, node.to_penn()

def add_hypernyms(synsets,result):
    for syn in synsets:
        hyper=syn.getHypernyms()
        if not hyper:
            result.append(syn.getWords()[0].word)
        else:
            add_hypernyms(hyper,result)

def strip_aux(s):
    if s.endswith('%aux'):
        return s[:-4]
    if s.endswith('%passiv'):
        return s[:-7]
    return s
def get_verbs(n):
    fin_verbs=[]
    nfin_verbs=[]
    gather_verbs(n.children,fin_verbs,nfin_verbs)
    flags=set()
    if not fin_verbs:
        for n1 in n.children:
            if n1.cat=='FKOORD':
                print >>sys.stderr, "looking for finite verb in FKOORD..."
                for n2 in n1.children:
                    if n2.cat=='FKONJ':
                        gather_verbs(n2.children,fin_verbs,[])
                        if fin_verbs: break
    if not fin_verbs:
        if nfin_verbs:
            main_v=nfin_verbs[0]
            all_v=nfin_verbs[1:]
            flags.add('nonfin')
        else:
            main_v=n.head
            all_v=[]
            flags.add('null')
    else:
        main_v=fin_verbs[0]
        all_v=nfin_verbs
        if len(main_v.morph)==4:
            flags.add('tense=%s'%(main_v.morph[3]))
            flags.add('mood=%s'%(main_v.morph[2]))
        else:
            print >>sys.stderr, "strange morph: %s for %s"%(main_v.morph,main_v)
    print main_v,all_v
    pred=strip_aux(main_v.lemma)
    for n in all_v:
        if n.cat.endswith('PP'):
            if pred in 'werden':
                flags.add('passive:dyn')
                pred=strip_aux(n.lemma)
            elif pred=='haben':
                flags.add('perfect')
                pred=strip_aux(n.lemma)
            elif pred=='sein':
                lem=strip_aux(n.lemma)
                if (lem not in smor_pos.info_map or
                    'perfect:sein' in smor_pos.info_map[lem]):
                    flags.add('perfect')
                else:
                    flags.add('passive:stat')
                pred=strip_aux(n.lemma)
            elif pred=='bleiben':
                flags.add('passive:stat')
                # bleiben markieren
                pred=strip_aux(n.lemma)
        elif n.cat.endswith('INF'):
            if pred=='werden':
                flags.add('future')
                pred=strip_aux(n.lemma)
            elif pred in ['wollen','sollen','müssen','können']:
                flags.add('mod:'+pred)
                pred=strip_aux(n.lemma)
        elif n.cat=='PTKVZ' and '#' not in pred:
            pred='%s#%s'%(n.word,pred)
    return (pred,flags)

def extend_v(lst,nodes):
    for n in nodes:
        if n.isTerminal() and n.cat[0]=='V':
            lst.append(n)
        elif n.cat[0]!='V':
            continue
        else:
            for n1 in n.children:
                if n1.edge_label=='KONJ':
                    if n1.isTerminal():
                        lst.append(n1)
                    else:
                        extend_v(lst,n1.children)
                    break
    print lst,nodes

def gather_verbs_fin(chlds,fin_v):
    for n1 in chlds:
        if (n1.cat[:2] in ['VV','VA','VM'] and
            n1.cat[2:] in ['FIN','IMP']):
            fin_v.append(n1)
        elif n1.cat=='VXFIN':
            gather_verbs_fin(n1.children,fin_v)
            if n1.edge_label=='KONJ':
                break
        elif n1.cat in ['TRUNC','KON']:
            continue
        else:
            assert False,n1
def gather_verbs_nfin(chlds,nfin_v):
    for n1 in chlds:
        if n1.isTerminal():
            if n1.cat[0]=='V':
                nfin_v.append(n1)
        elif n1.cat=='VXINF':
            gather_verbs_nfin(n1.children,nfin_v)
            if n1.edge_label=='KONJ':
                break
        elif n1.cat=='ADVX':
            continue
        else:
            assert False, n1
def gather_verbs(nodes,fin_v,nfin_v):
    for n in nodes:
        if n.cat in ['LK','VC']:
            for n1 in n.children:
                if n1.cat=='VXFIN':
                    gather_verbs_fin(n1.children,fin_v)
                elif n1.cat=='VXINF':
                    gather_verbs_nfin(n1.children,nfin_v)
                elif n1.cat=='PTKVZ':
                    nfin_v.append(n1)
                else:
                    print "unclear:",n1.cat
        elif n.cat=='FKOORD':
            for n1 in n.children:
                if n1.cat=='FKONJ':
                    gather_verbs(n1.children,fin_v,nfin_v)
                    break
                elif n1.cat in ['LK','VC']:
                    for n2 in n1.children:
                        if n2.cat=='VXFIN':
                            gather_verbs_fin(n2.children,fin_v)
                        elif n2.cat=='VXINF':
                            gather_verbs_nfin(n2.children,nfin_v)
                        elif n2.cat=='PTKVZ':
                            nfin_v.append(n2)
                        else:
                            print "unclear:",n1.cat
                    break


def get_target(anno):
    rel1=anno.rel1
    tgt=[hier_map[rel1]]
    if hasattr(anno,'rel2') and anno.rel2 not in ['NULL','',None]:
        tgt.append(hier_map[anno.rel2])
    return tgt

def grok_encoding(s):
    if isinstance(s,str):
        return s.decode('ISO-8859-15')
    else:
        return unicode(s)

def compatible_pronoun(n1,n2):
    if n1.head.cat!='PPER':
        return False
    if n2.head.cat=='PPER':
        return n1.head.morph[1:]==n2.head.morph[1:]
    if n1.head.morph[3]!='3':
        return False
    if n1.head.morph[1:3]!=n2.head.morph[1:3]:
        return False
    for n in n2.children:
        if n.cat=='PPOSAT':
            return False
    return True


def get_features(t, sub_cl, main_cl):
    feats=[]
    sub_parent=sub_cl.parent
    # if sub_parent.cat=='ADVX' and sub_parent.parent:
    #     #feats.append('modADV')
    #     cls=classify_adverb(sub_parent)
    #     #if cls:
    #     #    feats.append('modADV='+cls)
    #     #feats.append('modADV='+sub_parent.head.lemma)
    if (sub_parent.cat in ['ADVX','PX','NX'] and
        sub_parent.parent and
        sub_parent.parent.cat in ['VF','MF','NF']):
        sub_parent=sub_parent.parent
    print "field:",sub_parent.cat
    if 'nobaseline' not in wanted_features:
        feats.append('fd'+sub_parent.cat)
    neg_sub=find_negation(sub_cl)
    print "neg[sub]: ",neg_sub
    if 'neg' in wanted_features:
        if neg_sub:
            feats.append('NS+')
        else:
            feats.append('NS-')
    neg_main=find_negation(main_cl)
    print "neg[main]:",neg_main
    if 'neg' in wanted_features:
        if neg_main:
            feats.append('NM+')
        else:
            feats.append('NM-')
    (p,flags)=get_verbs(sub_cl)
    if 'lex' in wanted_features:
        feats.append('LS'+p)
    if 'assoc' in wanted_features and p in assoc_features:
        for k in assoc_features[p]:
            feats.append('AS'+k)
    if 'tmp' in wanted_features:
        for k in flags:
            feats.append('TFS'+k)
    print >>sys.stderr, p,flags
    print >>sys.stderr, get_verb_features(p)
    print >>sys.stderr, "args[sub]:"
    nomargs_sub=find_nomargs(sub_cl)
    for k,v in nomargs_sub:
        print >>sys.stderr, "  %s: %s[%s]"%(k,v.to_penn(), semclass_for_node(v))
    (p,flags)=get_verbs(main_cl)
    if 'lex' in wanted_features:
        feats.append('LM'+p)
    if 'assoc' in wanted_features and p in assoc_features:
        for k in assoc_features[p]:
            feats.append('AM'+k)
    if 'tmp' in wanted_features:
        for k in flags:
            feats.append('TFM'+k)
    print >>sys.stderr, p,flags
    print >>sys.stderr, get_verb_features(p)
    print >>sys.stderr, "args[main]:"
    nomargs_main=find_nomargs(main_cl)
    for k,v in nomargs_main:
        print >>sys.stderr, "  %s: %s[%s]"%(k,v.to_penn(),semclass_for_node(v))
    if 'mod' in wanted_features:
        mod_main=find_adjuncts(main_cl,[sub_cl])
        mod_sub=find_adjuncts(sub_cl)
        for (mod,modS) in [(mod_main,'M'),(mod_sub,'S')]:
            for k in ['tmp','causal','concessive','focus-koord-incl','comment']:
                if k in mod:
                    val='+'
                else:
                    val='-'
                feats.append('mod%s%s%s'%(modS,k,val))
        if mod_main:
            print >>sys.stderr, "mod[main]"
            for k in mod_main:
                for v in mod_main[k]:
                    print >>sys.stderr, " %s: %s"%(k,v.to_penn())
        if mod_sub:
            print >>sys.stderr, "mod[sub]"
            for k in mod_sub:
                for v in mod_sub[k]:
                    print >>sys.stderr, " %s: %s"%(k,v.to_penn())
    if 'csubj' in wanted_features:
        subj_main=[x[1] for x in nomargs_main if x[0]=='ON']
        subj_sub=[x[1] for x in nomargs_sub if x[0]=='ON']
        if len(subj_main)==1 and len(subj_sub)==1:
            if compatible_pronoun(subj_main[0],subj_sub[0]):
                feats.append('cpM+')
            else:
                feats.append('cpM-')
            if compatible_pronoun(subj_sub[0],subj_main[0]):
                feats.append('cpS+')
            else:
                feats.append('cpS-')
    main_end=main_cl.end
    if 'punc' in wanted_features:
        if len(t.terminals)>main_end and t.terminals[main_end].cat=='$.':
            feats.append('PUNC'+t.terminals[main_end].word)
    return feats

def production_features(t,sub_cl,main_cl, do_filter=True):
    result=[]
    lst_main=[]
    lst_sub=[]
    get_productions(main_cl,[sub_cl],lst_main)
    get_productions(sub_cl,[main_cl],lst_sub)
    set_main=set(lst_main)
    if do_filter:
        set_main.intersection_update(wanted_productions)
    set_sub=set(lst_sub)
    if do_filter:
        set_sub.intersection_update(wanted_productions)
    for k in set_main:
        if k in set_sub:
            result.append('prB%s'%(k,))
        else:
            result.append('prM%s'%(k,))
    for k in set_sub:
        if k not in set_main:
            result.append('prS%s'%(k,))
    return result

ignore_pos=['NN','NE','VVFIN','VVINF','VVIZU','VVPP']
#ignore_pos=[]

#TODO: only include a given wordpair once?
def wordpair_features(t,sub_cl,main_cl,conn, do_filter=True):
    result=[]
    idx_main=set(xrange(main_cl.start,main_cl.end))
    idx_sub=set(xrange(sub_cl.start,sub_cl.end))
    idx_main.difference_update(idx_sub)
    #idx_sub.discard(conn)
    #idx_sub.difference_update(idx_main)
    words_main=[t.terminals[i].lemma for i in sorted(idx_main)
                if t.terminals[i].cat not in ignore_pos]
    words_sub=[t.terminals[i].lemma for i in sorted(idx_sub)
               if t.terminals[i].cat not in ignore_pos]
    wps=['%s_%s'%(x,y) for x in words_main for y in words_sub]
    #wps.intersection_update(wanted_pairs)
    if do_filter:
        for k in wps:
            if k in wanted_pairs:
                result.append('WP'+k)
        print result
    else:
        for k in wps:
            result.append('WP'+k)
    return result

def get_terminals_by_idx(t,idxs):
    nodes=[]
    for idx in idxs:
        n=t.terminals[idx]
        nodes.append(n)
    return nodes

def retrieve_synsets(nodes):
    wanted_lemmas=[]
    for n in nodes:
        if n.cat in ['NN','NE','ADJA','ADJD','VVFIN','VVINF','VVIZU']:
            synsets=wordsenses.synsets_for_lemma(n.lemma,n.cat)
            wanted_lemmas.append((n,synsets))
    return wanted_lemmas

def expanded_synsets_for_lemma(lemma,pos):
    synsets=wordsenses.synsets_for_lemma(lemma,pos)
    if not synsets:
        return []
    depths=[wordsenses.synset_depth(syn) for syn in synsets]
    max_depth=max(depths)
    min_depth=min(depths)
    if pos=='NN':
        limit=max(3,min_depth-2)
    else:
        limit=max(0,min_depth-2)
    hyp_map={}
    for syn,d in izip(synsets,depths):
        wordsenses.gather_hyperonyms(syn.synsetId,hyp_map,0,d-limit)
    return hyp_map.keys()

def lexrel_features(t, sub_cl, main_cl,feats):
    idx_main=set(xrange(main_cl.start,main_cl.end))
    idx_sub=set(xrange(sub_cl.start,sub_cl.end))
    idx_main.difference_update(idx_sub)
    terminals_s=get_terminals_by_idx(t,idx_sub)
    terminals_m=get_terminals_by_idx(t,idx_main)
    lexrel_features_1(terminals_s, terminals_m,feats)

def lexrel_features_1(terminals_s, terminals_m,feats):
    print >>sys.stderr, "lexrel_features_1 %s // %s"%(' '.join([n.word for n in terminals_s]),
                                                      ' '.join([n.word for n in terminals_m]))
    lemmas_s=retrieve_synsets(terminals_s)
    lemmas_m=retrieve_synsets(terminals_m)
    hyp_map={}
    for n_s,synsets_s in lemmas_s:
        for n_m,synsets_m in lemmas_m:
            assert n_s!=n_m
            # gwn_path=wordsenses.relate_senses(synsets_s,
            #                                   synsets_m)
            # if gwn_path:
            #     print 'GWN', n_s.lemma, n_m.lemma, gwn_path
            lcs,dist=wordsenses.lcs_path(synsets_s,synsets_m)
            if lcs is not None:
                lcs_depth=wordsenses.synset_depth(lcs)
                print >>sys.stderr, "GWN-LCS",n_s,n_m,lcs.explain(), dist, lcs_depth
                # if lcs in synsets_s:
                #     if lcs in synsets_m:
                #         feats.append('lcs_synonym')
                #     else:
                #         feats.append('lcs_hyperS')
                #         #feats.append('lcs_hyperS_%d'%(lcs.synsetId,))
                # elif lcs in synsets_m:
                #         feats.append('lcs_hyperM')
                #         #feats.append('lcs_hyperM_%d'%(lcs.synsetId,))                      
                #feats.append('lcs_exact_%d'%(lcs.synsetId,))
                if lcs.lexGroup.wordclass[0]!='n':
                    wordsenses.gather_hyperonyms(lcs.synsetId,hyp_map,0,2)
                elif lcs_depth>2:
                    wordsenses.gather_hyperonyms(lcs.synsetId,hyp_map,0,min(2,lcs_depth-3))
    for k in hyp_map:
        feats.append('lcs_super_%d'%(k,))

class FeatureCounter:
    def __init__(self):
        self.counts=defaultdict(int)
        self.total=0
    def add(self,lst):
        all_productions=set(lst)
        for k in all_productions:
            self.counts[k]+=1
        self.total+=1
    def by_entropy(self,x):
        if x[1]==self.total: return 0
        p=float(x[1])/self.total
        return -p*math.log(p)*(1.0-p)*math.log(1-p)
    def select(self,key=None,N=500,min_count=5):
        if key==None:
            key=self.by_entropy
        counts=sorted(self.counts.iteritems(),key=key)
        print "Cutoff at %s (with min_count=%d)"%(counts[N],min_count)
        return set([x[0] for x in counts[:N] if x[1]>=min_count])

def do_counting(spans):
    constituents=FeatureCounter()
    wordpairs=FeatureCounter()
    for span in spans:
        sent_no=db.sentences.cpos2struc(span[0])
        sent_span=db.sentences[sent_no]
        offset=span[0]-sent_span[0]
        t=export.from_json(db.get_parses(sent_no)['release'])
        for n in t.topdown_enumeration():
            if n.cat=='NCX': n.cat='NX'
        for n,lemma in izip(t.terminals,lemmas[sent_span[0]:sent_span[1]+1]):
            n.lemma=lemma
        stupid_head_finder(t)
        all_productions=set()
        n=t.terminals[offset]
        sub_cl,main_cl=find_args(n)
        if not main_cl: continue
        if not sub_cl: continue
        lst=[]
        get_productions(main_cl,[sub_cl],lst)
        get_productions(sub_cl,[main_cl],lst)
        constituents.add(lst)
        idx_main=set(xrange(main_cl.start,main_cl.end))
        idx_sub=set(xrange(sub_cl.start,sub_cl.end))
        idx_main.difference_update(idx_sub)
        #idx_sub.discard(offset)
        #idx_sub.difference_update(idx_main)
        words_main=[t.terminals[i].lemma for i in sorted(idx_main)
                    if t.terminals[i].cat not in ignore_pos]
        words_sub=[t.terminals[i].lemma for i in sorted(idx_sub)
                   if t.terminals[i].cat not in ignore_pos]
        wordpairs.add(['%s_%s'%(x,y) for x in words_main for y in words_sub])
    features_const=constituents.select()
    features_pairs=wordpairs.select()
    return features_const,features_pairs

buckets=[3,5,10,15,20]
def discretize(n):
    if n<buckets[0]:
        return n
    for b in buckets[1:]:
        if n<=b:
            return b
    return buckets[-1]+1

pos_groups={'$,':'c','ADJA':'a','NN':'n',
            'VVFIN':'v','VVPP':'v','VVINF':'v',
            'VVIZU':'v'}
def arg_lengths(t,sub_cl,main_cl,conn):
    result=[]
    idx_main=set(xrange(main_cl.start,main_cl.end))
    idx_sub=set(xrange(sub_cl.start,sub_cl.end))
    idx_main.difference_update(idx_sub)
    idx_sub.discard(conn)
    result.append('alTM%d'%(discretize(len(idx_main,))))
    result.append('alTS%d'%(discretize(len(idx_sub,))))
    group_counts=defaultdict(int)
    for i in idx_main:
        t_cat=t.terminals[i].cat
        if t_cat in pos_groups:
            group_counts[pos_groups[t_cat]]+=1
    for k in group_counts:
        result.append('al%sM%d'%(k,discretize(group_counts[k])))
    group_counts=defaultdict(int)
    for i in idx_sub:
        t_cat=t.terminals[i].cat
        if t_cat in pos_groups:
            group_counts[pos_groups[t_cat]]+=1
    for k in group_counts:
        result.append('al%sS%d'%(k,discretize(group_counts[k])))
    return result

def mark_nodes(nodes1,nodes2):
    for n1 in nodes1:
        n1.is_root=True
    for n2 in nodes2:
        n2.is_root=True
    for n1 in nodes1:
        mark_node(n1,'1',nodes2)
    for n2 in nodes2:
        mark_node(n2,'2',nodes1)

def unmark_nodes(nodes1,nodes2):
    for n in nodes1:
        unmark_node(n)
    for n in nodes2:
        unmark_node(n)

def mark_node(n,mark,exclude):
    if n in exclude:
        n.is_root=False
        return
    n.mark=mark
    if not n.isTerminal():
        for n1 in n.children:
            mark_node(n1,mark,exclude)

def unmark_node(n):
    if hasattr(n,'mark'):
        del n.mark
        if hasattr(n,'is_root'):
            del n.is_root
        if not n.isTerminal():
            for n1 in n.children:
                unmark_node(n1)

def clause_children(n1, exclude):
    field=None
    result=[]
    for n2 in n1.children:
        if n2.cat in ['VF','MF','NF'] and n2 not in exclude:
            for n3 in n2.children:
                result.append((n2.cat, n3))
        elif n2.cat == 'C' and len(n2.children)==1:
            n3=n2.children[0]
            if n3.cat in ['NX','PX','ADVX']:
                result.append(('C',n3))
        elif n2.cat=='FKOORD':
            fkonj_seen=False
            for n2a in n2.children:
                if n2a.cat=='FKONJ' and not fkonj_seen and n2a not in exclude:
                    result+=clause_children(n2a,exclude)
                    fkonj_seen=True
                elif n2a.cat=='MF' and n2a not in exclude:
                    for n3 in n2a.children:
                        result.append((n2a.cat, n3))
                    fkonj_seen=True
    return result

def clause_nonchildren(n1, exclude):
    field=None
    result=[]
    for n2 in n1.children:
        if n2.cat in ['LK','VC'] and n2 not in exclude:
            result.append(n2)
        elif n2.cat=='FKOORD':
            fkonj_seen=False
            for n2a in n2.children:
                if n2a.cat=='FKONJ' and not fkonj_seen and n2a not in exclude:
                    result+=clause_nonchildren(n2a,exclude)
                    fkonj_seen=True
                elif n2a.cat in ['LK','VC'] and n2a not in exclude:
                    result.append(n2a)
                    fkonj_seen=True
    return result


def get_istatus(node):                
    istatus='new'
    if hasattr(node,'anaphora_info'):
        a_kind=node.anaphora_info[0]
        if a_kind in ['expletive','inherent_reflexive']:
            kind=a_kind
            istatus='expl'
        else:
            istatus='old'
    else:
        for n in node.children:
            if n.edge_label in ['HD','APP']:
                is2=get_istatus(n)
                if is2!='new':
                    return is2
            elif n.edge_label=='-' and n.cat in ['NX','PPOSAT']:
                is2=get_istatus(n)
                if is2!='new':
                    return 'mediated'
            elif n.edge_label=='-' and n.cat=='PX':
                if len(n.children)==2 and n.children[0].cat=='APPR' and n.children[1].cat=='NX':
                    is2=get_istatus(n.children[1])
                    if is2!='new':
                        return 'mediated'
    return istatus

def munge_single_phrase(node):
        feats=['cat:'+node.cat]
        kind='MOD'
        if node.edge_label in ['ON','OA','OD','OG','OPP','FOPP','PRED']:
            kind='ARG'
            feats.append('gf:'+node.edge_label)
        if node.cat=='PX':
            cls=classify_px(node)
            if cls is not None:
                feats.append('cls:'+cls)
        elif node.cat=='ADVX':
            cls=classify_adverb(node)
            if cls is not None:
                feats.append('cls:'+cls)
        elif node.cat=='NX':
            if 'nosemG' not in wanted_features:
                if hasattr(node,'ne_kind'):
                    sc=node.ne_kind
                    if sc=='GPE':
                        sc='LOC'
                else:
                    sc=semclass_for_node(node)
                if sc is not None:
                    feats.append('sem:%s'%(sc,))
            if 'istatusG' in wanted_features:
                istatus=get_istatus(node)
                feats.append('ref:%s'%(istatus,))
        if hasattr(node,'head') and 'delexG' not in wanted_features:
            feats.append('lm:'+node.head.lemma)
            if node.cat=='PX':
                if len(node.children)>1 and node.children[1].cat=='NX':
                    feats.append('arg:'+node.children[1].head.lemma)
            # if node.head.cat=='NN':
            #     feats+=['hyp%d'%(k,) for k in expanded_synsets_for_lemma(pred,'NN')]
        return kind,feats


def punc_type(node,doc):
    result=[]
    posn=node.span[1]
    last_punc=None
    quote_at_end=False
    while doc.w_objs[posn].cat[0]=='$':
        if doc.w_objs[posn].cat=='$(':
            quote_at_end=True
        else:
            last_punc=doc.words[posn]
        posn+=1
    try:
        if (quote_at_end and
            doc.w_objs[node.span[0]-1].cat=='$('):
            result.append('quoted')
    except IndexError:
        pass
    if last_punc is not None:
        result.append('PUNC'+last_punc)
    return result

def make_simple_tree(main_cl, exclude, other_terms=None, doc=None):
    if other_terms is None:
        other_terms=tree_yield(exclude,[main_cl])
        all_exclude=exclude
    else:
        all_exclude=exclude+other_terms
    (pred,flags)=get_verbs(main_cl)
    feats=list(flags)+['cat:'+main_cl.cat,'cltype:'+get_clause_type(main_cl)]
    if 'puncG' in wanted_features and doc is not None:
        feats+=punc_type(main_cl,doc)
    if 'delexG' not in wanted_features:
        feats.append('lm:'+pred)
    if 'sentimentG' in wanted_features:
        for n3 in  clause_nonchildren(main_cl, exclude):
            tag=sentiment.phrase_tag(n3)
            if tag is not None:
                feats.append('senti_'+tag[1])
    ni1=InfoNode('S',feats)
    for n2_cat, n3 in clause_children(main_cl, exclude):
        if n3 in exclude:
            extra_features=['fd:'+n2_cat]
            if 'puncG' in wanted_features and doc is not None:
                extra_features+=punc_type(n3,doc)
            if n3.cat=='R-SIMPX':
                ni2=InfoNode('REL_CL',extra_features)
            else:
                ni2=InfoNode('SUB_CL',extra_features+['cltype:'+get_clause_type(n3)])
            ni1.add_edge(ni2)
            continue
        if n3.edge_label.endswith('-MOD') and n3.edge_label!='V-MOD':
            continue
        (kind,feats)=munge_single_phrase(n3)
        if 'lexrelG' in wanted_features:
            lexrel_features_1(tree_yield([n3],all_exclude),other_terms,feats)
        if 'sentimentG' in wanted_features:
            tag=sentiment.phrase_tag(n3)
            if tag is not None:
                feats.append('senti_'+tag[1])
        feats+=['fd:'+n2_cat]
        ni2=InfoNode(kind,feats)
        ni1.add_edge(ni2)
    last_child=main_cl.children[-1]
    if last_child.cat=='NF' and last_child in exclude and last_child.children[-1] not in exclude:
        lbl=last_child.children[-1].edge_label
        (kind,feats)=munge_single_phrase(last_child.children[-1])
        ni2=InfoNode('RV',feats)
        ni1.add_edge(ni2)
    return ni1

def node2tree(n):
    ti=InfoTree()
    ti.add_node(n,True)
    return ti.as_json()

def process_spans(spans,annotator):
    for span in spans:
        sent_no=db.sentences.cpos2struc(span[0])
        sent_span=db.sentences[sent_no]
        #t=export.from_json(db.get_parses(sent_no)['release'])
        t=doc.get_tree(sent_no)
        for n in t.topdown_enumeration():
            if n.cat=='NCX': n.cat='NX'
        stupid_head_finder(t)
        for n,lemma in izip(t.terminals,lemmas[sent_span[0]:sent_span[1]+1]):
            n.lemma=lemma
        offset=span[0]-sent_span[0]
        n=t.terminals[offset]
        anno=db.get_annotation(annotator,'konn2',span)
        if not hasattr(anno,'rel1') or anno.rel1=='NULL': continue
        print "--- s%s"%(sent_no+1,)
        sub_cl,main_cl=find_args(n)
        if not main_cl: continue
        if not sub_cl: continue
        print "SUB: ",sub_cl.to_penn()
        print "MAIN:",main_cl.to_penn()
        #print "anno: temporal=%s contrastive=%s"%(anno.temporal,anno.contrastive)
        feats=get_features(t,sub_cl,main_cl)
        aux_lst=[]
        if 'productions' in wanted_features:
            feats+=production_features(t,sub_cl,main_cl)
        if 'productionsA' in wanted_features:
            aux_lst.append(production_features(t, sub_cl, main_cl, False))
        if 'wordpairs' in wanted_features:
            feats+=wordpair_features(t,sub_cl,main_cl,offset)
        if 'wordpairsA' in wanted_features:
            aux_lst.append(wordpair_features(t,sub_cl,main_cl,offset, False))
        if 'arglen' in wanted_features:
            feats+=arg_lengths(t, sub_cl, main_cl, offset)
        if 'lexrel' in wanted_features:
            lexrel_features(t, sub_cl, main_cl,feats)
        target=get_target(anno)
        print target
        #print "anno: rel1=%s rel2=%s"%(anno.rel1,anno._doc.get('rel2','NULL'))
        print feats
        #print anno._doc
        # print >>f_out, json.dumps([0,map(grok_encoding,feats),
        #                           target,[span[0],span[1]-1]],encoding='ISO-8859-15')
        print >>f_out, json.dumps([0,{'_type':'multipart','parts':[map(grok_encoding,feats)]+aux_lst,
                                      'trees':[node2tree(make_simple_tree(main_cl, [sub_cl], doc=doc)),
                                               node2tree(make_simple_tree(sub_cl, [], doc=doc))]},
                                  target,[span[0],span[1]-1]],encoding='ISO-8859-15')

#wanted_features=['csubj','mod','lex','tmp','neg','punc','lexrel','assoc']
#wanted_features=['csubj','mod','lex','tmp','neg','punc','lexrel','assoc','wordpairs','productions']
wanted_features=['csubj','mod','lex','tmp','neg','punc','lexrel','wordpairs','productions']
#wanted_features=['csubj','mod','lex','tmp','neg','punc','lexrel']
#wanted_features=['csubj','mod','lex','tmp','neg','punc','lexrel','wordpairsA','productionsA']
#wanted_features=['nobaseline']
#wanted_features=['wordpairs','productions']
#wanted_features=[]

def init_db(wanted_features, dbname='R6PRE1'):
    global db
    global doc
    global lemmas
    doc=exml2db.DBDocument(dbname)
    db=annodb.get_corpus(dbname)
    lemmas=db.corpus.attribute('lemma','p')

def do_initialization(wanted_features, dbname='R6PRE1'):
    global assoc_features
    global wanted_productions
    global wanted_pairs
    if 'assoc' in wanted_features:
        assoc_features={}
        for l in file('word_assoc.txt'):
            line=l.strip().split()
            assoc_features[line[0]]=line[1:]
    if 'productions' in wanted_features or 'wordpairs' in wanted_features:
        #wanted_productions,wanted_pairs=do_counting(spans_n+spans_w2)
        wanted_productions,wanted_pairs=do_counting(spans_n+spans_w)
        print wanted_productions
        print wanted_pairs

def test_get_verbs(sent_no=17096):
    global db
    global lemmas
    db=annodb.get_corpus('TUEBA4')
    lemmas=db.corpus.attribute('lemma','p')
    #
    sent_span=db.sentences[sent_no]
    t=export.from_json(db.get_parses(sent_no)['release'])
    for n in t.topdown_enumeration():
        if n.cat=='NCX': n.cat='NX'
    stupid_head_finder(t)
    for n,lemma in izip(t.terminals,lemmas[sent_span[0]:sent_span[1]+1]):
        n.lemma=lemma
    for node in t.topdown_enumeration():
        if node.cat=='SIMPX':
            print node.to_penn()
            pred,flags=get_verbs(node)
            print pred
            print flags
            print get_clause_type(node)

oparse=optparse.OptionParser()
oparse.add_option('--fprefix',dest='fprefix',
                  default='')
oparse.add_option('--features',dest='features')
oparse.add_option('--corpus',dest='corpus_name',
                  default='R6PRE1')

if __name__=='__main__':
    opts,args=oparse.parse_args(sys.argv[1:])
    if len(args)>=1:
        wanted_features=args[0].split(',')
    fprefix=opts.fprefix
    init_db(wanted_features,opts.corpus_name)
    tasks_n=[db.get_task('task_nachdem%d_new'%(n,)) for n in xrange(1,7)]
    tasks_n5=[db.get_task('task_nachdem%d_new'%(n,)) for n in [5]]
    tasks_w=[db.get_task('task_waehrend%d_new'%(n,)) for n in xrange(1,7)]
    tasks_w2=[db.get_task('task_waehrend%d_new'%(n,)) for n in xrange(1,11)]
    tasks_und=[db.get_task('task_und_r6_%d'%(n,)) for n in xrange(1,7)]
    spans_n=sorted(set([tuple(span) for task in tasks_n for span in task.spans]))
    spans_n5=sorted(set([tuple(span) for task in tasks_n5 for span in task.spans]))
    spans_w=sorted(set([tuple(span) for task in tasks_w for span in task.spans]))
    spans_w2=sorted(set([tuple(span) for task in tasks_w2 for span in task.spans]))
    spans_und=sorted(set([tuple(span) for task in tasks_und for span in task.spans]))

    ## 1. prepare necessary data (assoc, productions, wordpairs)
    do_initialization(wanted_features,opts.corpus_name)

    ## 2. create data for nachdem and waehrend
    f_out=file(fprefix+'nachdem_1-6.json','w')
    process_spans(spans_n,'melike')
    f_out.close()

    f_out=file(fprefix+'nachdem_5.json','w')
    process_spans(spans_n5,'melike')
    f_out.close()

    # TBD: update wanted_productions
    f_out=file(fprefix+'waehrend_1-4.json','w')
    process_spans(spans_w,'melike')
    f_out.close()

    f_out=file(fprefix+'waehrend_1-10.json','w')
    process_spans(spans_w2,'melike')
    f_out.close()

    # tasks2=[db.get_task('task_aberA_new')]
    # print tasks2
    # spans=sorted(set([tuple(span) for task in tasks2 for span in task.spans]))

    # f_out=file('aberA.json','w')
    # process_spans(spans,'sabrina')
    # f_out.close()
