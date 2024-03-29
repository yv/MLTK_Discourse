__doc__='''Language-independent utilities for creating or
transforming trees'''

def tree_yield(nodes,exclude,result=None):
    '''
    returns all terminal nodes in a tree that are
    descendants of a node in ``nodes`` but not passing
    through a node in the ``exclude`` list.
    '''
    if result is None:
        result=[]
    for n in nodes:
        if n in exclude:
            print >>sys.stderr, "excluded:"+n.to_penn()
            continue
        if n.isTerminal():
            result.append(n)
        else:
            tree_yield(n.children,exclude,result)
    return result

def get_productions(n,exclude,lst):
    '''
    gets all CFG production in the subtree given by
    ``n``, minus those that are in the ``exclude`` list.
    The resulting productions will be added to ``lst``.
    '''
    if n.isTerminal():
        lst.append('%s=%s'%(n.cat,n.word))
    else:
        lst.append('%s=%s'%(n.cat,'-'.join([n1.cat for n1 in n.children])))
        for n1 in n.children:
            if n1 not in exclude:
                get_productions(n1,exclude,lst)

def extract_bigrams(prefix,attr,terms,want_unigram=True):
    '''
    extracts all bigrams for one particular attribute.
    Prefixes the extracted items with ``prefix``
    '''
    if want_unigram:
        result=[prefix+getattr(n,attr) for n in terms]
    else:
        result=[]
    if not terms:
        return result
    last_w=getattr(terms[0],attr)
    for i in xrange(1,len(terms)):
        next_w=getattr(terms[i],attr)
        result.append('%s%s_%s'%(prefix,last_w,next_w))
        last_w=next_w
    return result
