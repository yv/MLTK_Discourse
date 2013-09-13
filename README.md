MLTK_Discourse: feature extraction for discourse relations
==========================================================

This package performs the extraction of linguistic, shallow and
structural features for discourse relations from corpora in
JSON-Exml format.

Quickstart
----------
Take a corpus in JSON-Exml format, let's say we have one called ``r7final_exml.json``.

We run

    features_exml r7final_exml.json

which produces a lot of output and creates several files in the current directory:
 * nachdem_exml.json, waehrend_exml.json, als_exml.json: Files with connective examples
 * unmarked_all_exml.json: File with implicit relations
 * unmarked_exml.json: File with implicit relation, excluding all where temporal adverbs,
   reporting verbs or other weak indicators are present.

The resulting files can be fed to MLTK's xvalidate_mlab program (in the example,
we use feature selection to use only :

    xvalidate_mlab --featsel chi2 --featsize 0,-1 unmarked_all_exml.json

This should give some results detailing a Dice score of 0.539 on the coarsest level.

We can transform the graphs that have been extracted by the program into features
by using MLTK's ``graphs_to_features`` tool:

   graphs_to_features unmarked_all_exml.json /tmp/some-temporary-directory > unmarked_gfeatures.json

The resulting file contains an additional feature group containing graph features,
and can, again, be fed into xvalidate_mlab, this time with some real/more serious
attempt at feature selection:

    xvalidate_mlab --featsel chi2 --featsel 0,5000,-1,5000,20000 --labelfilter unmarked_gfeatures.json

Finer points
------------

features_exml allows to set which linguistic features to use via the
``--features`` flag. Most features are explained in Versley 2011
("Multilabel Tagging of Discourse Relations in Ambiguous Temporal
Connectives"). With release 7 of the treebank, results for *nachdem*
differ by 0.5% from those in the RANLP paper because of a change in
the syntax annotation.