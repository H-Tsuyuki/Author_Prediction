#! /usr/bin/python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
import sys
import MeCab

m = MeCab.Tagger()
a = m.parse('我輩は猫である。名前はまだ無い。')
a = [i.split("\t")[0] for i in a.split('\n')][:-2]
from IPython.core.debugger import Pdb; Pdb().set_trace()
#
# ----------------------------------------------------------------------
