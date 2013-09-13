#!/usr/bin/env python
from setuptools import setup, find_packages, Extension
#from Cython.Distutils import build_ext
import sys
import os
import os.path
import numpy

incdirs=[numpy.get_include(),'include','pyx_src']


setup(name='MLTK-Discourse',
      version='0.1',
      description='Feature Extraction for Connectives and Implicit Discourse Relations',
      author='Yannick Versley',
      author_email='versley@sfs.uni-tuebingen.de',
      entry_points = { 'console_scripts': [
            'features_exml = mltk_discourse.features_exml:features_exml_main']
                       },
      packages=['mltk_discourse'],
      package_dir={'':'py_src'},
      package_data={'mltk_discourse':['konn2_schema.txt','disc_schema.txt']}
      )                 
