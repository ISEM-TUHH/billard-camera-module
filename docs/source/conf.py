# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import glob

source_code_dir_name = "Game" # change here

sys.path.insert(0, os.path.abspath('../..'))  # Source code dir relative to this file
sys.path.insert(0, os.path.abspath('../../..'))
# for recursion: add all paths below

print("CWD:", os.getcwd())
sys.path.append(f"../../{source_code_dir_name}/")
for path in glob.glob(f"../../{source_code_dir_name}/**/*/", recursive=True):
    pass
    #print(os.path.abspath(path))
    #sys.path.append(os.path.abspath(path))

def add_to_path():

    partial_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')
    print("PARTIAL:", partial_path)
    workspace_path = os.path.abspath(partial_path)
    assert os.path.exists(workspace_path)

    projects = []

    for current, dirs, c in os.walk(str(workspace_path)):
        for dir in dirs:

            project_path = os.path.join(workspace_path, dir, 'src')

            if os.path.exists(project_path):
                print("PP:", project_path)
                projects.append(project_path)

    for project_str in projects:
        sys.path.append(project_str)

#add_to_path()

print(sys.executable)
print(sys.path)
#import Game
#import GameEngine



project = 'camera module - Billard@ISEM'
copyright = '2025, Mathis Wolter'
author = 'Mathis Wolter'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_mdinclude',
    'sphinx.ext.todo'
]

templates_path = ['_templates']
exclude_patterns = []

#autoclass_content = 'both'
autosummary_generate = True

todo_include_todos = True # for showing todos

autodoc_default_options = {'inherited-members': False}
toc_object_entries_show_parents = "hide" # only write Class instead of Parent.Parent.Module.Class in TOC and headings

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
