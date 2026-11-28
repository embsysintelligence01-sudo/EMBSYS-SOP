"""Temporary legacy import identity bridge; never duplicates implementation state."""
import importlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
_ROOT=Path(__file__).resolve().parents[1]
_MAPPING=json.loads((_ROOT/'configs/import_compatibility.json').read_text())

class _AliasLoader(importlib.abc.Loader):
    def __init__(self,target):self.target=target
    def create_module(self,spec):return importlib.import_module(self.target)
    def exec_module(self,module):pass

class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        canonical=_MAPPING.get(fullname)
        if canonical is None:
            for prefix in ('acid_test','acidphintegration','new_rub_detection'):
                if fullname.startswith(prefix+'.'):
                    canonical=_MAPPING[prefix]+fullname[len(prefix):]
                    break
        if canonical is None:return None
        package=fullname in ('standalone','acid_test','acidphintegration','new_rub_detection') or fullname.endswith('.model_training')
        return importlib.util.spec_from_loader(fullname,_AliasLoader(canonical),is_package=package)

def install():
    if not any(isinstance(f,_AliasFinder) for f in sys.meta_path):sys.meta_path.insert(0,_AliasFinder())
    for directory in (_ROOT,_ROOT/'engines/Jeweltracker',_ROOT/'engines/Packetsealing_001'):
        if str(directory) not in sys.path:sys.path.append(str(directory))

def alias(name,target):
    install()
    module=importlib.import_module(target)
    sys.modules[name]=module
    return module

# Transitional bridge for the former monolithic tracker. Executing each exact
# AST definition at its original position preserves globals, defaults, QObject
# identity and tests that patch tracker symbols. No algorithm transformation.
from functools import lru_cache
import ast

@lru_cache(maxsize=None)
def _definitions(relative):
    path=_ROOT/relative
    tree=ast.parse(path.read_text())
    return {node.name:node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}

def bind_definition(relative,name,namespace):
    node=_definitions(relative)[name]
    module=ast.Module(body=[node],type_ignores=[])
    exec(compile(module,str(_ROOT/relative),'exec'),namespace)
