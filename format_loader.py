from __future__ import print_function, unicode_literals
import os
import traceback
import importlib
import glob
from types import ModuleType
import time

_formats = {}

class _FormatLoader(object):

    REQUIRED_ATTRIBUTES = [
        'LEVEL_CLASS',
        'REGION_CLASS',
        'CHUNK_CLASS',
        'MATERIALS_CLASS',
        'identify'
    ]

    def __init__(self, search_directory='.'):
        self.search_directory = search_directory

        self._find_formats()

    def _find_formats(self):
        global _formats
        directories = glob.glob(os.path.join(self.search_directory, '*', ''))
        for d in directories:
            if not os.path.exists(os.path.join(d, '__init__.py')):
                continue
            format_name = os.path.dirname(d)[2:]
            success, module = self.load_format(format_name)
            if success:
                _formats[format_name] = module

    def load_format(self, directory):
        try:
            format_module = importlib.import_module(directory)
        except Exception as e:
            traceback.print_exc()
            time.sleep(0.01)
            print('Could not import the "{}" format due to the above Exception'.format(directory))
            return False, None
        for attr in self.REQUIRED_ATTRIBUTES:
            if not hasattr(format_module, attr):
                print('Disabled the "{}" format due to missing required attributes'.format(directory))
                return False, None
        return True, format_module

    def reload(self):
        self._find_formats()

    def add_external_format(self, name, module):
        global _formats
        if isinstance(name, (str, unicode)) and isinstance(module, ModuleType):
            _formats[name] = module
        else:
            raise Exception('To add an external format you must supply a name and a module object!')

def load_world(world_directory):
    for format_module in _formats.itervalues():
        if format_module.identify(world_directory):
            return format_module.LEVEL_CLASS(world_directory)
    return None

loader = _FormatLoader()

if __name__ == '__main__':
    world = load_world(os.path.join('tests', '1.13 World'))
    chunk = world.getChunk(0,0)
    print(world.heightMapAt(0,0))
    print(world.biomeAt(0,0))
    world.preloadChunks()