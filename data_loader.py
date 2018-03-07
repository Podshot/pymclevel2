from __future__ import print_function, unicode_literals
import os
import json
import collections

def update_dict(orig_dict, new_dict):
    for key, val in new_dict.iteritems():
        if isinstance(val, collections.Mapping):
            if orig_dict.get(key, {}) == val:
                continue
            tmp = update_dict(orig_dict.get(key, { }), val)
            orig_dict[key] = tmp
        elif isinstance(val, list):
            if orig_dict.get(key, []) == val:
                continue
            orig_dict[key] = (orig_dict.get(key, []) + val)
        else:
            if orig_dict.get(key, None) == new_dict[key]:
                continue
            orig_dict[key] = new_dict[key]
    return orig_dict

class MCVersion(object):

    BLOCKS = 'blocks.json'
    ENTITIES = 'entities.json'
    TILE_ENTITIES = 'tileentities.json'
    versions = {}

    @classmethod
    def getMCVersion(cls, version):
        return cls.versions.get(version, None)

    def __init__(self, path):
        self._path = path
        self._version = os.path.basename(path)
        self._blocks_file = os.path.join(path, self.BLOCKS)
        self._entity_file = os.path.join(path, self.ENTITIES)
        self._tile_entity_file = os.path.join(path, self.TILE_ENTITIES)

        self.blocks = {}
        self.entities = {}
        self.tile_entities = {}

        self.load()
        self.versions[self._version] = self

    def load_dependency(self, version, attr):
        if version in self.versions:
            mcversion = self.versions[version]
        else:
            mcversion = MCVersion(os.path.join(os.path.dirname(self._path), version))
        return getattr(mcversion, attr)

    def load(self):
        blocks_fp = open(self._blocks_file)
        blocks_data = json.load(blocks_fp)
        blocks_fp.close()
        if 'depends_on' in blocks_data:
            block_deps = self.load_dependency(blocks_data['depends_on'], 'blocks')
        else:
            block_deps = {}
        self.blocks = update_dict(block_deps, blocks_data)
        if 'depends_on' in self.blocks:
            del self.blocks['depends_on']

        entities_fp = open(self._entity_file)
        entity_data = json.load(entities_fp)
        entities_fp.close()
        if 'depends_on' in entity_data:
            entity_deps = self.load_dependency(entity_data['depends_on'], 'entities')
        else:
            entity_deps = {}
        self.entities = update_dict(entity_deps, entity_data)
        if 'depends_on' in self.entities:
            del self.entities['depends_on']

        tile_entity_fp = open(self._tile_entity_file)
        tile_entity_data = json.load(tile_entity_fp)
        tile_entity_fp.close()
        if 'depends_on' in tile_entity_data:
            tile_entity_deps = self.load_dependency(tile_entity_data['depends_on'], 'tile_entities')
        else:
            tile_entity_deps = {}
        self.tile_entities = update_dict(tile_entity_deps, tile_entity_data)
        if 'depends_on' in self.tile_entities:
            del self.tile_entities['depends_on']

if __name__ == '__main__':
    ver = MCVersion(os.path.join('new_mcver', '1.12'))
    print(ver.blocks['minecraft'].keys())
    ver2 = MCVersion(os.path.join('new_mcver', '1.13'))
    print(ver2.blocks['minecraft'].keys())
    print(ver2.entities['minecraft'].keys())



