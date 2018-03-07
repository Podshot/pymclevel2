"""
Be weary ye who enter, this shall probably be the worst and over-designed code thou will ever see
"""
from __future__ import print_function, unicode_literals
import os
import data_loader
import numpy as np

class Blockstate(object):

    comparable_attributes = ('_resource_loc', '_name', '_properties')

    def __init__(self, resource_location='minecraft', name='air', properties=None):
        self._resource_loc = resource_location
        self._name = name
        if properties:
            self._properties = properties
        else:
            self._properties = {}
        self._str = self.buildStr(self._resource_loc, self._name, self._properties)

    def __repr__(self):
        return 'Blockstate({}, {}, {})'.format(self._resource_loc, self._name, self._properties)

    def __str__(self):
        return self._str

    @property
    def str(self):
        return self._str

    @property
    def resource_location(self):
        return self._resource_loc

    @property
    def name(self):
        return self._name

    @property
    def properties(self):
        return self._properties

    def __eq__(self, other):
        if isinstance(other, Blockstate):
            for attr in self.comparable_attributes:
                if getattr(self, attr) != getattr(other, attr):
                    return False
            return True
        elif isinstance(other, (str, unicode)):
            return self._str == other
        else:
            return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    @staticmethod
    def buildStr(resource_loc, name, properties):
        _str = '{}:{}'.format(resource_loc, name)
        if properties:
            _str += '['
            for (key, value) in properties.iteritems():
                if key.startswith('<'):
                    continue
                _str += '{}={},'.format(key, value)
            _str = _str[:-1] + ']'
        return _str

    @classmethod
    def fromString(cls, string):
        if ':' in string:
            resource_location, blockstate = string.split(':')
            sep = blockstate.split('[')
            if len(sep) == 1:
                return cls(resource_location, sep[0])
            base, props = sep
            properties = {}
            for prop in props[:-1].split(','):
                split = prop.split('=')
                properties[split[0]] = split[1]
            return cls(resource_location, base, properties)
        else:
            sep = string.split('[')
            if len(sep) == 1:
                return cls(sep[0])
            base, props = sep
            properties = {}
            for prop in props[:-1].split(','):
                split = prop.split('=')
                properties[split[0]] = split[1]
            return cls(name=base, properties=properties)

class Materials(object):
    defaultMapColor = (201, 119, 240, 255)

    def mapColor(self, item):
        return self._mapColors.get(item, self.defaultMapColor)

    def __init__(self, version):
        self._version = data_loader.MCVersion(os.path.join('new_mcver', version))
        self._blockstates = self._version.blocks
        self._blocks_by_name = {}

        self.types = {}

        for resource_location in self._blockstates:
            for key, value in self._blockstates[resource_location].iteritems():
                if value.get('properties', []):
                    for property_set in value['properties']:
                        props = self._stripExtraData(property_set)
                        blockstate = Blockstate(resource_location, key, props)
                        self._blocks_by_name[blockstate.str] = blockstate
                else:
                    blockstate = Blockstate(resource_location, key)
                    self._blocks_by_name[blockstate.str] = blockstate

        length = len(self._blocks_by_name)
        self.lightEmission = np.zeros(length, dtype='uint8')
        self._mapColors = {}

        #print(self._blockstates['minecraft'])
        for blockstate in self._blocks_by_name.itervalues():
            self._mapColors[blockstate.str] = self.getExtraData(blockstate).get('mapcolor', self.defaultMapColor)

    def getExtraData(self, blockstate):
        base_state = self._blockstates.get(blockstate.resource_location, {}).get(blockstate.name, {})
        if blockstate.properties and base_state.get('properties', []):
            state = {}
            for property_set in base_state['properties']:
                for key, value in property_set.iteritems():
                    if key.startswith('<'):
                        continue
                    if blockstate.properties.get(key, None) == value:
                        state = property_set
                    else:
                        state = {}
            return state
        else:
            return base_state

    def _stripExtraData(self, d):
        new = {}
        for key, value in d.iteritems():
            if key.startswith('<'):
                continue
            new[key] = value
        return new

    def __getitem__(self, item):
        if isinstance(item, (str, unicode)):
            return self._blocks_by_name.get(item, Blockstate.fromString(item))
        elif isinstance(item, (tuple, list)):
            block_id, block_data = item
            for resource_loc in self._blockstates.keys():
                for name, blockstate in self._blockstates[resource_loc].iteritems():
                    if block_id == blockstate['id']:
                        for sub_blockstate in blockstate.get('properties', []):
                            if block_data == sub_blockstate['<data>']:
                                props = self._stripExtraData(sub_blockstate)
                                return self[Blockstate.buildStr(resource_loc, name, props)]
                        else:
                            return self[Blockstate.buildStr(resource_loc, name, {})]
            raise KeyError('Could not find any Block state with (id, data): {}'.format(item))
        elif isinstance(item, int):
            return self[(item,0)]

if __name__ == '__main__':
    mats = Materials('1.11')
    print(mats['minecraft:stone'])
    print(mats['mod:test'])
    print(mats['minecraft:stone'])
    print(mats[(0,0)].str)
    print(mats[0])
    print(mats['minecraft:chain_command_block[facing=south,conditional=true]'])
    print(mats.mapColor('minecraft:stone'))
