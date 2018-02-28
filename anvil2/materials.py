from __future__ import print_function, unicode_literals
import glob
import json
import os
import nbt


class Blockstate(object):

    __comp_attributes = ('_resource_location', '_basename', '_properties')

    def __init__(self, resource_location='minecraft', basename='air', properties=None):
        self._resource_location = resource_location
        self._basename = basename
        if properties:
            self._properties = properties
        else:
            self._properties = properties = {}
        self._str = Blockstate.__buildStr(resource_location, basename, properties)
        #self._block_state_map[self._str] = self

    def __repr__(self):
        return self._str

    def __eq__(self, other):
        if isinstance(other, Blockstate):
            result = True
            for attr in self.__comp_attributes:
                result = result and getattr(other, attr) == getattr(self, attr)
            return result
        elif isinstance(other, (str, unicode)):
            return other == self._str
        else:
            return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def toNBT(self):
        root = nbt.TAG_Compound()
        root['Name'] = nbt.TAG_String('{}:{}'.format(self._resource_location, self._basename))
        if self._properties:
            props = nbt.TAG_Compound()
            for (key, value) in self._properties.iteritems():
                props[key] = nbt.TAG_String(value)
            root['Properties'] = props
        return root


    @staticmethod
    def __buildStr(resource_location, basename, properties):
        _str = '{}:{}'.format(resource_location, basename)
        if properties:
            _str += '['
            for (key, value) in properties.iteritems():
                _str += '{}={},'.format(key, value)
            _str = _str[:-1] + ']'
        return _str

    @staticmethod
    def __decompStr(string):
        seperated = string.split("[")

        if len(seperated) == 1:
            if not seperated[0].startswith("minecraft:"):
                seperated[0] = "minecraft:" + seperated[0]
            return seperated[0], {}

        name, props = seperated

        if not name.startswith("minecraft:"):
            name = "minecraft:" + name

        properties = {}

        props = props[:-1]
        props = props.split(",")
        for prop in props:
            prop = prop.split("=")
            properties[prop[0]] = prop[1]
        return name, properties

    @classmethod
    def getBlockstateFromNBT(cls, nbt_data):
        resource, base = nbt_data['Name'].value.split(':')
        props = {key:value.value for (key, value) in nbt_data.get('Properties', {}).iteritems()}
        return cls(resource, base, props)

    @classmethod
    def getBlockstateFromData(cls, resource='minecraft', basename='air', properties=None):
        if not properties:
            properties = {}
        return cls(resource, basename, properties)

class BlockstateMaterials(object):

    def __init__(self):
        self.blockstates = [Blockstate(), ]
        self.load()

    def load(self):
        for f in glob.glob(os.path.join('blockstates', '*.json')):
            name = os.path.basename(f).replace('.json', '')
            fp = open(f)
            block_json = json.load(fp)
            fp.close()
            if 'multipart' in block_json: # TODO: Add support for multipart blocks
                continue
            blockstates = block_json['variants'].keys()
            for blockstate in blockstates:
                if blockstate == 'normal':
                    self.blockstates.append(Blockstate(basename=name))
                    continue
                elif blockstate == 'map':
                    continue
                serialized_props = blockstate.split(',')
                props = {}
                for prop in serialized_props:
                    key, value = prop.split('=')
                    props[key] = value
                #for prop in serialized_props:

                self.blockstates.append(Blockstate(basename=name, properties=props))


    def __getitem__(self, item):
        if isinstance(item, int):
            return self.blockstates[item]
        elif isinstance(item, (str, unicode)):
            for blockstate in self.blockstates:
                if blockstate == item:
                    return blockstate
        elif isinstance(item, nbt.TAG_Compound):
            return NotImplementedError()
        return None