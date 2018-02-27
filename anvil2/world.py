from __future__ import unicode_literals, print_function

import collections
import json
import os
import struct
import zlib
from uuid import UUID

import numpy as np
import cPickle
import glob

import time

import nbt

SECTOR_BYTES = 4096
SECTOR_INTS = SECTOR_BYTES / 4
CHUNK_HEADER_SIZE = 5
VERSION_GZIP = 1
VERSION_DEFLATE = 2

def decodeBlockstateArray(array):
    return_value = [0] * 4096
    bit_per_index = len(array) * 64 / 4096
    print('Bit per Index', bit_per_index)
    current_reference_index = 0

    for i in xrange(len(array)):
        current = array[i]

        overhang = (bit_per_index - (64 * i) % bit_per_index) % bit_per_index
        print('Overhang', overhang)
        if overhang > 0:
            return_value[current_reference_index - 1] |= current % ((1 << overhang) << (bit_per_index - overhang))
        current >>= overhang
        print('Current', current)
        print('Curr', current >> overhang)

        remaining_bits = 64 - overhang
        for j in xrange((remaining_bits + (bit_per_index - remaining_bits % bit_per_index) % bit_per_index) / bit_per_index):
            return_value[current_reference_index] = current % (1 << bit_per_index)
            current_reference_index += 1
            current >>= bit_per_index
    return return_value

def encodeBlockstateArray(array):

    def sequence_shift(num):
        num -= 1
        for i in xrange(5):
            num |= num >> (1 << i)
        return num + 1
    return_value = [0] * 4096
    bit_per_index = max(4, 6)

def TagProperty(tagName, tagType, default_or_func=None):
    def getter(self):
        if tagName not in self.root_tag["Data"]:
            if hasattr(default_or_func, "__call__"):
                default = default_or_func(self)
            else:
                default = default_or_func

            self.root_tag["Data"][tagName] = tagType(default)
        return self.root_tag["Data"][tagName].value

    def setter(self, val):
        self.root_tag["Data"][tagName] = tagType(value=val)

    return property(getter, setter)

class Blockstate(object):

   # _block_state_map = {}
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
        built = Blockstate.__buildStr(resource, base, props)
        #if built in cls._block_state_map:
        #    return cls._block_state_map[built]
        return cls(resource, base, props)

    @classmethod
    def getBlockstateFromData(cls, resource='minecraft', basename='air', properties=None):
        if not properties:
            properties = {}
        built = Blockstate.__buildStr(resource, basename, properties)
        #if built in cls._block_state_map:
        #    return cls._block_state_map[built]
        return cls(resource, basename, properties)

    #@classmethod
    #def getBlockstateFromStr(cls, string):
        #if string in cls._block_state_map:
        #    return cls._block_state_map[string]
        #return cls()

class BlockstateLevel(object):

    SizeOnDisk = TagProperty('SizeOneDick', nbt.TAG_Long, 0)
    RandomSeed = TagProperty('RandomSeed', nbt.TAG_Long, 0)
    Time = TagProperty('Time', nbt.TAG_Long, 0)
    DayTime = TagProperty('DayTime', nbt.TAG_Long, 0)
    LastPlayed = TagProperty('LastPlayed', nbt.TAG_Long, lambda self: long(time.time() * 1000))
    LevelName = TagProperty('LevelName', nbt.TAG_String, lambda self: self.displayName)
    GeneratorName = TagProperty('generatorName', nbt.TAG_String, 'default')
    MapFeatures = TagProperty('MapFeatures', nbt.TAG_Byte, 1)
    GameType = TagProperty('GameType', nbt.TAG_Int, 0)

    def __init__(self, path):
        self.path = path
        self.players = []
        self.player_cache = {}

        self.initTime = -1
        self.lockAcquireFuncs = []
        self.acquireSessionLock()

        self._loadedChunks = None # TODO
        self._loadedChunkData = {}
        self.recentChunks = collections.deque(maxlen=20)
        self.chunkNeedingLighting = set()
        self._allChunks = None
        self.dimensions = {}

        self.loadLevelDat()

        self.loadPlayers()

        self.preloadDimensions()

    @classmethod
    def gamePlatform(cls):
        return 'Java'

    @classmethod
    def levelFormat(cls):
        return 'anvil2'

    def acquireSessionLock(self):
        lock_file = os.path.join(self.path, 'session.lock')
        self.initTime = int(time.time() * 1000)
        with open(lock_file, "wb") as f:
            f.write(struct.pack(">q", self.initTime))
            f.flush()
            os.fsync(f.fileno())

        for func in self.lockAcquireFuncs:
            func()

    def loadLevelDat(self):
        self.root_tag = nbt.load(os.path.join(self.path, 'level.dat'))
        self.gameVersion = self.root_tag['Data']['Version'].get('Name', nbt.TAG_String('Unknown')).value

    def loadPlayers(self):
        players = os.listdir(os.path.join(self.path, 'playerdata'))
        for p in players:
            if p.endswith('.dat'):
                try:
                    UUID(p[:-4], version=4)
                except ValueError:
                    continue
                self.players.append(p)

    def preloadDimensions(self):
        raise NotImplementedError()

class BlockstateRegionFile(object):

    length_struct = struct.Struct('>I')
    format_struct = struct.Struct('B')

    def __init__(self, path):
        self._path = path
        self._chunks = {}
        self._free_sectors = []
        self._offsets = None
        self._modification_times = None
        self._file_size = -1

        self.load()

    def getChunk(self, cx, cz):
        if (cx, cz) in self._chunks:
            return self._chunks[cx, cz]
        else:
            chunk = self._getChunkFromFile(cx, cz)
            if chunk:
                self._chunks[cx, cz] = chunk
            return chunk


    def _getChunkFromFile(self, cx, cz):

        fp = open(self._path, 'rb+')

        cx &= 0x1f
        cz &= 0x1f

        chunk_offset = self._offsets[(cx & 0x1f) + (cz & 0x1f) * 32]
        if chunk_offset == 0:
            #print('Chunk does not exist')
            return

        sector_start = chunk_offset >> 8
        sector_nums = chunk_offset & 0xff

        if sector_nums == 0:
            #print('Chunk does not exist')
            return

        if sector_start + sector_nums > len(self._free_sectors):
            #print('Chunk does not exist')
            return

        fp.seek(sector_start * SECTOR_BYTES)
        data = fp.read(sector_nums * SECTOR_BYTES)

        if len(data) < 5:
            print('Chunk/Sector is malformed')
            return

        length = struct.unpack_from('>I', data)[0]
        _format = struct.unpack_from('B', data, 4)[0]
        data = data[5:length + 5]

        readable_data = None
        if _format == VERSION_GZIP:
            readable_data = nbt.gunzip(data)
            # print 'Chunk is in GZIP format'
        if _format == VERSION_DEFLATE:
            # print 'Chunk is in DEFLATE format'
            readable_data = zlib.decompress(data)

        fp.close()
        return BlockstateChunk(nbt.load(buf=readable_data))


    def load(self):
        fp = open(self._path, 'rb+')

        file_size = os.path.getsize(self._path)
        if file_size & 0xfff:
            file_size = (file_size | 0xfff) + 1
            fp.truncate(file_size)

        if file_size == 0:
            file_size = SECTOR_BYTES * 2
            fp.truncate(file_size)

            self._file_size = file_size

        fp.seek(0)

        tmp_offsets = fp.read(SECTOR_BYTES)
        tmp_mod_times = fp.read(SECTOR_BYTES)

        self._free_sectors = [True] * (file_size / SECTOR_BYTES)
        self._free_sectors[0:2] = False, False

        offsets = np.fromstring(tmp_offsets, dtype='>u4')
        modification_times = np.fromstring(tmp_mod_times, dtype='>u4')

        self._offsets = offsets
        self._modification_times = modification_times

        for offset in offsets:
            sector = offset >> 8
            count = offset & 0xff

            for i in xrange(sector, sector + count):
                if i >= len(self._free_sectors):
                    print('Offset table went past EOF')
                    break
                    self._free_sectors[i] = False

        fp.close()

class BlockstateChunk(object):

    def __init__(self, nbt_data):
        self.cx, self.cz = nbt_data['Level']['xPos'].value, nbt_data['Level']['zPos'].value
        self._data_version = nbt_data['DataVersion'].value
        self._entities = [e for e in nbt_data['Level']['Entities']]
        self._tile_entities = [te for te in nbt_data['Level']['TileEntities']]
        self._tile_ticks = [tt for tt in nbt_data['Level'].get('TileTicks', [])]
        self._biomes = nbt_data['Level']['Biomes'].value
        self._height_map = nbt_data['Level']['HeightMap'].value
        self._sections = {}
        self._blocks = np.zeros((16,255,16), dtype=Blockstate)
        self._total_palette = []
        for section in nbt_data['Level']['Sections']:
            sect = BlockstateChunkSection(self, section)
            self._sections[sect.Y] = sect
            lower_bound = sect.Y << 4
            upper_bound = (sect.Y + 1) << 4
            self._blocks[:, lower_bound:upper_bound, :] = sect._blocks
            for block in sect.palette:
                if block not in self._total_palette:
                    self._total_palette.append(block)
        #print(self._blocks)
        #self._blocks = ChunkBlockWrapper(self._sections)

    def save(self):
        raise NotImplementedError()

    @property
    def HeightMap(self):
        return self._height_map

    @property
    def Biomes(self):
        return self._biomes

    @property
    def Entities(self):
        return self._entities

    @property
    def TileEntities(self):
        return self._tile_entities

    @property
    def TileTicks(self):
        return self._tile_ticks

    @property
    def DataVersion(self):
        return self._data_version

    @property
    def Sections(self):
        return self._sections

    @property
    def Blocks(self):
        return self._blocks

    @Blocks.getter
    def get_Blocks(self):
        return self._blocks

class BlockstateChunkSection(object):

    def __init__(self, parent, data):
        self._parent = parent
        self._y = data['Y'].value
        self._block_light = data['BlockLight'].value
        self._sky_light = data['SkyLight'].value
        self._palette = data['Palette'].value
        self.palette = [Blockstate.getBlockstateFromNBT(nbt_data) for nbt_data in self._palette]
        #self.palette = PaletteArrayWrapper(self._palette)
        #for blockstate in self._palette:
        #    Blockstate.getBlockstateFromNBT(blockstate)

        encoded_states = [long(int(n)) for n in data['BlockStates'].value]
        decoded_states = decodeBlockstateArray(encoded_states)
        self._block_states = np.array(decoded_states).reshape((16,16,16))
        self._block_states = np.swapaxes(np.swapaxes(self._block_states, 0, 1), 0, 2)

        temp_array = np.swapaxes(np.swapaxes(self._block_states, 2, 0), 1, 0)
        temp_array = temp_array.ravel()
        temp_array = [long(int(n)) for n in temp_array]
        #print(temp_array)
        #print(decoded_states)
        #print(encodeBlockstateArray(temp_array))
        #print('===', encoded_states == encodeBlockstateArray(temp_array))

        self._blocks = self._block_states.ravel().astype(object)
        for i in xrange(len(self._palette)):
            self._blocks[np.in1d(self._blocks, i)] = Blockstate.getBlockstateFromNBT(self._palette[i])
        self._blocks = np.reshape(self._blocks, (16,16,16))

    def recalculate_palette(self):
        updated_palette = [self._parent.materials[0],]
        for block in self._parent.materials:
            if block in self._blocks and block != 'minecraft:air':
                updated_palette.append(block)
        self._palette = updated_palette

    @property
    def Y(self):
        return self._y

    @property
    def BlockLight(self):
        return self._block_light

    @property
    def SkyLight(self):
        return self._sky_light

    @property
    def Blocks(self):
        return self.blocks

    @Blocks.getter
    def __getBlocks(self, pos):
        return pos

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

def identify(directory):
    if not (os.path.exists(os.path.join(directory, 'region')) or os.path.exists(os.path.join(directory, 'playerdata'))):
        return False
    if not (os.path.exists(os.path.join(directory, 'DIM1')) or os.path.exists(os.path.join(directory, 'DIM-1'))):
        return False
    if not (os.path.exists(os.path.join(directory, 'data')) or os.path.exists(os.path.join(directory, 'level.dat'))):
        return False
    root = nbt.load(os.path.join(directory, 'level.dat'))
    if 'FML' in root:
        return False
    if root.get('Data', nbt.TAG_Compound()).get('Version', nbt.TAG_Compound()).get('Id', nbt.TAG_Int(-1)).value < 1451:
        return False
    return True

if __name__ == '__main__':
    obj = BlockstateRegionFile('C:\\Users\\gotharbg\\Documents\\MC Worlds\\1.13 World\\region\\r.0.0.mca')
    obj2 = BlockstateRegionFile('C:\\Users\\gotharbg\\Downloads\\Podshot 1_13 Snapshot\\region\\r.0.0.mca')

    chunk = obj.getChunk(0,0)
    #print(chunk.Blocks[0,3,0])
    #print(chunk.TileEntities)
    chunk.TileEntities.append({'test': 'value'})
    chunk2 = obj2.getChunk(0,0)
    #print(chunk.TileEntities)
    for i in xrange(16):
        print(chunk.Blocks[:,i,1])

    chunk.Blocks[1,0,1] = Blockstate.getBlockstateFromData('mod', 'mod_block')
    print(chunk.Blocks[1,0,1])
    no_block = Blockstate.getBlockstateFromData('minecraft', 'sponge')
    print(no_block in chunk.Blocks)
    print('minecraft:air' in chunk.Blocks)
    print(chunk._total_palette)
    print('===')
    print(chunk2._total_palette)
    mats = BlockstateMaterials()
    print(mats['minecraft:air'])
    print(mats[0])
    print(mats['minecraft:grass_block[snowy=true]'])
    #print(chunk.Sections)
    #print(chunk.Sections[0]._block_states)
    #print(chunk.Sections[0].blocks[0,0,0])
    #chunk.Sections[0].blocks[0,0,0] = 10
    #print(chunk.Sections[0].palette[0])
    #print(chunk.Sections[0].palette['minecraft:dirt'])
    #print('Block:', chunk.Blocks[0,0,0])
    #print('Type', type(chunk.Sections[0].blocks[0,0,0]))
    #print('Block:', chunk.Blocks[0,16,0])
    #print('Slices:', chunk.Blocks[::2, ::2, :64])

    fp = open('text.cPickle', 'rb')
    data = cPickle.load(fp)
    fp.close()
    result = np.append(data, chunk.Sections[0]._block_states)
    result = result.reshape((16, 32, 16))
    #print(result.shape)
    #print(result[0,0,0])
    #print(result[0,15,0])
    #print(chunk.Sections[0].blocks[0,0,0] == chunk.Blocks[0,1,0])
    #print(chunk.Sections[0].blocks[0,0,0] == 'minecraft:bedrock')
#print(Blockstate.getBlockstate())