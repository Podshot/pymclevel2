from __future__ import unicode_literals, print_function
import os
import struct
import zlib
import numpy as np

import nbt

SECTOR_BYTES = 4096
SECTOR_INTS = SECTOR_BYTES / 4
CHUNK_HEADER_SIZE = 5
VERSION_GZIP = 1
VERSION_DEFLATE = 2

def decodeBlockstateArray(array):
    return_value = [0] * 4096
    bit_per_index = len(array) * 64 / 4096
    current_reference_index = 0

    for i in xrange(len(array)):
        current = array[i]

        overhang = (bit_per_index - (64 * i) % bit_per_index) % bit_per_index
        if overhang > 0:
            return_value[current_reference_index - 1] |= current % ((1 << overhang) << (bit_per_index - overhang))
        current >>= overhang

        remaining_bits = 64 - overhang
        for j in xrange((remaining_bits + (bit_per_index - remaining_bits % bit_per_index) % bit_per_index) / bit_per_index):
            return_value[current_reference_index] = current % (1 << bit_per_index)
            current_reference_index += 1
            current >>= bit_per_index
    return return_value

def encodeBlockstateArray(array):
    def egcd(a, b):
        if a == 0:
            return (b, 0, 1)
        else:
            g, y, x = egcd(b % a, a)
            return (g, x - (b // a) * y, y)

    def modinv(a, m):
        g, x, y = egcd(a, m)
        if g != 1:
            raise Exception('modular inverse does not exist')
        else:
            return x % m

    return_value = [0] * 4096
    bit_per_index = len(array) * 64/ 4096
    current_reference_index = 0

    for i in xrange(len(array)):
        current = array[i]

        overhang = modinv(bit_per_index - modinv(64 * i, bit_per_index), bit_per_index)
        if overhang > 0:
            return_value[current_reference_index - 1] |= modinv(current, ((1 << overhang) << (bit_per_index - overhang)))
        current >>= overhang

        remaining_bits = 64 - overhang
        for j in xrange((remaining_bits + (bit_per_index - modinv(modinv(remaining_bits, bit_per_index), bit_per_index) / bit_per_index))):
            return_value[current_reference_index] = modinv(current, (1 << bit_per_index))
            current_reference_index += 1
            current >>=bit_per_index

    return return_value

class Blockstate(object):

    _block_state_map = {}

    def __init__(self, resource_location='minecraft', basename='air', properties=None):
        self._resource_location = resource_location
        self._basename = basename
        if properties:
            self._properties = properties
        else:
            self._properties = properties = {}
        self._str = Blockstate.__buildStr(resource_location, basename, properties)
        self._block_state_map[self._str] = self

    def __repr__(self):
        return self._str

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
        if built in cls._block_state_map:
            return cls._block_state_map[built]
        return cls(resource, base, props)

    @classmethod
    def getBlockstateFromData(cls, resource='minecraft', basename='air', properties=None):
        if not properties:
            properties = {}
        built = Blockstate.__buildStr(resource, basename, properties)
        if built in cls._block_state_map:
            return cls._block_state_map[built]
        return cls(resource, basename, properties)

    @classmethod
    def getBlockstateFromStr(cls, string):
        if string in cls._block_state_map:
            return cls._block_state_map[string]
        #return cls()

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
        for section in nbt_data['Level']['Sections']:
            sect = BlockstateChunkSection(section)
            self._sections[sect.Y] = sect
        self._blocks = ChunkBlockWrapper(self._sections)

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

class BlockstateArrayWrapper(object):

    def __init__(self, base_array, palette):
        self._base = base_array
        self._palette = palette

    def __getitem__(self, item):
        index = self._base[item]
        return Blockstate.getBlockstateFromNBT(self._palette[index])

    def __setitem__(self, key, value):
        #self._palette)
        index = self._palette
        #print(key, value)

class PaletteArrayWrapper(object):

    def __init__(self, palette):
        self._palette = palette
        #self._internal_palette = []
        self._map = {}
        self.convertToBlockstates()

    def __getitem__(self, item):
        if isinstance(item, int):
            if item in self._map:
                return self._map[item]
            self._map[item] = block = Blockstate.getBlockstateFromNBT(self._palette[item])
            return block
        elif isinstance(item, (str, unicode)):
            for (index, block) in self._map.iteritems():
                if str(block) == item:
                    return index
            return -1

    def convertToBlockstates(self):
        for i in xrange(len(self._palette)):
            self._map[i] = Blockstate.getBlockstateFromNBT(self._palette[i])

    def converToNBT(self):
        result = nbt.TAG_List()
        for block in self._map.itervalues():
            result.append(block.toNBT())
        return result

class ChunkBlockWrapper(object):

    def __init__(self, sections):
        self._sections = sections

    def __getitem__(self, item):
        x, y, z = item
        sy = y >> 4
        if sy >= len(self._sections):
            raise NotImplementedError('Accessing non-existent Chunk sections is not supported yet!')
        return self._sections[sy].Blocks[x,y,z]

    def __setitem__(self, key, value):
        raise NotImplementedError('Setting Blocks is not supported yet!')

class BlockstateChunkSection(object):

    def __init__(self, data):
        self._y = data['Y'].value
        self._block_light = data['BlockLight'].value
        self._sky_light = data['SkyLight'].value
        self._palette = data['Palette'].value
        self.palette = PaletteArrayWrapper(self._palette)
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
        #print('===', encoded_states == encodeBlockstateArray(temp_array))
        self.blocks = BlockstateArrayWrapper(self._block_states, self._palette)

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

if __name__ == '__main__':
    obj = BlockstateRegionFile('C:\\Users\\gotharbg\\Documents\\MC Worlds\\1.13 World\\region\\r.0.0.mca')

    chunk = obj.getChunk(0,0)
    print(chunk.TileEntities)
    chunk.TileEntities.append({'test': 'value'})
    print(chunk.TileEntities)
    #print(chunk.Blocks[0,0,0])
    print(chunk.Sections)
    print(chunk.Sections[0]._block_states)
    print(chunk.Sections[0].blocks[0,0,0])
    chunk.Sections[0].blocks[0,0,0] = 10
    print(chunk.Sections[0].palette[0])
    print(chunk.Sections[0].palette['minecraft:dirt'])
    print('Block:', chunk.Blocks[0,0,0])
    print('Type', type(chunk.Sections[0].blocks[0,0,0]))
    print('Block:', chunk.Blocks[0,16,0])
#print(Blockstate.getBlockstate())