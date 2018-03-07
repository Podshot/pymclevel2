from __future__ import unicode_literals, print_function

import collections
import os
import struct
import weakref
import zlib
from uuid import UUID

import math
import numpy as np
import glob
import time

import nbt
from materials import BlockstateMaterials, Blockstate
import api
from box import BoundingBox

SECTOR_BYTES = 4096
SECTOR_INTS = SECTOR_BYTES / 4
CHUNK_HEADER_SIZE = 5
VERSION_GZIP = 1
VERSION_DEFLATE = 2

def decodeBlockstateArray(array):
    return_value = [0] * 4096
    bit_per_index = len(array) * 64 / 4096
    print('Bit per Index (R)', bit_per_index)
    current_reference_index = 0

    for i in xrange(len(array)):
        current = array[i]

        overhang = (bit_per_index - (64 * i) % bit_per_index) % bit_per_index
        #print('Overhang', overhang)
        if overhang > 0:
            return_value[current_reference_index - 1] |= current % ((1 << overhang) << (bit_per_index - overhang))
        current >>= overhang
        #print('Current', current)
        #print('Curr', current >> overhang)

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

    def boolean(num):
        return num != 0 and (num & (num - 1)) == 0x0
    return_value = [0] * 4096
    bit_per_index = max(4, 6)
    print('if', len(array) if boolean(len(array)) else sequence_shift(len(array)))
    print('sequence shift', sequence_shift(len(array)))
    print('Bit per index (S)', bit_per_index)

class BlockstateWorld(api.World):

    SizeOnDisk = api.TagProperty('SizeOneDick', nbt.TAG_Long, 0)
    RandomSeed = api.TagProperty('RandomSeed', nbt.TAG_Long, 0)
    Time = api.TagProperty('Time', nbt.TAG_Long, 0)
    DayTime = api.TagProperty('DayTime', nbt.TAG_Long, 0)
    LastPlayed = api.TagProperty('LastPlayed', nbt.TAG_Long, lambda self: long(time.time() * 1000))
    LevelName = api.TagProperty('LevelName', nbt.TAG_String, lambda self: self.displayName)
    GeneratorName = api.TagProperty('generatorName', nbt.TAG_String, 'default')
    MapFeatures = api.TagProperty('MapFeatures', nbt.TAG_Byte, 1)
    GameType = api.TagProperty('GameType', nbt.TAG_Int, 0)

    Height = 256

    def __init__(self, path):
        super(BlockstateWorld, self).__init__(path)
        self.players = []
        self.player_cache = {}

        self.initTime = -1
        self.lockAcquireFuncs = []
        self.acquireSessionLock()

        self._materials = BlockstateMaterials()

        self._loadedChunks = weakref.WeakValueDictionary()
        self._loadedChunkData = {}
        self.recentChunks = collections.deque(maxlen=20)
        self.chunkNeedingLighting = set()
        self._allChunks = set()
        self.dimensions = {}
        self.regionFiles = {}
        self._bounds = None

        self.loadLevelDat()

        self.loadPlayers()

        self.preloadDimensions()

    @property
    def gamePlatform(self):
        return 'Java'

    @property
    def levelFormat(self):
        return 'anvil2'

    @property
    def materials(self):
        return self._materials

    @property
    def allChunks(self):
        if not self._allChunks:
            self.preloadChunks()
        return self._allChunks.__iter__()

    @property
    def chunkCount(self):
        if not self._allChunks:
            self.preloadChunks()
        return len(self._allChunks)

    @property
    def bounds(self):
        if not self._bounds:
            self._bounds = self.getWorldBounds()
        return self._bounds

    @property
    def size(self):
        return self.bounds.size

    def getWorldBounds(self):
        if self.chunkCount == 0:
            return BoundingBox((0,0,0), (0,0,0))

        allChunks = np.array(list(self.allChunks))
        min_cx = (allChunks[:, 0]).min()
        max_cx = (allChunks[:, 0]).max()
        min_cz = (allChunks[:, 1]).min()
        max_cz = (allChunks[:, 1]).max()

        origin = (min_cx << 4, 0, min_cz << 4)
        size = ((max_cx - min_cx + 1) << 4, self.Height, (max_cz - min_cz + 1) << 4)

        return BoundingBox(origin, size)

    def containsChunk(self, cx, cz):
        bounds = self.bounds
        return ((bounds.mincx <= cx < bounds.maxcx) and
                (bounds.mincz <= cz < bounds.maxcz))

    def containsPoint(self, x, y, z):
        return (x, y, z) in self.bounds

    def _getSlices(self, box):
        if box == self.bounds:
            print('All chunks selected! Selecting {} chunks instead of {}'.format(self.chunkCount, box.chunkCount))
            y = box.miny
            slices = slice(0,16), slice(0, 16), slice(0, box.maxy)

            def getAllSlices():
                for cPos in self.allChunks:
                    x, z = cPos
                    x *= 16
                    z *= 16
                    x -= box.minx
                    z -= box.minz
                    yield cPos, slices, (x, y, z)
            return getAllSlices()
        return api.getSlices(box, self.Height)

    def getChunkSlices(self, box):
        for cPos, slices, point in self._getSlices(box):
            if self.containsChunk(*cPos):
                yield (self.getChunk(*cPos), slices, point)

    def getAllChunkSlices(self):
        slices = (slice(None), slice(None), slice(None))
        box = self.bounds
        x, y, z = box.origin

        for cPos in self.allChunks:
            cx, cz = cPos
            try:
                chunk = self.getChunk(cx, cz)
            except (api.ChunkMalformed, api.ChunkNotPresent):
                continue
            yield (chunk, slices, (cx * 16 - x, 0, cz * 16 - z))

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
        dimensions = glob.glob(os.path.join(self.path, 'DIM*', ''))
        for dimension in dimensions:
            try:
                dimNum = int(os.path.basename(os.path.dirname(dimension))[3:])
                #dim = BedrockDimension(self, dimNum) # TODO: Create dimension
                dim = None
                self.dimensions[dimNum] = dim
            except Exception as e:
                pass

    def dirhash(self, n):
        return self.dirhashes[n % 64]

    def _dirhash(self):
        n = self
        n %= 64
        s = u""
        if n >= 36:
            s += u"1"
            n -= 36
        s += u"0123456789abcdefghijklmnopqrstuvwxyz"[n]

        return s

    dirhashes = [_dirhash(n) for n in xrange(64)]

    def getRegionForChunk(self, cx, cz):
        """
        :return: The region for the chunk
        :rtype: BlockstateRegionFile
        """
        rx = cx >> 5
        rz = cz >> 5
        return self.getRegionFile(rx, rz)

    def getRegionFile(self, rx, rz):
        region = self.regionFiles.get((rx, rz))
        if region:
            return region
        region = BlockstateRegionFile(self, os.path.join(self.path, 'region', 'r.{}.{}.mca'.format(rx, rz)))
        self.regionFiles[rx, rz] = region
        return region

    def getChunk(self, cx, cz):
        chunk = self._loadedChunks.get((cx,cz))
        if chunk:
            return chunk
        region = self.getRegionForChunk(cx,cz)
        chunk = region.getChunk(cx,cz)

        self._loadedChunks[cx, cz] = chunk
        self.recentChunks.append(chunk)
        return chunk

    def heightMapAt(self, x, z):
        cx = x >> 4
        cz = z >> 4
        xInChunk = x & 0xf
        zInChunk = z & 0xf

        chunk = self.getChunk(cx, cz)

        return chunk.HeightMap[zInChunk, xInChunk]

    def biomeAt(self, x, z):
        cx = x >> 4
        cz = z >> 4
        xInChunk = x & 0xf
        zInChunk = z & 0xf

        chunk = self.getChunk(cx, cz)

        return int(chunk.Biomes[(z - zInChunk) * 16 + (x - xInChunk)])

    def setBiomeAt(self, x, z, biomeID):
        cx = x >> 4
        cz = z >> 4
        xInChunk = x & 0xf
        zInChunk = z & 0xf

        chunk = self.getChunk(cx, cz)
        chunk.Biomes[(z - zInChunk) * 16 + (x - xInChunk)] = biomeID

    def addEntity(self, tag):
        if isinstance(tag, nbt.TAG_Compound):
            x, y, z = tag['Pos'][0].value, tag['Pos'][1].value, tag['Pos'][2].value
            x, y, z = map(lambda i: int(math.floor(i)), (x, y, z))
            cx = x >> 4
            cz = z >> 4

            chunk = self.getChunk(cx, cz)
            chunk.addEntity(tag)
            chunk.dirty = True
        else:
            raise ValueError('Entity tag must be a TAG_Compound')

    def tileEntityAt(self, x, y, z):
        cx = x >> 4
        cz = z >> 4
        chunk = self.getChunk(cx, cz)
        return chunk.tileEntityAt(x, y, z)

    def addTileEntity(self, tag):
        if isinstance(tag, nbt.TAG_Compound):
            x, y, z = tag['x'].value, tag['y'].value, tag['z'].value
            cx = x >> 4
            cz = z >> 4

            chunk = self.getChunk(cx, cz)
            chunk.addTileEntity(tag)
            chunk.dirty = True
        else:
            raise ValueError('Tile Entity tag must be a TAG_Compound')

    def addTileTick(self, tag):
        if isinstance(tag, nbt.TAG_Compound):
            x, y, z = tag['x'].value, tag['y'].value, tag['z'].value
            cx = x >> 4
            cz = z >> 4

            chunk = self.getChunk(cx, cz)
            chunk.addTileTick(tag)
            chunk.dirty = True
        else:
            raise ValueError('Tile Tick tag must be a TAG_Compound')

    def findAllChunksInRegions(self):
        chunks = set()
        regions = glob.glob(os.path.join(self.path, 'region', '*.mca'))
        for region in regions:
            region = BlockstateRegionFile(self, region)
            if region.offsets.any():
                self.regionFiles[region.rx, region.rz] = region

                for index, offset in enumerate(region.offsets):
                    if offset:
                        cx = index & 0x1f
                        cz = index >> 5

                        cx += region.rx << 5
                        cz += region.rz << 5

                        chunks.add((cx, cz))
        return chunks

    def preloadChunks(self):
        self._allChunks.update(self.findAllChunksInRegions())
        with api.work_directory(self):
            self._allChunks.update(self.findAllChunksInRegions())

class BlockstateRegionFile(api.RegionFile):

    CHUNK_HEADER_SIZE = 5

    length_struct = struct.Struct('>I')
    format_struct = struct.Struct('B')

    def __init__(self, world, path):
        super(BlockstateRegionFile, self).__init__(world, path)
        self._free_sectors = []
        self._offsets = None
        self._modification_times = None
        self._file_size = -1
        self.rx, self.rz = map(int, os.path.basename(path).split('.')[1:3])

        self.load()

    @property
    def offsets(self):
        return self._offsets

    def getOffset(self, cx, cz):
        return self.offsets[(cx & 0x1f) + (cz & 0x1f) * 32]

    def setOffset(self, cx, cz, offset):
        self.offsets[(cx & 0x1f) + (cz & 0x1f) * 32] = offset
        with open(self._path, 'rb+') as f:
            f.seek(0)
            f.write(self.offsets.tostring())

    def getChunk(self, cx, cz):
        #if (cx, cz) in self._chunks:
        #    return self._chunks[cx, cz]
        #else:
        #    chunk = self._getChunkFromFile(cx, cz)
        #    if chunk:
        #        self._chunks[cx, cz] = chunk
        #    return chunk
        return self._getChunkFromFile(cx, cz)

    def saveChunk(self, cx, cz, chunk_data):
        compressed_data = zlib.compress(chunk_data, 2)
        try:
            self._saveChunk(cx, cz, compressed_data)
        except api.ChunkTooBig as e:
            raise api.ChunkTooBig('{} ({} uncompressed)'.format(e.message, len(chunk_data)))

    def _saveChunk(self, cx, cz, chunk_data):
        cx &= 0x1f
        cz &= 0x1f
        offset = self.offsets[(cx & 0x1f) + (cz & 0x1f) * 32]

        sectorNumber = offset >> 8
        sectorsAllocated = offset & 0xff
        sectorsNeeded = (len(chunk_data) + self.CHUNK_HEADER_SIZE) / SECTOR_BYTES + 1

        if sectorsNeeded >= 256:
            raise api.ChunkTooBig('Chunk too big! {} bytes exceeds 1MB'.format(len(chunk_data)))

        if sectorNumber != 0 and sectorsAllocated >= sectorsNeeded:
            self.writeSector(sectorNumber, chunk_data)
        else: # Allocate new sectors
            for i in xrange(sectorNumber, sectorNumber + sectorsAllocated):
                self._free_sectors[i] = True

            runLength = 0
            runStart = 0

            try:
                runStart = self._free_sectors.index(True)

                for i in xrange(runStart, len(self._free_sectors)):
                    if runLength:
                        if self._free_sectors[i]:
                            runLength += 1
                        else:
                            runLength = 0
                    elif self._free_sectors[i]:
                        runStart = i
                        runLength = 1

                    if runLength >= sectorsNeeded:
                        break
            except ValueError:
                pass

            if runLength >= sectorsNeeded:
                sectorNumber = runStart
                self.setOffset(cx, cz, sectorNumber << 8 | sectorsNeeded)
                self.writeSector(sectorNumber, chunk_data)
                self._free_sectors[sectorNumber:sectorNumber + sectorsNeeded] = [False] * sectorsNeeded
            else:
                with open(self._path, 'rb+') as f:
                    f.seek(0, 2)
                    filesize = f.tell()

                    sectorNumber = len(self._free_sectors)

                    if sectorNumber * SECTOR_BYTES != filesize:
                        raise ValueError('sectorNumber * sector bytes doesn\'t equal current filesize')

                    filesize += sectorsNeeded * SECTOR_BYTES
                    f.truncate(filesize)

                self._free_sectors += [False] * sectorsNeeded

                self.setOffset(cx, cz, sectorNumber << 8 | sectorsNeeded)
                self.writeSector(sectorNumber, chunk_data)
        self.setTimestamp(cx, cz)

    def writeSector(self, sectorNumber, chunk_data):
        with open(self._path, 'rb+') as f:
            f.seek(sectorNumber * SECTOR_BYTES)
            f.write(struct.pack('>I', len(chunk_data) + 1)) # Chunk length
            f.write(struct.pack('B', 2))
            f.write(chunk_data)

    def setTimestamp(self, cx, cz, timestamp=None):
        if not timestamp:
            timestamp = time.time()

        self._modification_times[(cx & 0x1f) + (cz & 0x1f) * 32] = timestamp
        with open(self._path, 'rb+') as f:
            f.seek(SECTOR_BYTES)
            f.write(self._modification_times.tostring())

    def _getChunkFromFile(self, cx, cz):

        fp = open(self._path, 'rb+')

        cx &= 0x1f
        cz &= 0x1f

        chunk_offset = self.getOffset(cx, cz)
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
        return BlockstateChunk(self.world, self, nbt.load(buf=readable_data))


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

class BlockstateChunk(api.Chunk):

    def __init__(self, world, region, nbt_data):
        super(BlockstateChunk, self).__init__(world, nbt_data)
        self.region = region
        self.cx, self.cz = nbt_data['Level']['xPos'].value, nbt_data['Level']['zPos'].value
        self._data_version = nbt_data['DataVersion'].value
        self._entities = [e for e in nbt_data['Level']['Entities']]
        self._tile_entities = [te for te in nbt_data['Level']['TileEntities']]
        self._tile_ticks = [tt for tt in nbt_data['Level'].get('TileTicks', [])]
        self._biomes = nbt_data['Level']['Biomes'].value
        #self._biomes = np.reshape(self._biomes, (16,16)) TODO: Enable if Biomes can be read as a 2D array
        self._height_map = nbt_data['Level']['HeightMap'].value
        self._height_map = np.reshape(self._height_map, (16,16))
        self._sections = {}
        self._blocks = np.full((16,255,16), self.world.materials['minecraft:air'], dtype=Blockstate)
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
        root = self._nbt

        root['Level']['Biomes'] = nbt.TAG_Byte_Array(self._biomes)
        root['Level']['HeightMap'] = nbt.TAG_Int_Array(self._height_map.ravel())

        tile_entities = nbt.TAG_List()
        for te in self._tile_entities:
            tile_entities.append(te)
        root['Level']['TileEntities'] = tile_entities

        entities = nbt.TAG_List()
        for e in self._entities:
            entities.append(e)
        root['Level']['Entities'] = entities

        sections = nbt.TAG_List()
        for section in self._sections.itervalues():
            sections.append(section.toNBT())
        root['Level']['Sections'] = sections

        self.region.saveChunk(self.cx, self.cz, root.save(compressed=False))

    @property
    def nbt(self):
        return self._nbt['Level']

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

        self.__old_blockstates = data['BlockStates'].value
        encoded_states = [long(int(n)) for n in data['BlockStates'].value]
        decoded_states = decodeBlockstateArray(encoded_states)
        self._block_states = np.array(decoded_states).reshape((16,16,16))
        self._block_states = np.swapaxes(np.swapaxes(self._block_states, 0, 1), 0, 2)

        temp_array = np.swapaxes(np.swapaxes(self._block_states, 2, 0), 1, 0)
        temp_array = temp_array.ravel()
        temp_array = [long(int(n)) for n in temp_array]
        #print(temp_array)
        #print(decoded_states)
        print('Array to save', encodeBlockstateArray(temp_array))
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

    def toNBT(self):
        root = nbt.TAG_Compound()
        root['BlockStates'] = nbt.TAG_Long_Array(self.__old_blockstates)
        root['Y'] = nbt.TAG_Byte(self._y)
        root['BlockLight'] = nbt.TAG_Byte_Array(self._block_light)
        root['SkyLight'] = nbt.TAG_Byte_Array(self._sky_light)
        palette = nbt.TAG_List()
        for block in self.palette:
            palette.append(block.toNBT())
        root['Palette'] = self._palette
        return root

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
    #obj = BlockstateRegionFile('C:\\Users\\gotharbg\\Documents\\MC Worlds\\1.13 World\\region\\r.0.0.mca')
    #obj2 = BlockstateRegionFile('C:\\Users\\gotharbg\\Downloads\\Podshot 1_13 Snapshot\\region\\r.0.0.mca')

    #
    world = BlockstateWorld('C:\\Users\\gotharbg\\Documents\\Python Projects\\WorldRewriteProto\\tests\\1.13 World')
    save_test = BlockstateRegionFile(world, 'C:\\Users\\gotharbg\\Documents\\Python Projects\\WorldRewriteProto\\tests\\1.13 World\\r.0.0.mca')
    chunk = save_test.getChunk(0,0)
    te = nbt.TAG_Compound()
    te['id'] = nbt.TAG_String('bed')
    te['x'] = nbt.TAG_Int(1)
    te['y'] = nbt.TAG_Int(5)
    te['z'] = nbt.TAG_Int(1)
    te['color'] = nbt.TAG_Int(15)
    chunk.TileEntities.append(te)
    chunk.save()

    #chunk = obj.getChunk(0,0)
    #print(chunk.Blocks[0,3,0])
    #print(chunk.TileEntities)
    #chunk.TileEntities.append({'test': 'value'})
    #chunk2 = obj2.getChunk(0,0)
    #print(chunk.TileEntities)
    #for i in xrange(16):
    #    print(chunk.Blocks[:,i,1])

    #chunk.Blocks[1,0,1] = Blockstate.getBlockstateFromData('mod', 'mod_block')
    #print(chunk.Blocks[1,0,1])
    #no_block = Blockstate.getBlockstateFromData('minecraft', 'sponge')
    #print(no_block in chunk.Blocks)
    #print('minecraft:air' in chunk.Blocks)
    #print(chunk._total_palette)
    #print('===')
    #print(chunk2._total_palette)
    #mats = BlockstateMaterials()
    #print(mats['minecraft:air'])
    #print(mats[0])
    #print(mats['minecraft:grass_block[snowy=true]'])
