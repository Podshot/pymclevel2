from __future__ import unicode_literals
from contextlib import contextmanager
import os

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

@contextmanager
def work_directory(world):
    old_path = world.path
    world.path = os.path.join(old_path, '##MCEDIT.TEMP##')
    if not os.path.exists(world.path): os.mkdir(world.path)
    yield
    world.path = old_path

def getSlices(box, height):
    """ call this method to iterate through a large slice of the world by
        visiting each chunk and indexing its data with a subslice.

    this returns an iterator, which yields 3-tuples containing:
    +  a pair of chunk coordinates (cx, cz),
    +  a x,z,y triplet of slices that can be used to index the AnvilChunk's data arrays,
    +  a x,y,z triplet representing the relative location of this subslice within the requested world slice.

    Note the different order of the coordinates between the 'slices' triplet
    and the 'offset' triplet. x,z,y ordering is used only
    to index arrays, since it reflects the order of the blocks in memory.
    In all other places, including an entity's 'Pos', the order is x,y,z.
    """

    # when yielding slices of chunks on the edge of the box, adjust the
    # slices by an offset
    minxoff, minzoff = box.minx - (box.mincx << 4), box.minz - (box.mincz << 4)
    maxxoff, maxzoff = box.maxx - (box.maxcx << 4) + 16, box.maxz - (box.maxcz << 4) + 16

    newMinY = 0
    if box.miny < 0:
        newMinY = -box.miny
    miny = max(0, box.miny)
    maxy = min(height, box.maxy)

    for cx in xrange(box.mincx, box.maxcx):
        localMinX = 0
        localMaxX = 16
        if cx == box.mincx:
            localMinX = minxoff

        if cx == box.maxcx - 1:
            localMaxX = maxxoff
        newMinX = localMinX + (cx << 4) - box.minx

        for cz in xrange(box.mincz, box.maxcz):
            localMinZ = 0
            localMaxZ = 16
            if cz == box.mincz:
                localMinZ = minzoff
            if cz == box.maxcz - 1:
                localMaxZ = maxzoff
            newMinZ = localMinZ + (cz << 4) - box.minz
            slices, point = (
                (slice(localMinX, localMaxX), slice(localMinZ, localMaxZ), slice(miny, maxy)),
                (newMinX, newMinY, newMinZ)
            )

            yield (cx, cz), slices, point

class Chunk(object):

    def __init__(self, world, nbt_data):
        self.world = world
        self._nbt = nbt_data

    def save(self):
        raise NotImplementedError()

    @property
    def HeightMap(self):
        raise NotImplementedError()

    @property
    def Biomes(self):
        raise NotImplementedError()

    @property
    def Entities(self):
        raise NotImplementedError()

    @property
    def TileEntities(self):
        raise NotImplementedError()

    @property
    def TileTicks(self):
        raise NotImplementedError()

    @property
    def DataVersion(self):
        raise NotImplementedError()

    @property
    def Sections(self):
        raise NotImplementedError()

    @property
    def Blocks(self):
        raise NotImplementedError()

class RegionFile(object):

    def __init__(self, world, path):
        self.world = world
        self._path = path
        self._chunks = {}

    def getChunk(self, cx, cz):
        raise NotImplementedError()

    def _getChunkFromFile(self, cx, cz):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

class World(object):

    def __init__(self, path):
        self.path = path

    @property
    def gamePlatform(self):
        raise NotImplementedError()

    @property
    def levelFormat(self):
        raise NotImplementedError()

    def loadLevelDat(self):
        raise NotImplementedError()

    def loadPlayers(self):
        raise NotImplementedError()

    def preloadDimensions(self):
        raise NotImplementedError()

    def getRegionForChunk(self, cx, cz):
        raise NotImplementedError()

    def getRegionFile(self, rx, rz):
        raise NotImplementedError()

    def getChunk(self, cx, cz):
        raise NotImplementedError()

    def heightMapAt(self, x, z):
        raise NotImplementedError()

    def biomeAt(self, x, z):
        raise NotImplementedError()

    def setBiomeAt(self, x, z, biomeID):
        raise NotImplementedError()

    def addEntity(self, tag):
        raise NotImplementedError()

    def tileEntityAt(self, x, y, z):
        raise NotImplementedError()

    def addTileEntity(self, tag):
        raise NotImplementedError()

    def addTileTick(self, tag):
        raise NotImplementedError()

class ChunkNotPresent(Exception):
    pass

class RegionMalformed(Exception):
    pass

class ChunkMalformed(ChunkNotPresent):
    pass

class ChunkTooBig(ValueError):
    pass