
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
