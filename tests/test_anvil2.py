import unittest
import os

import anvil2

class AttributeHolder(object):
    pass

class TestChunk(unittest.TestCase):

    def setUp(self):
        world = AttributeHolder()
        world.materials = anvil2.BlockstateMaterials()
        #setattr(world, 'materials', anvil2.BlockstateMaterials())
        self.region = anvil2.BlockstateRegionFile(world, os.path.join('1.13 World', 'region', 'r.0.0.mca'))

        self.chunk = self.region.getChunk(0,0)

    def test_get_chunk(self):
        chunk = self.region.getChunk(0,0)
        self.assertEqual(chunk.cx, 0)
        self.assertEqual(chunk.cz, 0)

    def test_types(self):
        self.assertIsInstance(self.chunk.Sections[0], anvil2.BlockstateChunkSection)

    def test_blocks(self):
        self.assertEqual(self.chunk.Blocks[0,0,0], 'minecraft:bedrock')

    def test_modification(self):
        self.chunk.Blocks[1,10,1] = 'minecraft:gold_block'
        self.assertEqual(self.chunk.Blocks[1,10,1], 'minecraft:gold_block')

class TestWorld(unittest.TestCase):

    def setUp(self):
        self.world = anvil2.BlockstateWorld(os.path.join('1.13 World'))

    def test_types(self):
        self.assertIsInstance(self.world.getChunk(0,0), anvil2.BlockstateChunk)
        self.assertIsInstance(self.world.getRegionForChunk(0,0), anvil2.BlockstateRegionFile)

    def test_identify(self):
        self.assertTrue(anvil2.identify(os.path.join('1.13 World')))

    def test_version(self):
        self.assertEqual(self.world.gameVersion, '17w50a')

    def test_heightmap(self):
        self.assertEqual(self.world.heightMapAt(0,0), 4)

    def test_biome(self):
        self.assertEqual(self.world.biomeAt(0,0), 1)
'''
class TestSections(unittest.TestCase):

    def setUp(self):
        self.region = anvil2.BlockstateRegionFile(os.path.join('1.13 World', 'region', 'r.0.0.mca'))

        self.chunk = self.region.getChunk(0, 0)
        self.section = self.chunk.Sections[0]

    def test_types(self):
        self.assertIsInstance(self.section, anvil2.BlockstateChunkSection)
        self.assertIsInstance(self.section.blocks, anvil2.BlockstateArrayWrapper)
        self.assertIsInstance(self.section.palette, anvil2.PaletteArrayWrapper)

    def test_blocks(self):
        self.assertEqual(len(self.chunk.Sections), 1)
        self.assertEqual(self.chunk.Sections[0].blocks[0,0,0], 'minecraft:bedrock') # Still working on Blockstate class, use str comparison for now

    def test_palette(self):
        self.assertEqual(self.chunk.Sections[0].palette[1], 'minecraft:bedrock')
        self.assertEqual(self.chunk.Sections[0].palette['minecraft:bedrock'], 1)
        self.assertEqual(self.chunk.Sections[0].palette['minecraft:air'], 0)
        self.assertEqual(self.chunk.Sections[0].palette['minecraft:grass_block[snowy=false]'], 3)

    @unittest.expectedFailure
    def test_modification(self):
        self.section.Blocks[1, 10, 1] = 'minecraft:gold_block'
        self.assertEqual(self.section.Blocks[1, 10, 1], 'minecraft:gold_block')
'''


def setup_suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestChunk))
    test_suite.addTest(unittest.makeSuite(TestWorld))
    #test_suite.addTest(unittest.makeSuite(TestSections))
    return test_suite

if __name__ == '__main__':
    suite = setup_suite()
    runner = unittest.TextTestRunner()
    runner.run(suite)