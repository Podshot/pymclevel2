import unittest
import os

import anvil2

class TestChunk(unittest.TestCase):

    def setUp(self):
        self.region = anvil2.BlockstateRegionFile(os.path.join('1.13 World', 'region', 'r.0.0.mca'))

        self.chunk = self.region.getChunk(0,0)

    def test_get_chunk(self):
        chunk = self.region.getChunk(0,0)
        self.assertEqual(chunk.cx, 0)
        self.assertEqual(chunk.cz, 0)

    def test_types(self):
        self.assertIsInstance(self.chunk.Sections[0], anvil2.BlockstateChunkSection)
        self.assertIsInstance(self.chunk.Sections[0].blocks, anvil2.BlockstateArrayWrapper)
        self.assertIsInstance(self.chunk.Sections[0].palette, anvil2.PaletteArrayWrapper)

    def test_blocks(self):
        self.assertEqual(str(self.chunk.Blocks[0,0,0]), 'minecraft:bedrock')

        with self.assertRaises(NotImplementedError):
            self.chunk.Blocks[0,16,0]
        with self.assertRaises(NotImplementedError):
            self.chunk.Blocks[1,1,1] = 'minecraft:stone'

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
        self.assertEqual(str(self.chunk.Sections[0].blocks[0,0,0]), 'minecraft:bedrock') # Still working on Blockstate class, use str comparison for now

        with self.assertRaises(NotImplementedError):
            self.chunk.Blocks[0, 16, 0]
        with self.assertRaises(NotImplementedError):
            self.chunk.Blocks[1, 1, 1] = 'minecraft:stone'

    def test_palette(self):
        self.assertEqual(str(self.chunk.Sections[0].palette[1]), 'minecraft:bedrock')
        self.assertEqual(self.chunk.Sections[0].palette['minecraft:bedrock'], 1)
        self.assertEqual(self.chunk.Sections[0].palette['minecraft:air'], 0)
        self.assertEqual(self.chunk.Sections[0].palette['minecraft:grass_block[snowy=false]'], 3)


def setup_suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestChunk))
    test_suite.addTest(unittest.makeSuite(TestSections))
    return test_suite

if __name__ == '__main__':
    suite = setup_suite()
    runner = unittest.TextTestRunner()
    runner.run(suite)