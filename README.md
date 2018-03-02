# pymclevel2

This is a complete rewrite of the [original pymclevel by Codewarrior0](https://github.com/mcedit/pymclevel). The rewrites
focus to make an easily extend-able library that can support multiple world formats without having to modify other world
formats for compatibility. This library is in it's alpha stages, if you would like to help, please refer to the TODO section
later in this document

## Setup
### Requirements
* Python 2.7+ (Python 3 is not supported)
* numpy (Preferably >=1.13.3)

After installing the requirements, please copy the `blockstates` directory from a Minecraft .jar into the base repository
directory. This cannot be included in the repository due to legal concerns.

## Usage
#### Note: Only Blockstate format worlds are currently supported, and are in a read-only state

```python
import os
import format_loader

world = format_loader.load_world(os.path.join('tests', '1.13 World'))
chunk = world.getChunk(0,0)
print chunk.Blocks[0,0,0] # minecraft:bedrock
```

The library is missing many functions that are present in the original pymclevel, but feature parity is slowly improving

## TODO
- [x] Common access point to load a world
- [x] Have the world loader find world formats at runtime
- [x] Load a Blockstate chunk from a .mca region file and decode the Blockstate array
- [ ] Re-encode the Blockstate array so Minecraft can load the modified world
- [ ] Add missing functions to the api and the Blockstate format module
- [ ] Add anvil support
- [ ] Complete the `tests` module so we have comprehensive testing of the library
- [ ] Add in missing functionality of the original pymclevel

#### Feel free to submit a Pull Request to help complete any of these items