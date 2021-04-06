import struct
import json

class _Pnt:
    '''
        Position: x, y, z
        UV_tran: u, v
        Color: r, g, b, a
        Material indice: material
    '''
    u = 0.0
    v = 0.0
    r = 0.0
    g = 0.0
    b = 0.0
    a = 0.0
    ImgSource = -1
    def __init__(self, xx = 0.0, yy = 0.0, zz = 0.0):
        self.x = xx
        self.y = yy
        self.z = zz

def getTileset(folderName):
    f = open(folderName+"/tileset.json", 'r')
    content = f.read()
    tiles = json.loads(content)
    f.close()
    del content
    return tiles

def _getBufferOffset(gltf, param):
    '''
    Get buffer byteOffset
    # param: indice of assossor
    '''
    accessor = gltf["accessors"][param]
    bufferView = accessor["bufferView"]
    bufferViewOffset = gltf["bufferViews"][bufferView]["byteOffset"]
    if "byteOffset" in accessor:
        byteOffset = accessor["byteOffset"]
    else:
        byteOffset = 0
    count = accessor["count"]
    # 避免两次重定位
    return bufferViewOffset + byteOffset, count

def getPnt(gltf, fBin):
    '''
    get gltf's all positions
    gltf: json object, 
    fbin: bin file point
    '''
    fseek = fBin.seek(0,1)
    nodes = gltf["scenes"][0]["nodes"]
    print(nodes)
    allPnt = []
    for node in nodes:
        print(node)
        if 'mesh' in gltf["nodes"][node]:
            mesh = gltf["nodes"][node]["mesh"]
            for primitive in gltf["meshes"][mesh]["primitives"]:
                #get indices
                position = primitive["attributes"]["POSITION"]
                [byteOffset,count] =_getBufferOffset(gltf, position)
                fBin.seek(byteOffset + fseek,0)
                for i in range(0, count):
                    _newPnt = _Pnt()
                    _newPnt.x = struct.unpack('f', fBin.read(4))[0]
                    _newPnt.y = struct.unpack('f', fBin.read(4))[0]
                    _newPnt.z = struct.unpack('f', fBin.read(4))[0]
                    allPnt.append(_newPnt)
                #if TEXCOORD_1 exist
                ##print(position)
                if "TEXCOORD_0" in primitive["attributes"]:
                    ##print("Texture1")
                    texcoord_0 = primitive["attributes"]["TEXCOORD_0"]
                    [byteOffset,count] =_getBufferOffset(gltf, texcoord_0)
                    fBin.seek(byteOffset + fseek,0)
                    for i in range(0, count):
                        allPnt[i - count] = struct.unpack('f', fBin.read(4))[0]
                        allPnt[i - count] = struct.unpack('f', fBin.read(4))[0]
    fBin.seek(fseek)
    return allPnt

def readB3dm(fname):
    fb3dm = open(fname,"rb")

    # b3dmHeader
    b3dmHeader = fb3dm.read(28)
    gltfOffset = 0
    for i in range(12,28,4):
        gltfOffset += struct.unpack('i',b3dmHeader[i:i+4])[0]
    fb3dm.seek(gltfOffset,1)

    magic = struct.unpack('I',fb3dm.read(4))[0]
    version = struct.unpack('I',fb3dm.read(4))[0]
    length = struct.unpack('I',fb3dm.read(4))[0]

    chunkLength = struct.unpack('I',fb3dm.read(4))[0]
    chunkType = struct.unpack('I',fb3dm.read(4))[0]

    #gltf decoder
    gltf = json.loads(fb3dm.read(chunkLength).decode("utf-8"))

    chunkLength = struct.unpack('I',fb3dm.read(4))[0]
    chunkType = struct.unpack('I',fb3dm.read(4))[0]

    d = getPnt(gltf, fb3dm)
    fb3dm.close()
    return d

def read3dTilesB3dm(fname):
    # get tileset.json
    tiles = getTileset(fname)
    root = tiles['root']
    uri = root['content']['uri'] 
    d = readB3dm(uri)
    print(d)
    
