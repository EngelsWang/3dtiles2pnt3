import struct
import json
import math
import cv2 as cv
import queue
import numpy as np
import sympy

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

def getPosTranMat(node):
    if "matrix" in node:
        return np.mat(node["matrix"]).T
    else:
        #单位矩阵
        matrix = np.matlib.eye(4)
        if "translation" in node:
            matrix[0, 3] =  node["translation"][0]
            matrix[1, 3] =  node["translation"][1]
            matrix[2, 3] =  node["translation"][2]
        if "rotation" in node:
            w = node["rotation"][0]
            x = node["rotation"][1]
            y = node["rotation"][2]
            z = node["rotation"][3]
            #四元数转矩阵
            rotationMat = np.mat([
                [1 - 2*(y**2) - 2*(z**2), 2*x*y - 2*z*w, 2*x*z + 2*y*w, 0],
                [2*x*y + 2*z*w, 1 - 2*(x**2) - 2*(z**2), 2*x*z + 2*y*w, 0],
                [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*(x**2) - 2*(y**2), 0],
                [0, 0, 0, 1]
            ])
            matrix = matrix*rotationMat
        if "scale" in node:
            scaleMat = np.mat([
                [node["scale"][0],0,0,0],
                [0,node["scale"][1],0,0],
                [0,0,node["scale"][2],0],
                [0,0,0,1]
            ])
            matrix = matrix*scaleMat
        return matrix

def posTran(x,y,z,matrix):
    vec4 = np.mat([[x],[y],[z],[1]])
    vec4 = matrix*vec4
    return vec4[0,0],vec4[1,0],vec4[2,0]

def normalizeUV(u):
    '''
     解析点uv坐标，假设平铺，uv坐标转换
    '''
    u = math.modf(u)[0]
    if u < 0:
        u = u + 1
    return u

def _getBufferOffset(param):
    '''
    Get buffer byteOffset
    # param: indice of assossor
    '''
    accessor = gltf["accessors"][param]
    bufferView = accessor["bufferView"]
    if "byteOffset" in gltf["bufferViews"][bufferView]:
        bufferViewOffset = gltf["bufferViews"][bufferView]["byteOffset"]
    else:
        bufferViewOffset = 0
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
    #print(nodes)
    allPnt = []
    
    for node in nodes:
        #print(node)
        tranMat = getPosTranMat(gltf["nodes"][node])
        if 'mesh' in gltf["nodes"][node]:
            mesh = gltf["nodes"][node]["mesh"]
            for primitive in gltf["meshes"][mesh]["primitives"]:
                #get indices
                position = primitive["attributes"]["POSITION"]
                [byteOffset,count] =_getBufferOffset(position)
                fBin.seek(byteOffset + fseek)
                #print(count)
                for i in range(0, count):
                    #解析点位置
                    _newPnt = _Pnt()
                    [_newPnt.x, _newPnt.y ,_newPnt.z ] = posTran(
                        struct.unpack('f', fBin.read(4))[0],
                        struct.unpack('f', fBin.read(4))[0],
                        struct.unpack('f', fBin.read(4))[0],
                        tranMat)
                    allPnt.append(_newPnt)

                #if TEXCOORD_1 exist，纹理坐标
                if "TEXCOORD_0" in primitive["attributes"]:
                    texcoord_0 = primitive["attributes"]["TEXCOORD_0"]
                    [byteOffset,count] =_getBufferOffset(texcoord_0)
                    fBin.seek(byteOffset + fseek)
                    for i in range(0, count):
                        #解析点uv坐标，假设平铺，uv坐标转换
                        allPnt[i - count].u = normalizeUV(struct.unpack('f', fBin.read(4))[0])
                        allPnt[i - count].v = normalizeUV(struct.unpack('f', fBin.read(4))[0])

                #获取RGBA数据
                if "material" in primitive:
                    material = primitive["material"]
                    pbrMetallicRoughness = gltf["materials"][material]["pbrMetallicRoughness"]
                    if "baseColorTexture" in pbrMetallicRoughness:
                        #print(1)
                        textureIndex = pbrMetallicRoughness["baseColorTexture"]["index"]
                        imgSource = gltf["textures"][textureIndex]["source"]
                        if not imgSource in allImg:
                            if "uri" in gltf["images"][imgSource]:
                                imgURI = gltf["images"][imgSource]["uri"]
                                allImg[imgSource] = cv.imread(imgURI)
                            else:
                                #图片存储在buffer中
                                bufferview = gltf["images"][imgSource]["bufferView"]
                                if gltf["images"][imgSource]["mimeType"] == "image/jpeg":
                                    byteOffset = gltf["bufferViews"][bufferview]["byteOffset"]
                                    byteLength = gltf["bufferViews"][bufferview]["byteLength"]
                                    fBin.seek(byteOffset + fseek)
                                    imgNpArr = np.frombuffer(fBin.read(byteLength), np.uint8)
                                    allImg[imgSource] = cv.imdecode(imgNpArr, cv.IMREAD_COLOR)

                        [u, v, t] = allImg[imgSource].shape
                        u = u-1
                        v = v-1
                        for i in range(0, count):
                            _idx = i - count
                            realu = int(u*allPnt[_idx].u)
                            realv = int(v*allPnt[_idx].v)
                            allPnt[_idx].imgSource = imgSource
                            ## python 图像存储为bgr，此次需修改
                            allPnt[_idx].r = allImg[imgSource][realu][realv][0]
                            allPnt[_idx].g = allImg[imgSource][realu][realv][1]
                            allPnt[_idx].b = allImg[imgSource][realu][realv][2]
                            if t == 4:
                                allPnt[_idx].a = allImg[imgSource][realu][realv][3]



    fBin.seek(fseek)
    return allPnt

def triangleArea(pnt1, pnt2, pnt3):
    '''
    calculate space triangle area
    '''
    # 构建点
    P1 = np.array([pnt1.x, pnt1.y, pnt1.z], np.float)
    P2 = np.array([pnt2.x, pnt2.y, pnt2.z], np.float)
    P3 = np.array([pnt3.x, pnt3.y, pnt3.z], np.float)

    # A和B两个向量尾部相连
    A = P3 - P1
    B = P2 - P1
    # 计算叉乘
    A_B = np.cross(A, B)
    # 计算叉乘的膜
    AB_mo = np.linalg.norm(A_B)
    # 计算面积
    Area = AB_mo / 2
    return Area

def triangleCentroid(pnt1, pnt2, pnt3):
    return (pnt1.x+pnt2.x+pnt3.x)/3, (pnt1.y+pnt2.y+pnt3.y)/3, (pnt1.z+pnt2.z+pnt3.z)/3

def isTriangle(pnt1,pnt2,pnt3):
    if pnt1.x==pnt2.x  and  pnt1.y==pnt2.y  and  pnt1.z==pnt2.z:
        return False
    if pnt2.x==pnt3.x  and  pnt2.y==pnt3.y  and  pnt2.z==pnt3.z:
        return False
    if pnt1.x==pnt3.x  and  pnt1.y==pnt3.y  and  pnt1.z==pnt3.z:
        return False
    return True

def getTransMat(pnt1, pnt2, pnt3):
    '''
    返回 降维变换矩阵 和 二维仿射变换矩阵
    '''
    P = np.mat([[pnt1.x, pnt2.x, pnt3.x],[pnt1.y, pnt2.y, pnt3.y],[pnt1.z, pnt2.z, pnt3.z]])
    #print(P)
    #-------------降维--------------
    #旋转到与XOY平行的平面
    #求法线
    _A1 =  np.array([pnt1.x - pnt2.x, pnt1.y - pnt2.y, pnt1.z - pnt2.z])
    _A2 =  np.array([pnt3.x - pnt2.x, pnt3.y - pnt2.y, pnt3.z - pnt2.z])
    normal = np.cross(_A1, _A2)
    #print(normal)
    #法线旋转
    norm = np.linalg.norm(normal)
    normXY = np.linalg.norm(normal[0:2])

    #绕z轴旋转到ZOY平面
    cosFai = normal[1]/normXY
    sinFai = normal[0]/normXY
    R1 = np.mat([[cosFai, -sinFai, 0],[sinFai, cosFai, 0],[0, 0, 1]])
    #绕x轴到z轴
    cosFai = normal[2]/norm
    sinFai = normXY/norm
    R2 = np.mat([[1,0,0],[0,cosFai,-sinFai],[0,sinFai,cosFai]])
    #降维以及平移预处理
    R3 = np.mat([[1, 0, 0],[0, 1, 0],[0, 0, 0]])
    P = R3*R2*R1*P
    #print(P)
    P[2]=np.mat([1,1,1])
    #print(P)

    #--------------二维---------------
    #二维仿射变换
    m11 = sympy.Symbol('m11')
    m12 = sympy.Symbol('m12')
    m21 = sympy.Symbol('m21')
    m22 = sympy.Symbol('m22')
    xt = sympy.Symbol('xt')
    yt = sympy.Symbol('yt')
    result = sympy.solve([
        m11*P[0,0] + m12*P[1,0] + xt - pnt1.u,
        m21*P[0,0] + m22*P[1,0] + yt - pnt1.v,
        m11*P[0,1] + m12*P[1,1] + xt - pnt2.u,
        m21*P[0,1] + m22*P[1,1] + yt - pnt2.v,
        m11*P[0,2] + m12*P[1,2] + xt - pnt3.u,
        m21*P[0,2] + m22*P[1,2] + yt - pnt3.v
        ],
        [m11,m12,m21,m22,xt,yt]
        )
    #print(result)
    R4 = np.mat([
        [result[m11],result[m12],result[xt]],
        [result[m21],result[m22],result[yt]],
        [0,0,1]
        ],dtype = float)
    return R3*R2*R1,R4

def xyz2uv(dimRedMat, affineMat, pnt):
    P = np.mat([[pnt.x],[pnt.y],[pnt.z]])
    P = dimRedMat * P
    P[2] = 1
    P = affineMat * P
    return P[0,0],P[1,0]

def insertPoint(allPnt, sizeA = 1):
    for i in range(0, len(allPnt), 3):
        #用于表示是否需要采样
        if isTriangle(allPnt[i], allPnt[i + 1], allPnt[i + 2]):
            # 获取uv变换矩阵
            [dimRedMat, affineMat]= getTransMat(allPnt[i], allPnt[i + 1], allPnt[i + 2])
        else:
            continue
        # 获取源图像
        if allPnt[i].imgSource > -1:
            [u, v, t] = allImg[allPnt[i].imgSource].shape

        # 创建三角形队列
        triQ = queue.Queue()
        # 三角形入队
        triQ.put([allPnt[i], allPnt[i + 1], allPnt[i + 2]])
        while not triQ.empty():
            [pnt1, pnt2, pnt3] = triQ.get()
            _area = triangleArea(pnt1, pnt2, pnt3)
            if _area > sizeA:
                _center = triangleCentroid(pnt1, pnt2, pnt3)
                #print(_center)
                #坐标设置
                _pnt = _Pnt(_center[0], _center[1], _center[2])
                #纹理坐标设置
                [_pnt.u , _pnt.v] = xyz2uv(dimRedMat ,affineMat, _pnt)
                #RGBA设置
                
                if allPnt[i].imgSource > -1:
                    realu = int(u*_pnt.u)
                    realv = int(v*_pnt.v)
                    _pnt.imgSource = allPnt[i].imgSource
                    _pnt.r = allImg[_pnt.imgSource][realu][realv][0]
                    _pnt.g = allImg[_pnt.imgSource][realu][realv][1]
                    _pnt.b = allImg[_pnt.imgSource][realu][realv][2]

                #pnt入链表
                allPnt.append(_pnt)
                #print(_pnt.x, _pnt.y, _pnt.z, _pnt.u, _pnt.v,_pnt.r, _pnt.g,_pnt.b)

                #子三角形入队
                triQ.put([pnt1, pnt2, _pnt])
                triQ.put([pnt2, pnt3, _pnt])
                triQ.put([pnt1, pnt3, _pnt])
    

def readGLTFjosn(fname):
    fGltf = open(fname, 'r')
    content = fGltf.read()
    gltf = json.loads(content)
    fGltf.close()
    return gltf



def readBin(fname, gltf):
    fBin = open(fname, "rb")
    allImg = {}
    gltfPnts = getPnt(gltf, fBin)
    insertPoint(gltfPnts, sizeA = 10)
    fBin.close()

