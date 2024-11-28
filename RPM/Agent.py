import cv2
import numpy as np
import os
import math
from itertools import product
from idlelib.config import _warn

#from CV2Utils import CV2Utils 

APPPATH = os.path.dirname(__file__)

THRESHOLD = 1
# threshod 
#SAMEIMG_THRESHOLD = 0.03 
SAMEIMG_THRESHOLD = 0.015 

#PIXCELS_THRESHOLD_FOR_NOEMPLY = 10  # 用于判断 是否为 空白图片
THRESHOLD_FOR_NOEMPLY = 0.015  # 用于判断 是否为 空白图片


#
#  线 宽度 一般 四个像素
#
def load_problem_images(problem):
    """
    Converts problem images to grayscale and stores them in dictionaries.

    Args:
        problem: The RavensProblem instance.

    Returns:
        tuple: A dictionary of main images and a dictionary of potential answers.
    """
    images = {}
    #potential_answers = {}
    #imgs = {}
    for key, img in problem.figures.items():
        # Convert the image to grayscale for faster processing
        #
        # In many cases, especially when the images differ mainly in structure, patterns, or shapes, 
        # converting to grayscale can help focus on those key elements 
        # without being distracted by color variations.
        # 
        # If the images have strong edges, shapes, or patterns that are primarily defined by contrast (e.g., black and white images), 
        # grayscale conversion will generally work well.
        img = cv2.cvtColor(cv2.imread(img.visualFilename), cv2.COLOR_BGR2GRAY)
        #
        # 有 !=0 和 255 的  像素, (如  Challenge E-02) 这样会影响到 位运算, cv2.countNonZero 等
        #
        ravens_image = img
        _, ravens_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        #cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        #_, ravens_image = cv2.threshold(img, 64, 255, cv2.THRESH_BINARY)

        # print("ravens_image:", ravens_image)
        #
        # images["A"] = .../A.png
        # potential_answers["1"] = .../1.png 
        #
        #if key.isalpha():
        #    images[key] = ravens_image
        #elif key.isdigit():
        #    potential_answers[key] = ravens_image
        images[key] = Image1(key,ravens_image)
    return images #, potential_answers,imgs
    #return images, potential_answers,imgs

def binarySearch(array:list,key:any,cmp)->int:
    low = 0
    high = len(array)-1
    while low <= high:
        mid = (low + high) >> 1
        c = cmp(array[mid], key)
        if c < 0:
            low = mid + 1;
        elif c > 0:
            high = mid - 1;
        else:
            return mid;
    return -(low + 1)

def indexOf(list,value):
    if list==None:
        return -1
    try:
        return list.index(value)
    except ValueError:    
        return -1
    
    #
    # return 0 : v1==v2==v3
    #        1 : v1+v2==v3
    #        2 : v1-v2==v3 
    #
def _compare3(v1,v2,v3)->bool:
    if v3>0 and abs((v1+v2-v3) / v3 )<0.01:
        return 1
    if v1>0 and abs((v2+v3-v1) / v1 )<0.01:
        return 2
    if v3>0 and abs((v1-v3) / v3 )<0.01 and abs((v2-v3) / v3 )<0.01:
        return 0
    return -1

def countImageDiff(image1,image2):
    return cv2.countNonZero(cv2.absdiff(image1,image2))
#
# 不同相似的 比例
#
def countImageDiffRatio(image1,image2):
    height, width = image1.shape
    diffCount = countImageDiff(image1,image2)
    return diffCount / (height*width), diffCount,height*width

NeighborPixelsYX = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
#
# 
#
def countImageDiffCaseNeighbor(image1,image2):
    height,width = image1.shape
    count = 0
    for y,x in product(range(height), range(width)):
        v1 = image1[y][x]
        v2 = image2[y][x]
        if  v1==v2: continue
        if y>0 and x>0 and x<width-1 and y<height-1:
            neighborCount = 0
            #neighborCount = 0
            if v1==0:
                for yi,xi in NeighborPixelsYX:
                    if image1[y+yi][x+xi]==0:# and image2[y+yi][x+xi]!=0: # 周围 不同的点
                        neighborCount += 1
                        if neighborCount>2:break        
            else:
                for yi,xi in NeighborPixelsYX:
                    if image2[y+yi][x+xi]==0:# and image1[y+yi][x+xi]!=0:
                        neighborCount += 1
                        if neighborCount>3: break     
            #print("y=%d x=%d neighborCount=%d, v1=%d,v2=%d" %(y,x,neighborCount,v1,v2))   
            if neighborCount<=3:
                continue
        #if (v1!=0 and v1!=255) or (v2!=0 and v2!=255):
        #    raise BaseException("??? v1=%d v2=%d" %(v1,v2))
        count += 1
    return count        
#
#  y 行 最后一个 黑点
#
def endBlackPointX(image,y,fromX=0,endX=0):
    if endX<=0:
        _, endX = image.shape
    for x in range(endX-1,fromX-1,-1):
        if image[y,x]==0:
             return x
    return -1     

#
#  y 行 第一个 黑点
#
def startBlackPointX(image,y,fromX=0,endX=0):
    if endX<=0:
        _, endX = image.shape
    for x in range(fromX,endX):
        if image[y,x]==0:
             return x
    return -1     

#
#  x 列 第一个 黑点
#
def startBlackPointY(image,x,fromY=0,endY=0):
    if endY<=0:
        endY,_ = image.shape
    for y in range(fromY,endY):
        if image[y,x]==0:
             return y
    return -1   

#
# x线上 最后一个黑点
#
def endBlackPointY(image,x,fromY=0,endY=0):
    if endY<=0:
        endY,_ = image.shape
    for y in range(endY-1,fromY-1,-1):
        if image[y,x]==0:
             return y
    return -1     
  
#
#  获取  水平线(horizontals)(坐标==y) 上的线段 起始 结束 点
#   返回 起始/结束 点的 
#   
def getHLineSegments(image,y,ignoreSegSize, fromX=0,endX=0):
    if endX<=0:
        _, endX = image.shape
    setments = []
    x1 = -1
    for x in range(fromX,endX):
        if image[y,x]==0:
            if x1<0:
                x1 = x
        else:
            if x1>=0:
               if x-x1>ignoreSegSize : setments.append((x1,x))
               x1 = -1   
    if x1>=0 and x-x1>ignoreSegSize :
        setments.append((x1,x))
    return setments

#
# 获取 垂直线(坐标==x) 方向 上的线段  
#
def getVLineSegments(image,x,ignoreSegSize,fromY=0,endY=0):
    if endY<=0:
        endY,_ = image.shape
    setments = []
    y1 = -1
    for y in range(fromY,endY):
        if image[y,x]==0:
            if y1<0:
                y1 = y
        else:
            if y1>=0:
               if y-y1>ignoreSegSize :setments.append((y1,y))
               y1 = -1   
    if y1>=0 and y-y1>ignoreSegSize :
        setments.append((y1,y))
    return setments

def  toRelativeCenterPoint(image,x,y):
    height,width = image.shape
    return x-width/2, y-height/2



#
# 图片中的一个元素 (图片的相连区域)
# 
class ImageElement:
    def __init__(self,shape,name:str=""):
        if shape!=None:
            self.image = np.full(shape, 255, np.uint8)  # pixel
        #self.imgFilledRatio = None
        self.blackPixelCount = 0
        self.transformImgs = {} # 缓存 翻转, 旋转, 填充 等 变换
        self.hLineSegs = {}  # 缓存 水平 线段
        self.vLineSegs = {} # 缓存 垂直 线段
        self.name = name
        self.IgnoreSegSize = 0  # 取 
        self.x0 = 0
        self.y0 = 0
        self.ex = 0
        self.ey = 0
        #self.match = "none"
        #  "DELETED",  
        #self.transform = "not matched"
        #self.match_weight = 0
    def update(self):
        self.blackPixelCount= 0
        self.x0 = 0
        self.y0 = 0
        self.ex = 0
        self.ey = 0
        self.transformImgs = {} # 缓存 翻转, 旋转, 填充 等 变换
        self.hLineSegs = {}  # 缓存 水平 线段
        self.vLineSegs = {} # 缓存 垂直 线段
        height, width = self.image.shape
        for y,x in product(range(height), range(width)):
            if self.image[y,x]==0:
                self._forPixel(x,y)
    #def isBlack(self,x:int,y:int):
    #    return self.image[y,x]==0 if x>=0 and x<self.ex and y>=0 and y<self.ey else False
    def addPixel(self,x:int,y:int):
        if  self.image[y,x]!=0:
            self.image[y,x] = 0
            self._forPixel(x,y)
            
    def _forPixel(self,x:int,y:int):    
        if self.blackPixelCount==0:
            self.x0 = x
            self.y0 = y
            self.ex = x+1
            self.ey = y+1
        else:    
            if self.ex < x+1:
                self.ex = x+1
            if self.ey < y+1:
                self.ey = y+1
            if self.x0 > x:
                self.x0 = x
            if self.y0 > y:
                self.y0 = y
        self.blackPixelCount += 1    
    # 返回 图片元素的 高, 宽
    def  getSize(self):
        return (self.ey-self.y0,self.ex-self.x0)
    def  getCenter(self):
        return (self.ex+self.x0-1)/2,(self.ey+self.y0-1)/2
    def getWidth(self):
        return self.ex-self.x0
    def getHeight(self):
        return self.ey-self.y0
    # @return 图形面积
    def getTotalPixel(self):
        return (self.ex-self.x0) * (self.ey-self.y0)
    def getBlackPixelRatio(self):
        return float(self.blackPixelCount) / float((self.ex-self.x0) * (self.ey-self.y0))
    #
    # 取 第 y 行 第一个像素点
    #
    def getStartPointX(self,y:int):
        return  startBlackPointX(self.image,y,self.x0,self.ex)
    
    #
    # 取 第 y 行 最后一个像素点
    #
    def getEndPointX(self,y:int):
        return  endBlackPointX(self.image,y,self.x0,self.ex)
 
   #
    # 取 第 x 列 第一个像素点
    #
    def getStartPointY(self,x:int):
        return  startBlackPointY(self.image,x,self.y0,self.ey)
    
    #
    # 取 第 x 列 最后一个像素点
    #
    def getEndPointY(self,x:int):
        return  endBlackPointY(self.image,x,self.y0,self.ey)
  
    #
    # y 横线上的 线段
    #
    def getHLineSegments(self,y):
        if y in self.hLineSegs:
            #print("使用缓存")
            return self.hLineSegs[y]
        v = getHLineSegments(self.image,y,self.IgnoreSegSize,self.x0,self.ex)
        self.hLineSegs[y] = v
        return v
    
    def getVLineSegments(self,x):
        if x in self.vLineSegs:
            #print("使用缓存")
            return self.vLineSegs[x]
        v = getVLineSegments(self.image,x,self.IgnoreSegSize,self.y0,self.ey)
        self.vLineSegs[x] = v
        return v
    
    def getHLine(self,y):
        pass
    #
    #
    """
    def getImageFilledRatio(self):
        try:
            if self.imgFilledRatio!=None:
                #print("使用缓存...")
                return self.imgFilledRatio  
        except AttributeError as e:
            pass
    #error: has not attribute
        self.imgFilledRatio = getImageFilledRatio(self.image,self.x0,self.y0,self.ex,self.ey)
        return self.imgFilledRatio   
    """

    #
    #  判断 连个 图形 款/高 比例 一致
    #
    def isImageShapeHWSimilar(self,otherImgElement,hwRatioThreadhold=0.03):
        h1,w1 = self.getSize()
        h2,w2 = otherImgElement.getSize()
        #  C-11 的元素较小, (h1=24,w1=23),图2(h2=24,w2=24) : scale 可能超过hwRatioThreadhold, 使用 h1-h2>2 or w1-w2>2  控制
        return abs(h1/h2 - w1/w2)<=hwRatioThreadhold or (abs(h1-h2)<=2 and abs(w1-w2)<=2),(h1*w1) / (h2*w2) 
    
    def isImageShapeMatched(self,otherImgElement,hwThreadhold=2):
        h1,w1 = self.getSize()
        h2,w2 = otherImgElement.getSize()
        #print("isImageShapeMatched : %s.size=(%d,%d),%s.size=(%d,%d)" %(self.name,h1,w1,otherImgElement.name,h2,w2))
        return abs(h1-h2)<=hwThreadhold and abs(w1-w2)<=hwThreadhold
    
    def isBlackPixelRatioEquals(self,otherImgElement,ratioThreadhold=0.05):
        return abs(self.getBlackPixelRatio()-otherImgElement.getBlackPixelRatio()) <= ratioThreadhold;

    def isBlackPixelEquals(self,otherImgElement,threadhold=20):
        n1 = self.blackPixelCount
        n2 = otherImgElement.blackPixelCount
        return abs(n1-n2) <= 20 
        #if n1==0: return n2<10
        #print("isBlackPixelEquals %s.blackPixelCount=%d,%s.blackPixelCount=%d" %(self.name,n1,otherImgElement.name,n2))
        #return abs(n1-n2)*2/(n1+n2) <= ratioThreadhold;

    #
    #  图形 轮廓的 相似度
    #  返回   (图1 与  图2 的 相似度(0-1.0),  图1/图2 图形大小比例)
    #
    def getImageElementSimilarScale(self,otherImgElement):
        #cached
        cacheKey = "SimilarScale["+self.name+"/"+otherImgElement.name+"]"
        if cacheKey in ImageElement.cached:
            #print("[getImageElementSimilarScale]使用缓存 %s" % cacheKey)
            return ImageElement.cached[cacheKey]
        
        #print("isImageShapeHWSimilar %s ..." % cacheKey)
        hwMatched,scale = self.isImageShapeHWSimilar(otherImgElement)
        if not hwMatched: # 两个图的 长 宽 比例 不一致, 认为他们 不 相似
            #if( Agent._DEBUG ):
            #    h1,w1 = self.getSize()
            #    h2,w2 = otherImgElement.getSize()
            #    print("scaleH=%f,scaleW=%f 图1(h=%d,w=%d),图2(h=%d,w=%d)" %(scale,w1 /  w2,h1,w1,h2,w2))
            ImageElement.cached[cacheKey] = (0,0,False,0)
            return ImageElement.cached[cacheKey]
        #h1,w1 = self.getSize()
        #h2,w2 = otherImgElement.getSize()
        #scale = (h1*w1) / (h2*w2)   # 图1 与  图2 的 比率
        #
        # 确保 self>=imageElement2
        #
        if scale>=1: #h1>=h2: # 
            largeElement = self  
            smallElement = otherImgElement
        else:    
            largeElement = otherImgElement
            smallElement = self
        largeH,largeW = largeElement.getSize()
        smallH,smallW = smallElement.getSize()  
        largeScaleH = largeH /  smallH
        largeScaleW = largeW /  smallW

        
        similarH = ImageElement.checkImageElementHWSimilar1(smallElement,largeElement,largeScaleW,largeScaleH,0) # 水平线相似度
        if similarH>=0.7 :
            similarV = ImageElement.checkImageElementHWSimilar1(smallElement,largeElement,largeScaleW,largeScaleH,1) # 垂直线相似度
        else:
            similarV = 0
            #return ImageElement.cached[cacheKey]
        #print("matchedH/totalLinesH=%d/%d,matchedV/totalLinesV=%d/%d" %(matchedH,totalLinesH,matchedV,totalLinesV))
        #print("similarH = %f,similarV=%f" %(similarH, similarV))
        similar =  0 if similarV==0  else  (similarH+similarV)/2
        similar2 = 0
        pixMatched = False
        if similar<0.9 :
            checkPixMatch = abs(scale-1)<0.03 and abs(smallElement.blackPixelCount-largeElement.blackPixelCount)<15
            #print("%s:%s : scale=%f, blackPixelCount=%d,%d" %(self.name,otherImgElement.name,scale,smallElement.blackPixelCount,largeElement.blackPixelCount))
            # 只有  blackPixelCount , scale 相近的图 才 考虑 match2, 否则 Challenge B-03 的 AB 被认为 相似
            # Challenge B-07 : 
            # D-09 : B3 : 外形相似, 内部 不匹配
            similarH2,pixMatchedH2 = ImageElement.checkImageElementHWSimilar2(smallElement,largeElement,largeScaleW,largeScaleH,0,checkPixMatch) # 水平线相似度
            #print("similarH2 =%s pixMatchedH2=%s" % (similarH2,pixMatchedH2))
            if similarH2>=0.7:
                similarV2,pixMatchedV2 = ImageElement.checkImageElementHWSimilar2(smallElement,largeElement,largeScaleW,largeScaleH,1,checkPixMatch) # 垂直线相似度
                similar2 = (similarH2+similarV2)/2
                pixMatched = pixMatchedH2 and pixMatchedV2
        ImageElement.cached[cacheKey] = (similar,similar2,pixMatched,scale)
        return  ImageElement.cached[cacheKey]
        
 
    CheckSampleLineCount = 50    
    
    #
    #  检测 水平 或 垂直方向 的 相似度
    #   scale : hvType==0 时 水平 或 垂直 的 
    #
    def checkImageElementHWSimilar1(smallElement,largeElement,largeScaleW:float,largeScaleH:float,hvType:int):
        testSampleCount = 50    
        if hvType==0: # 水平线
            lines = ImageElement.getSamplePointForImgSimilarDetect(smallElement.y0,smallElement.ey,testSampleCount)
        else: #垂直线
            lines = ImageElement.getSamplePointForImgSimilarDetect(smallElement.x0,smallElement.ex,testSampleCount)
        totalLines = 0     # 总 检测的 线段 数
        matchedLines = 0   # 其中 匹配的 线段数    
        for i in  range(len(lines)):
            if hvType==0: # 水平线
                y1 = lines[i]  #
                y2 = largeElement.y0+int((y1-smallElement.y0)*largeScaleH+0.5)
                matched = ImageElement.isImageElementLineSegmentSimilar(smallElement,largeElement,y1,y2,largeScaleW,0)
            else:
                x1 = lines[i]  #
                x2 = largeElement.x0+int((x1-smallElement.x0)*largeScaleW+0.5)
                matched = ImageElement.isImageElementLineSegmentSimilar(smallElement,largeElement,x1,x2,largeScaleH,1)
            totalLines += 1
            if matched:
                matchedLines += 1
            if (i==10 and matchedLines/totalLines < 0.5) or (i==20 and matchedLines/totalLines < 0.7):
                break
        return matchedLines/totalLines

    #
    # 检测类似 Challenge Problem B-07 : 使用 checkImageElementHWSimilar1 线段发不能 检测的
    # @return (外形匹配度,内部像素匹配) 
    #
    def checkImageElementHWSimilar2(smallElement,largeElement,largeScaleW:float,largeScaleH:float,hvType:int,checkPixMatch:bool):
        testSampleCount = 50    
        if hvType==0: # 水平线
            lines = ImageElement.getSamplePointForImgSimilarDetect(smallElement.y0,smallElement.ey,testSampleCount)
        else: #垂直线
            lines = ImageElement.getSamplePointForImgSimilarDetect(smallElement.x0,smallElement.ex,testSampleCount)
        #
        # 先检测是否 外轮廓 相似:
        #
        totalLines = 0
        matchedLines = 0
        for i in  range(len(lines)):
            if hvType==0: # 水平线
                y1 = lines[i]  #
                y2 = largeElement.y0+int((y1-smallElement.y0)*largeScaleH+0.5)
                matched = ImageElement.isImageElementLineStartEndMatch(smallElement,largeElement,y1,y2,largeScaleW,0)
            else:
                x1 = lines[i]  #
                x2 = largeElement.x0+int((x1-smallElement.x0)*largeScaleW+0.5)
                matched = ImageElement.isImageElementLineStartEndMatch(smallElement,largeElement,x1,x2,largeScaleH,1)
            totalLines += 1
            if matched:
                matchedLines += 1
            if (i==10 and matchedLines/totalLines < 0.5) or (i==20 and matchedLines/totalLines < 0.7):
                break

        #if  matchedLines/totalLines<0.8:
        #    return 0,False
        if not checkPixMatch or matchedLines/totalLines<0.8:
            return matchedLines/totalLines,False

        #print("matchedLines2/totalLines2 = %d/%d = %f" %(matchedLines,totalLines,matchedLines/totalLines))
        # Challenge Problem B-07 : "C-ROTAGE90","6" 比较 无法使用线段比较 , 使用 像素 比较:
        #largeElement.blackPixelCount / smallElement.blackPixelCount
        #
        nParts = 8  # 将图片分为 八端, 比较每段的 像素 
        countBlackPixs1 = [0 for i in range(nParts)]  # 每端 的像素 个数
        countBlackPixs2 = [0 for i in range(nParts)]
        """
        #posLst1 = []  # 分界
        #posLst2 = []
        if hvType==0: # 水平线
            startPos1 = smallElement.y0
            endPos1 = smallElement.ey
            startPos2 = largeElement.y0
            endPos2 = largeElement.ey
        else:
            startPos1 = smallElement.x0
            endPos1 = smallElement.ex
            startPos2 = largeElement.x0
            endPos2 = largeElement.ex

        #for i in range(nParts):
        #    posLst1.append( startPos1+int((endPos1-startPos1)*(i+1)/(nParts+1)) )    
        #    posLst2.append( startPos2+int((endPos2-startPos2)*(i+1)/(nParts+1)) )    
        """

        #print("startPos=%d,endPos=%d,posLst=%s,startPos2=%d,endPos2=%d,posLst2=%s" %(startPos,endPos,posLst,startPos2,endPos2,posLst2))
        for y,x in product(range(smallElement.y0,smallElement.ey),range(smallElement.x0,smallElement.ex)):
            if smallElement.image[y,x]!=0:
                continue
            if hvType==0: # 水平线
                i = int((y-smallElement.y0)*nParts / (smallElement.ey-smallElement.y0))
            else:
                i = int((x-smallElement.x0)*nParts / (smallElement.ex-smallElement.x0))
            countBlackPixs1[i] += 1

        for y,x in product(range(largeElement.y0,largeElement.ey),range(largeElement.x0,largeElement.ex)):    
            if largeElement.image[y,x]!=0:
                continue
            if hvType==0: # 水平线
                i = int((y-largeElement.y0)*nParts / (largeElement.ey-largeElement.y0))
            else:
                i = int((x-largeElement.x0)*nParts / (largeElement.ex-largeElement.x0))
            countBlackPixs2[i] += 1

        totalRatioDiff = 0
        for i in range(nParts):
            ratio1 = countBlackPixs1[i]/smallElement.blackPixelCount
            ratio2 = countBlackPixs2[i]/largeElement.blackPixelCount
            totalRatioDiff += abs(ratio1-ratio2)
            #print( "%d : 图1 %d/%d=%f  图2 %d/%d=%f" %(i,countBlackPixs1[i],smallElement.blackPixelCount,ratio1,countBlackPixs2[i],largeElement.blackPixelCount,ratio2) )
        avgRatioDiff = totalRatioDiff / nParts
        return matchedLines/totalLines , avgRatioDiff<0.03
        #print("avgRatioDiff = ",avgRatioDiff)
        #if avgRatioDiff<0.03:  # 
        #    return matchedLines/totalLines
        #return 0    
        #nParts = len(countBlackPixs) 
        #return matchedLines/totalLines
        
    #
    #  @param hvType :0 水平线, 1:垂直线
    #   scale 是 imageElement2 / imageElement1 的 图形放缩比例
    #  @param imageElement1 小图, imageElement2 : 大图
    #    
    def isImageElementLineSegmentSimilar(imageElement1,imageElement2,linePos1:int,linePos2:int,scale:float,hvType:int):   
        if hvType==0: # 水平线
            lineSegments1 = imageElement1.getHLineSegments(linePos1)
            lineSegments2 = imageElement2.getHLineSegments(linePos2)
            imgElement1Pos = imageElement1.x0
            imgElement1PosEnd = imageElement1.ex
            imgElement2Pos = imageElement2.x0
            imgElement2PosEnd = imageElement2.ex
        else: #垂直线
            lineSegments1 = imageElement1.getVLineSegments(linePos1)  
            lineSegments2 = imageElement2.getVLineSegments(linePos2) 
            imgElement1Pos = imageElement1.y0
            imgElement1PosEnd = imageElement1.ey
            imgElement2Pos = imageElement2.y0
            imgElement2PosEnd = imageElement2.ey
        nSegments = len(lineSegments1)   
        nSegments2 = len(lineSegments2)
        #print("hvType=%d,nSegments = %d, nSegments2 = %d" %(hvType,nSegments,len(lineSegments2))) 
        # Challenge Problem B-06 : B/3 比较, 线比较多的情况下, 误差较大
        CenterPointDelta = 3 if nSegments<5 else 5
        if nSegments!=nSegments2:
            #
            # 处理 特除情况, 在 边框线 , 
            #
            if hvType==0: # 水平线
                minP, maxP = imageElement1.y0,imageElement1.ey
            else:
                minP, maxP = imageElement1.x0,imageElement1.ex     
            if nSegments==1 and nSegments2==2 and scale>0 and (linePos1<minP+5 or linePos1>maxP-5):
                x1_start,x1_end = lineSegments1[0] # 
                x2_start,_ = lineSegments2[0]
                _,x2_end = lineSegments2[1]
                c2 = (x2_start+x2_end-1) / 2
                c1 = (x1_start+x1_end-1) / 2
                c2By1 = imgElement2Pos + (c1-imgElement1Pos)*scale   # c1 映射到 imageElement2 的位置
                w1 = x1_end - x1_start
                w2 = x2_end - x2_start
                dw =   abs(w2-w1*scale)
                if abs(c2-c2By1)<=CenterPointDelta and (dw<=4 or dw/w2<0.05) : # 线段中心点 不在相同位置
                    #print("%s 位置=%d,%d : 中心点不等 %d!=%d "%("水平线" if hvType==0 else "垂直线",linePos1,linePos2,c2By1,c2 ) )
                    return True
                #print("c2=%f,c2By1=%f,w2=%f,w1*scale=%f,%f" %(c2,c2By1,w2,w1*scale,abs(w2-w1*scale)/w2))

            #if( Agent._DEBUG ):
            #    print("%s 位置=%d,%d : 线段数 不等 :%d!=%d 小图区间=(%d %d) 大图区间=(%d %d),minP=%d,maxP=%d " %("水平线" if hvType==0 else "垂直线",linePos1,linePos2,nSegments,nSegments2,imgElement1Pos,imgElement1PosEnd,imgElement2Pos,imgElement2PosEnd,minP, maxP ))
            #
            #  Challenge Problem B-06 : B/3 比较, 差一根线
            #     
            
            if  (nSegments>=4 or nSegments2>=4) :
                #and  abs(nSegments-nSegments2)==1
                if  nSegments==nSegments2-1 and ImageElement.isLineSegmentsContainsIn(lineSegments1,lineSegments2,imgElement1Pos,imgElement2Pos,scale):    
                    #print("isLineSegmentsContainsIn ok")
                    return True
                elif nSegments==nSegments2+1 and ImageElement.isLineSegmentsContainsIn(lineSegments2,lineSegments1,imgElement2Pos,imgElement1Pos,1/scale):        
                    #print("isLineSegmentsContainsIn ok")
                    return True
                #if  nSegments==nSegments2-1:
                #    print("isLineSegmentsContainsIn fail")    
            #print(lineSegments1, lineSegments2)
            return False
        
        for lineSeg1,lineSeg2 in zip(lineSegments1,lineSegments2):
            x1_start,x1_end = lineSeg1
            x2_start,x2_end = lineSeg2
            w1 = x1_end - x1_start
            w2 = x2_end - x2_start
            if w1<=5 and w2<=5:
                # 处理 边框线 特例
                if  (x1_start<=imgElement1Pos+1 and x2_start<=imgElement2Pos+1) or (x1_end>=imgElement1PosEnd-2 and x2_end>=imgElement2PosEnd-2):
                    continue
            c1 = (x1_start+x1_end-1) / 2
            c2 = (x2_start+x2_end-1) / 2
            c2By1 = imgElement2Pos + (c1-imgElement1Pos)*scale   # c1 映射到 imageElement2 的位置
            #print("[] %s 位置=%d,%d : 中心点 %d, %d "%("水平线" if hvType==0 else "垂直线",linePos1,linePos2,c2By1,c2 ) )
            if abs(c2-c2By1)>CenterPointDelta: # 线段中心点 不在相同位置
                #if( Agent._DEBUG ):
                #    print("%s 位置=%d,%d : 中心点不等 %f(by %f)!=%f, 线段=%s:%s 小图区间=(%d %d) 大图区间=(%d %d)  "%("水平线" if hvType==0 else "垂直线",linePos1,linePos2,c2By1,c1,c2,lineSeg1,lineSeg2,imgElement1Pos,imgElement1PosEnd,imgElement2Pos,imgElement2PosEnd ) )
                return False
            if w1<=5 and w2<=5:
                continue
            #if w1<=7 and w2<=7:
            # Challenge Problems B-01 三角形 垂直 方向 可能 为 9
            if w1<=9 and w2<=9 and nSegments==2:
                continue
            # 
            dw = abs(w2-w1*scale)
            if dw>4 and dw/w2>0.05: # 线段宽度
                #if( Agent._DEBUG ):
                #    print("%s 位置=%d,%d : 宽度不等 %f(w1(%f)*%f)!=w2(%f) ,dw=%f "%("水平线" if hvType==0 else "垂直线",linePos1,linePos2,w1*scale,w1,scale,w2,dw ) )
                return False
        return True         
             
    #
    #  从 lineSegments1 中 去掉 lineSegments2 中的 线段, 返回剩余的线段
    #          
    def isLineSegmentsContainsIn(lineSegments:list,inLineSegments:list,seg1StartPos,seg2StartPos,scale:float)->bool:
        # lineSeg1 是 binarySearch 中的 key, 来自于  lineSegments
        CenterPointDelta = 4
        def lineSegmentsCenterPosCmp(lineSeg2:tuple,lineSeg1:tuple)->int:
            x1_start,x1_end = lineSeg1
            x2_start,x2_end = lineSeg2
            c1 = (x1_start+x1_end-1) / 2
            c2 = (x2_start+x2_end-1) / 2
            c2By1 = seg2StartPos + (c1-seg1StartPos)*scale   # c1 映射到 imageElement2 的位置
            return   c2-c2By1
        inLineSegMatchedIdx = []
        i1 = 0  
        for lineSeg1 in lineSegments:
            x1_start,x1_end = lineSeg1
            i = binarySearch(inLineSegments,lineSeg1,lineSegmentsCenterPosCmp)
            if i<0:
                i = -(i+1)
            c1 = (x1_start+x1_end-1) / 2
            #print("i1=%d  i=%d ..."% (i1,i))
          
            # 在 i 附近 找 最近的 线:
            j = -1
            cenerDelta = 100
            if i<len(inLineSegments):
                x2_start,x2_end = inLineSegments[i]
                c2 = (x2_start+x2_end-1) / 2
                c2By1 = seg2StartPos + (c1-seg1StartPos)*scale
                #print("  [%d] %f "% (i,abs(c2-c2By1)))
                cenerDelta = abs(c2-c2By1)
                j = i
            if i<len(inLineSegments)-1:    
                x2_start,x2_end = inLineSegments[i+1]
                c2 = (x2_start+x2_end-1) / 2
                c2By1 = seg2StartPos + (c1-seg1StartPos)*scale
                #print("  [%d] %f j=%d"% (i+1,abs(c2-c2By1),j))
                if j<0 or cenerDelta>abs(c2-c2By1):
                    cenerDelta = abs(c2-c2By1)
                    j = i+1
            if i>0:
                x2_start,x2_end = inLineSegments[i-1]
                c2 = (x2_start+x2_end-1) / 2
                c2By1 = seg2StartPos + (c1-seg1StartPos)*scale
                #print("  [%d] %f j=%d"% (i-1,abs(c2-c2By1),j))
                if j<0 or cenerDelta>abs(c2-c2By1):
                    cenerDelta = abs(c2-c2By1)
                    j = i-1
            #print("i1=%d  i=%d j=%d cenerDelta=%f 被使用=%s"% (i1,i,j,cenerDelta,indexOf(inLineSegMatchedIdx,j)>=0))
            if j<0 or cenerDelta>CenterPointDelta or indexOf(inLineSegMatchedIdx,j)>=0:
                return False
            w1 = x1_end - x1_start
            x2_start,x2_end = inLineSegments[j]
            w2 = x2_end - x2_start
            if w1>5 and w2>5:
                dw = abs(w2-w1*scale)
                if dw>4 and dw/w2>0.05: # 线段宽度
                    return False
            inLineSegMatchedIdx.append(j)
            i1 += 1
        #
        #  剩下的 线 , 必须是 细线:
        #     
        for j in range(len(inLineSegments)):
            if indexOf(inLineSegMatchedIdx,j)<0:
                x2_start,x2_end = inLineSegments[j]
                if x2_end-x2_start>5:
                    return False
        return True
            #if  indexOf(inLineSegMatchedIdx,i)>=0:
            #    return False
            #lineSeg2 = 
    #
    #  判断 水平线 或 垂直线 起始点和终点匹配
    #  @param hvType :0 水平线, 1:垂直线
    #   scale 是 imageElement2 / imageElement1 的 图形放缩比例
    #  @param imageElement1 小图, imageElement2 : 大图
    #         
    def isImageElementLineStartEndMatch(imageElement1,imageElement2,linePos1:int,linePos2:int,scale:float,hvType:int):
        if hvType==0: # 水平线
            lineSegments1 = imageElement1.getHLineSegments(linePos1)
            lineSegments2 = imageElement2.getHLineSegments(linePos2)
            imgElement1Pos = imageElement1.x0
            #imgElement1PosEnd = imageElement1.ex
            imgElement2Pos = imageElement2.x0
            #imgElement2PosEnd = imageElement2.ex
        else: #垂直线
            lineSegments1 = imageElement1.getVLineSegments(linePos1)  
            lineSegments2 = imageElement2.getVLineSegments(linePos2) 
            imgElement1Pos = imageElement1.y0
            #imgElement1PosEnd = imageElement1.ey
            imgElement2Pos = imageElement2.y0
            #imgElement2PosEnd = imageElement2.ey
        nSegments = len(lineSegments1)   
        nSegments2 = len(lineSegments2)
        #print("%s 位置=%d,%d : nSegments=%d,nSegments2=%d "%("水平线" if hvType==0 else "垂直线",linePos1,linePos2,nSegments,nSegments2) )
        if nSegments==0 and nSegments2==0:
            return True
        if nSegments==0:
            return lineSegments2[nSegments-1][1]  - lineSegments2[0][0] <=5
        if nSegments2==0:
            return lineSegments1[nSegments-1][1]  - lineSegments1[0][0] <=5
        PosDelta = 3           
        startPos1 = lineSegments1[0][0]
        startPos2 = lineSegments2[0][0]
        startPos2By1 = imgElement2Pos + (startPos1-imgElement1Pos)*scale   # startPos1 映射到 imageElement2 的位置
        #print("%s 位置=%d,%d : startPos1=%d,startPos2=%d,startPos2By1=%d "%("水平线" if hvType==0 else "垂直线",linePos1,linePos2,startPos1,startPos2,startPos2By1) )
        if abs(startPos2-startPos2By1)>PosDelta:
            return False
        endPos1 = lineSegments1[nSegments-1][1] 
        endPos2 = lineSegments2[nSegments2-1][1] 
        endPos2By1 = imgElement2Pos + (endPos1-imgElement1Pos)*scale   # startPos1 映射到 imageElement2 的位置
        #print("%s 位置=%d,%d : endPos1=%d,endPos2=%d,endPos2By1=%d "%("水平线" if hvType==0 else "垂直线",linePos1,linePos2,endPos1,endPos2,endPos2By1) )
        return abs(endPos2-endPos2By1)<=PosDelta
    #
    # 从 start 到 end 之间(平均)抽取 maxCount 个 检测点 
    #  其中 前  10 个 按 (end-start)/10 的 间隔 提取
    #      接下 10 个 按 (end-start)/20 的 间隔 提取
    # 
    def getSamplePointForImgSimilarDetect(start,end,maxCount):
        if maxCount>end-start:
            maxCount = end-start
        a1 = [] # (end-start)/10 的 间隔
        a2 = [] # (end-start)/20 的 间隔   
        a3 = [] # 
        #i1 =  0 
        #i1Pos = int((0+0.5)*(end-start)/10)
        i2 =  0 #
        i2Pos = start+int((0+0.5)*(end-start)/20)
        for i in range(maxCount):
            v = start+int((i+0.5)*(end-start)/maxCount)
            if v>=i2Pos:
                if len(a1)<=len(a2): 
                   a1.append(v)
                else:
                   a2.append(v)    
                i2 += 1
                i2Pos = start+int((i2+0.5)*(end-start)/20) 
            else:   
                a3.append(v)
        return a1+a2+a3                 
    
    def isEquals(self,otherImgElement,similarTh=0.90,scaleTh=0.05)->bool:
        similar,_,_,scale = self.getImageElementSimilarScale(otherImgElement)
        return similar>=similarTh and abs(scale-1)<scaleTh

    def isSimilar(self,otherImgElement,similarTh=0.90)->bool:
        similar,_,_,_ = self.getImageElementSimilarScale(otherImgElement)
        #print("%s-%s : similar=%f" %(self.name,otherImgElement.name,similar))
        return similar>=similarTh
    
    def getIndexOfEqElements(self,inElements,excludeIdxs:list=None,similarTh=0.90,hwThreadhold=2,pixelThreadhold=20) ->int: 
        for i in range(len(inElements)):
            if indexOf(excludeIdxs,i)>=0: continue
            e = inElements[i]
            if hwThreadhold>0 and not self.isImageShapeMatched(e,hwThreadhold) : continue
            if pixelThreadhold>0 and not self.isBlackPixelEquals(e,pixelThreadhold) : continue
            similar,_,_,_ = self.getImageElementSimilarScale(e)
            #print("%s - %s : similar=%f" %(self.name,e.name,similar))
            if similar>=similarTh: return  i
        return -1        
    
    def isEqualsAllElements(self,elements,similarTh=0.90,hwThreadhold=2,pixelThreadhold=20)->bool:
        for e in elements:
            if hwThreadhold>0 and not self.isImageShapeMatched(e,hwThreadhold) : return False
            if pixelThreadhold>0 and not self.isBlackPixelEquals(e,pixelThreadhold) : return False
            similar,_,_,_ = self.getImageElementSimilarScale(e)
            #print("%s-%s  similar=%s" %(self.name,e.name,similar))
            if similar<similarTh : return False
        return True    
    
    def isElementsEquals2(elements1:list,elements2:list,similarTh=0.90,pixelThreadhold=20)->bool:
        for e1,e2 in zip(elements1,elements2):
            if pixelThreadhold>0 and not e1.isBlackPixelEquals(e2,pixelThreadhold) : return False
            if not e1.isEquals(e2,similarTh) : return False
        return True
    
    def isOuterSimilarAllElements(self,elements,similarTh=0.90,hwThreadhold=0)->bool:
        for e in elements:
            if hwThreadhold>0 and not self.isImageShapeMatched(e,hwThreadhold) : return False
            #if pixelThreadhold>0 and not self.isBlackPixelEquals(e,pixelThreadhold) : return False
            #similar,similar2,pixMatched,scale
            similar,similar2,_,_ = self.getImageElementSimilarScale(e)
            #print("%s-%s  similar=%f similar2=%f" %(self.name,e.name,similar,similar2))
            if similar<similarTh and similar2<similarTh : return False
        return True    

    def getTransImage(self,transMode:str):
        if transMode==IMGTRANSMODE_FLIPV or transMode==IMGTRANSMODE_FLIPH or transMode==IMGTRANSMODE_FLIPVH :
            return self.getFlipedImage(transMode)
        if transMode==IMGTRANSMODE_ROTATE090 or transMode==IMGTRANSMODE_ROTATE180 or transMode==IMGTRANSMODE_ROTATE270:    
            return self.getRotateImage(transMode)
        if transMode==IMGTRANSMODE_FILLED:
            return self.getFilledImage()
        raise BaseException("getTransImage: invalid transMode = ",transMode)
    #
    #
    #
    def getFlipedImage(self,flipMode:str):
        if flipMode in self.transformImgs:
            #print("使用缓存图片...")
            return self.transformImgs.get(flipMode)
        imgElement = ImageElement(self.image.shape,self.name+"-"+flipMode)
        img = imgElement.image #np.full(self.image.shape, 255, np.uint8)
        imgRegion = self.image[self.y0:self.ey, self.x0:self.ex]
        imgRegion = cv2.flip(imgRegion,getCVFlipMode(flipMode) ) 
        img[self.y0:self.ey, self.x0:self.ex] = imgRegion
        imgElement.update()
        self.transformImgs[flipMode] = imgElement
        return imgElement
    
    #
    # flipMode IMGTRANSMODE_WHOLEFLIPV
    #
    """
    def getWholeFlipedImage(self,flipMode:str):
        #imgKey = IMGTRANSMODE_WHOLEFLIPVH if flipMode==-1 else ( IMGTRANSMODE_WHOLEFLIPV if flipMode==0 else IMGTRANSMODE_WHOLEFLIPH )
        if flipMode in self.transformImgs:
            #print("使用缓存图片...")
            return self.transformImgs.get(flipMode)
        imgElement = ImageElement(None,self.name+"-"+flipMode)
        imgElement.image = cv2.flip(self.image,flipMode ) 
        imgElement.update()
        self.transformImgs[flipMode] = imgElement
        return imgElement
    """

    #
    # get
    #
    def  getWholeFlipedCenterPoint(self,flipMode):
        height,width = self.image.shape
        wholeX0,wholeY0  = width/2,height/2
        x0,y0 = (self.x0+self.ex)/2,(self.y0+self.ey)/2
        x1,y1 = x0-wholeX0, y0-wholeY0
        if flipMode==IMGTRANSMODE_FLIPVH or flipMode==-1 : # -1 : 
            x1 = -x1
            y1 = -y1
        elif flipMode==IMGTRANSMODE_FLIPV or flipMode==0 : # 垂直翻转(上下)
            y1 = -y1
        elif flipMode==IMGTRANSMODE_FLIPH or flipMode==1 : # 水平翻转 ( 左右 )
            x1 = -x1
        else:
            raise("flipMode=",flipMode)    
        return  int(x1+wholeX0+0.5),int(y1+wholeY0+0.5)

    #def 
    
    def getRotateImage(self,rotaMode:str):
        if rotaMode in self.transformImgs:
            #print("使用缓存图片...")
            return self.transformImgs.get(rotaMode)
        #print("rotaMode=",rotaMode )
        if rotaMode==IMGTRANSMODE_ROTATE090: #ROTAGE90
            rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE # ROTATE_90_CLOCKWISE==2 #rotaMode*90
        elif rotaMode==IMGTRANSMODE_ROTATE180:
            rotateCode = cv2.ROTATE_180 #==1
        elif rotaMode==IMGTRANSMODE_ROTATE270:
            rotateCode = cv2.ROTATE_90_CLOCKWISE # ==0
        else:
            rotateCode = None
        
        #print("rotateCode = %d"% rotateCode)
        imgElement = ImageElement(None,self.name+"-"+rotaMode)
        if rotateCode==None:
            if not rotaMode.startswith("ROTATE") : raise BaseException("Invalid rotaMode=%s" % rotaMode)
            angle = int(rotaMode[6:])
            #print("angle = ",angle)
            h, w = self.image.shape
            # 2x3 的仿射变换矩阵
            m = cv2.getRotationMatrix2D(center=(w // 2, h // 2), angle=angle, scale=1)
            imgElement.image = cv2.warpAffine(src=self.image, M=m, dsize=(h, w),borderValue=255) #img.shape[1::-1])
        else:    
            imgElement.image = cv2.rotate(self.image ,rotateCode ) 
        imgElement.update()    
        self.transformImgs[rotaMode] = imgElement
        return imgElement
        """
        imgElement = ImageElement(self.image.shape,self.name+"-"+rotaMode)
        img = imgElement.image # np.full(self.image.shape, 255, np.uint8)
        if rotateCode==cv2.ROTATE_180 or (self.ex-self.x0)==(self.ey-self.y0) :
            imgRegion = self.image[self.y0:self.ey, self.x0:self.ex]
            #print("1--大小 = %d,%d" % imgRegion.shape)
            #CV2Utils.printImage2(imgRegion)
            imgRegion = cv2.rotate(imgRegion,rotateCode ) 
            #print("1--大小 = %d,%d" % imgRegion.shape)
            #CV2Utils.printImage2(imgRegion)
            img[self.y0:self.ey, self.x0:self.ex] = imgRegion
        else:  #   
            cx = int((self.x0+self.ex)/2)
            cy = int((self.y0+self.ey)/2)
            r2 = self.ex-self.x0
            if r2<self.ey-self.y0:
                r2 = self.ey-self.y0
            r = int((r2+0.5)/2)
            imgHeight,imgWidth = self.image.shape
            #print("大小: width = %d,height=%d, 整图大小(%dx%d): x0=%d,y0=%d, 中心点:cx=%d,cy=%d,r=%d" % (self.getWidth(),self.getHeight(),imgWidth,imgHeight,self.x0,self.y0,cx,cy,r))
            if cx-r<0 or cx+r>=imgWidth or cy-r<0 or cy+r>=imgHeight:
                # Challenge D-04 : 旋转 后 可能 溢出
                imgRegion = np.full((r2,r2), 255, np.uint8)
                imgRegion1 = self.image[self.y0:self.ey, self.x0:self.ex]  # imgRegion1=>imgRegion 
                # x0,y0 :  imgRegion1 在 imgRegion 中的位置 
                x0 = max(0,r - int((self.ex-self.x0+0.5)/2))
                y0 = max(0,r - int((self.ey-self.y0+0.5)/2))
                #print("r2=%d,imgRegion1.height=%d,imgRegion1.width=%d ; x0=%d,y0=%d" %(r2,imgRegion1.shape[0],imgRegion1.shape[1],x0,y0)) 
                # r2=147,imgRegion1.height=76,imgRegion1.width=147  , x0=0,y0=147
                imgRegion[y0:y0+self.ey-self.y0,x0:x0+self.ex-self.x0] = imgRegion1
                imgRegion = cv2.rotate(imgRegion,rotateCode) 
                # 再 将 旋转后的图 imgRegion => img
                x0, y0 = min(max(0,cx-r),imgWidth-r2) , min(max(0,cy-r),imgHeight-r2)
                img[x0:x0+r2,y0:y0+r2] = imgRegion
            else:
                imgRegion = self.image[cy-r:cy+r,cx-r:cx+r]
                #print("1--大小 = %d,%d" % imgRegion.shape)
                #CV2Utils.printImage2(imgRegion)
                imgRegion = cv2.rotate(imgRegion,rotateCode ) 
                #print("2--大小 = %d,%d" % imgRegion.shape)
                #CV2Utils.printImage2(imgRegion)
                img[cy-r:cy+r,cx-r:cx+r] = imgRegion
        imgElement.update()    
        self.transformImgs[rotaMode] = imgElement
        return imgElement
        """
    
    def getFilledImage(self):
        imgKey = IMGTRANSMODE_FILLED
        if imgKey in self.transformImgs:
            #print("使用缓存图片...")
            return self.transformImgs.get(imgKey)
        self.transformImgs[imgKey] = self._newFlipedImage(self.name+"-"+imgKey)
        return self.transformImgs[imgKey]
    
    def _newFlipedImage(self,imgName):
        image = self.image
        height ,width = image.shape
        uf = UFarray() # 
        uf.makeLabel() # 忽略 0
        labels = np.full((height, width), 0, np.int32)
        for y, x in product(range(height), range(width)):
            if image[y,x] == 0:
                continue
            if y > 0 and image[y-1,x] != 0:
                # 上行(b)) 有,  
                labels[y,x] = labels[y-1,x]
                if x > 0 and image[y,x-1] != 0:
                    uf.union(labels[y,x], labels[y,x-1])  #  合并到  上面,    
            elif x > 0 and image[y,x-1] != 0:
                # 左 (d)
                labels[y,x] = labels[y,x-1]
            else: 
                labels[y,x] = uf.makeLabel()
 
                #print("(%d,%d) 新生成 Label =%d" %(y,x,labels[y, x]))

        # 精简
        uf.flatten()   
        imgElement = ImageElement(image.shape,imgName)     
        for y, x in product(range(height), range(width)):
            if uf.findRoot(labels[y,x])!=1: # 第一次出现的白色 为 背景
                imgElement.addPixel(x, y)
        return imgElement        
    
    def isFilledImage(self):
        cacheKey = "isFilledImage("+self.name+")"
        if cacheKey in ImageElement.cached:
            #print("[getImageElementSimilarScale]使用缓存 %s" % cacheKey)
            return ImageElement.cached[cacheKey]
        #if IMGTRANSMODE_FILLED in self.transformImgs:
        filledImg = self.getFilledImage() #self.transformImgs.get(imgKey)
        #r,_,_ = countImageDiffRatio(filledImg.image,self.image)
        #print("%s : r=%f" % (self.name, r))
        ImageElement.cached[cacheKey] = countImageDiff(filledImg.image,self.image)==0
        return ImageElement.cached[cacheKey]
        #for y in range(self.y0,self.ey):
        #    for x in range(self.getStartPointX(y)+1,self.getEndPointX(y)):

    #
    # return 2: 水平线填充; 1:垂直线; 3:斜线
    #
    def isLinesFielldImage(self):
        cacheKey = self.name+".isLinesFielldImage()" #+","+str(minLines)+")"
        if cacheKey in ImageElement.cached:
            #print("[getImageElementSimilarScale]使用缓存 %s" % cacheKey)
            return ImageElement.cached[cacheKey]
        def isLineFilledSegment(seg):
            n = len(seg)
            if n==0: return True
            if seg[0][1]-seg[0][0]>10  : return False
            #if len(seg)<=2: return seg[0][1]-seg[0][0]<7 and seg[1][1]-seg[1][0]<7 and seg[1][0]-seg[0][1]<7
            count = 0
            for j in range(1,n):
                s = seg[j]
                if s[1] - s[0]>(10 if j==n-1 else 7) or  s[0] - seg[j-1][1]>7 : return False
            return True
        testSampleCount = 20    
        hLines = ImageElement.getSamplePointForImgSimilarDetect(self.y0,self.ey,testSampleCount)
        vLines = ImageElement.getSamplePointForImgSimilarDetect(self.x0,self.ex,testSampleCount)
        matchedHLines = 0
        matchedVLines = 0
        for x in vLines:
            vlineSegments = self.getVLineSegments(x)  # [(y1,y2),...]
            #print("x=%d:%s, count=%s" %(x,vlineSegments,isLineFilledSegment(vlineSegments)) )
            if( isLineFilledSegment(vlineSegments)): matchedVLines += 1
        for y in hLines:
            hlineSegments = self.getHLineSegments(y)  # [(y1,y2),...]
            #print("y=%d:%s, count=%s" %(y,hlineSegments,isLineFilledSegment(hlineSegments)) )
            if( isLineFilledSegment(hlineSegments)): matchedHLines += 1
        #print("isLinesFielldImage : HLines=%d/%d=%f, VLines=%d/%d=%f" %(matchedHLines,len(hLines),matchedHLines/len(hLines),matchedVLines,len(vLines),matchedVLines/len(vLines)))
        # Challenge D-07 : C ♥ 形, 只有 80%
        # Challenge B-09  :C ♥ 形, 只有 75%
        # Challenge D-07 : 4 五角星 判断 有问题, 
        ImageElement.cached[cacheKey] = (2 if matchedHLines/len(hLines)>=0.75 else 0) | ( 1 if matchedVLines/len(vLines)>=0.75 else 0)
        return ImageElement.cached[cacheKey]



    #
    #  1: 含 全填充 ; 2: 含未全填充
    #     
    def getImagesFilledFlags(imgElements:list)->int:
        flags = 0
        for e in imgElements:
            if e.isFilledImage():
                flags |= 1
            else:
                flags |= 2
            if flags==3: break    
        return flags    
    
    def isElementsEqualsIgnoreOrder(imgElements1:list,imgElements2:list,similarTh=0.90,scaleTh=0.05)->bool:
        n = len(imgElements1)
        if n!=len(imgElements2):
            return False
        cmped = []
        for i2 in range(n):
            e2 = imgElements2[i2]
            matched = -1
            for i1 in  range(n):
                if indexOf(cmped,i1)>=0:
                    continue
                similar,_,_,scale = e2.getImageElementSimilarScale(imgElements1[i1])
                if similar>=similarTh and abs(scale-1)<scaleTh:
                    cmped.append(i1)
                    matched = i1
                    break
            if matched<0:
                return False
        return True          

    def isAllElementsEquals(imgElements:list,similarTh=0.90,scaleTh=0.15)->bool:
        if len(imgElements)==0:
            return True
        e0 = imgElements[0]
        for e in imgElements[1:]:
            similar,_,_,scale = e0.getImageElementSimilarScale(e)
            #print("%s - %s : similar=%f,scale=%f " %(e0.name,e.name,similar,scale))
            if similar<similarTh or abs(scale-1)>scaleTh:
                return False
        return True            
    
    #
    # elements 的所有元素 包含在 inElements 中
    #
    def isElementsContains(elements:list,inElements:list,similarTh=0.90,scaleTh=0.05)->bool:
        n = len(elements)
        n2 = len(inElements)
        if n>n2:
            return False
        cmped = []
        for i in range(n):
            e = elements[i]
            matched = -1
            for i2 in range(n2):
                if indexOf(cmped,i2)>=0:
                    continue
                similar,_,_,scale = e.getImageElementSimilarScale(inElements[i2])
                if similar>=similarTh and abs(scale-1)<scaleTh:
                    cmped.append(i2)
                    matched = i2
                    break
            if matched<0:
                return False    
        return True

    #
    # 如果所有元素 X 族 中心点 相同, 返回该 中心点, 否则 返回 -1
    #
    def allElementsInCenterX(elements:list)->float:
        if len(elements)==0:
            return 0
        if len(elements)==1:
            return (elements[0].ex+elements[0].x0)/2
        x0 = -1
        sum = 0
        for e in elements:
            x = (e.ex+e.x0)/2
            if x0<0:
                x0 = x
            elif abs(x-x0)>2:
                return -1
            sum += x 
        return sum / len(elements)
    
    def allElementsInCenterY(elements:list)->float:
        if len(elements)==0:
            return 0
        if len(elements)==1:
            return (elements[0].ey+elements[0].y0)/2
        y0 = -1
        sum = 0
        for e in elements:
            y = (e.ey+e.y0)/2
            if y0<0:
                y0 = y
            elif abs(y-y0)>2:
                return -1
            sum += y
        return sum / len(elements)    
    
    def allElementsInCenter(elements:list,threshod:int=4)->bool:
        for e in elements:
            height, width = e.image.shape
            x0,y0 = e.getCenter()
            #print( "%s : x=%f %f y=%f %f" %(e.name,x0,width/2,y0,height/2) )
            if abs(x0-width/2)>threshod or abs(y0-height/2)>threshod: return False
        return True
    
    #
    #  
    #  1 : 在同一个 横线上
    #  2 : 在同一个 纵线上
    #  3 : 在同一个 中线
    #  4 : 在同一个 斜线 上  , -45 度
    #  5 : 在同一个 斜线 上  , 45 度
    #
    def getAllElementsInLine(elements)->int:
        if len(elements)==0: return 0
        #height, width = elements[0].image.shape
        x0 = sum(map(lambda e:(e.x0+e.ex-1)/2,elements)) / len(elements)
        y0 = sum(map(lambda e:(e.y0+e.ey-1)/2,elements)) / len(elements)
        #print("x0=%f y0=%f" %(x0,y0))
        #for e in elements:
        #    dx, dy = (e.x0+e.ex-1)/2-x0 , (e.y0+e.ey-1)/2-y0 
            #x0 +=  (e.x0+e.ex-1)/2
            #y0 +=  (e.y0+e.ey-1)/2
            #print(" %s :  中线点=%f %f " %(e.name,(e.x0+e.ex-1)/2,(e.y0+e.ey-1)/2))
            #print(" %s : dx=%f dy=%f, 中线点=%f %f " %(e.name,dx,dy,(e.x0+e.ex-1)/2,(e.x0+e.ex-1)/2))
        v0 = 0
        for e in elements:
            dx, dy = (e.x0+e.ex-1)/2-x0 , (e.y0+e.ey-1)/2-y0 
            #print(" %s : dx=%f dy=%f, 中线点=%f %f " %(e.name,dx,dy,(e.x0+e.ex-1)/2,(e.x0+e.ex-1)/2))
            if abs(dx)<=4 :
                v = 3 if abs(dy)<=4 else 2
            elif abs(dy)<=4 : 
                v = 1
            elif abs(dx-dy)<=4:
                v = 4
            elif abs(dx+dy)<=4:
                v = 5
            else:
                return 0
            #print(" %s : v=%d, v0=%d" %(e.name,v,v0))
            if v0==0: v0 = v
            if v0==v or v==3:
               continue
            if v0==3:
               v0 = v
            else:
                return 0 
        return v0
    
    #
    # 距离 中心点的 位置
    #  [ (dy,dx),(dy,dx),...]
    #
    def getElementsCenterDistanceXY(elements:list)->list:
        distance = []
        for e in elements:
            height, width = e.image.shape
            x0,y0 = e.getCenter()
            distance.append((x0-width/2,y0-height/2))
        return distance    
    
    #def isBlackPixel(self,x:int,y:int,relativeXY0:bool=True):
    #    if relativeXY0:
    #        x += self.x0
    #        y += self.y0
    #    imgHeight,imgWidth = self.image.shape
    #    return 


    def getMaxSize(elements:list):
        height,width = elements[0].getSize()
        for e in elements[1:]:
            w,h = e.getSize()
            if height<h: height = h
            if width<w: width = w 
        return   height,width  
    
    def getMinXY0(elements:list):
        x0,y0 = elements[0].x0,elements[0].y0
        for e in elements[1:]:
            if x0>e.x0 : x0 = e.x0
            if y0>e.y0 : y0 = e.y0
        return x0,y0    
    
    #
    # 只保留 不同的点
    #
    def getXORImageElement(elements:list):
        cacheKey = "XORImage("+(",".join(map(lambda e:e.name,elements)))+")"
        if cacheKey in ImageElement.cached:
            #print("[getImageElementSimilarScale]使用缓存 %s" % cacheKey)
            return ImageElement.cached[cacheKey]
        elements2 =  elements[1:]
        height,width = elements[0].image.shape
        for e in elements2:
            h2,w2 = e.image.shape
            if height>h2 : height = h2
            if width>w2 : width = w2
        imgElement = ImageElement((height,width),cacheKey)    
        imgElement.IgnoreSegSize = 3
        for y,x in product(range(height), range(width)):
            v0 = elements[0].image[y,x]
            for e in elements2:
                if e.image[y,x]!=v0:
                    imgElement.addPixel(x,y) # img[y,x] = 0
                    break
        ImageElement.cached[cacheKey] = imgElement
        return imgElement
    
    def getANDImageElement(elements:list):
        cacheKey = "ANDImage("+(",".join(map(lambda e:e.name,elements)))+")"
        if cacheKey in ImageElement.cached:
            #print("[getImageElementSimilarScale]使用缓存 %s" % cacheKey)
            return ImageElement.cached[cacheKey]
        elements2 =  elements[1:]
        height,width = elements[0].image.shape
        for e in elements2:
            h2,w2 = e.image.shape
            if height>h2 : height = h2
            if width>w2 : width = w2
        imgElement = ImageElement((height,width),cacheKey)    
        imgElement.IgnoreSegSize = 3
        for y,x in product(range(height), range(width)):
            z = True
            for e in elements:
                if e.image[y,x]!=0:
                    z = False
                    break
            if z: imgElement.addPixel(x,y) # img[y,x] = 0
        ImageElement.cached[cacheKey] = imgElement
        return imgElement
    
    #
    # 按 中点 对齐 合并
    #
    def getElementsCenterAlignMerged(elements:list):
        cacheKey = "ElementsCenterAlignMerged("+(",".join(map(lambda e:e.name,elements)))+")"
        if cacheKey in ImageElement.cached:
            #print("[get2ElementsCenterAlignMerged]使用缓存 %s" % cacheKey)
            return ImageElement.cached[cacheKey]
        image = np.full(elements[0].image.shape, 255, np.uint8)
        height,width = image.shape
        x0,y0 = int((width+0.5)/2),int((height+0.5)/2)
        for e in elements:
            x,y = x0-int((e.ex-e.x0+0.5)/2),y0-int((e.ey-e.y0+0.5)/2)
            dst = image[y:y+e.ey-e.y0,x:x+e.ex-e.x0]
            cv2.bitwise_and(e.image[e.y0:e.ey,e.x0:e.ex],dst,dst=dst,mask=None)
        ImageElement.cached[cacheKey] = image
        return image
    
    """
    def getXORImage1(elements:list):
        cacheKey = "XORImage("+(",".join(map(lambda e:e.name,elements)))+")"
        if cacheKey in ImageElement.cached:
            #print("[getXORImageElement]使用缓存 %s" % cacheKey)
            return ImageElement.cached[cacheKey]
        #print("[getXORImageElement] cacheKey = %s" % cacheKey)
        e0 = elements[0]
        elements2 =  elements[1:]
        size = ImageElement.getMaxSize(elements)
        x0,y0 = ImageElement.getMinXY0(elements)
        #height,width = elements[0].image.shape
        #height,width = elements[0].getSize()
        
        imgElement = ImageElement(elements[0].image.shape,cacheKey)
        imgElement.IgnoreSegSize = 3
        #img = imgElement.image #np.full(self.image.shape, 255, np.uint8)
        for y,x in product(range(size[0]), range(size[1])):
            v0 = e0.image[e0.x0+y,e0.y0+x]
            for e in elements2:
                if e.image[y+e.y0,x+e.x0]!=v0:
                    imgElement.addPixel(x0+x,y0+y) # img[y,x] = 0
                    break
        ImageElement.cached[cacheKey] = imgElement
        return imgElement
    """ 
    #
    #  当前元素位置 与 otherElement 的 位置关系
    #     1:  self 包含在 otherElement 中
    #     2 : otherElement  包含在 self 中
    #     3  : 相交
    #     0  :  分离
    #   
    def getElementPosRel(self,otherElement)->bool:
        cacheKey = "ElementPosRel("+self.name+","+otherElement.name+")"
        if cacheKey in ImageElement.cached:
            #print("[getImageElementSimilarScale]使用缓存 %s" % cacheKey)
            return ImageElement.cached[cacheKey]
        rel = 0
        if  self.x0>otherElement.x0 and self.ex<otherElement.ex and self.y0>otherElement.y0 and self.ey<otherElement.ey:
            rel = 1 # self 包含在 otherElement 中
        elif self.x0<otherElement.x0 and self.ex>otherElement.ex and self.y0<otherElement.y0 and self.ey>otherElement.ey:
            rel = 2
        elif    (self.x0>otherElement.x0 and self.x0<otherElement.ex) \
            or  (self.ex>otherElement.x0 and self.ex<otherElement.ex) \
            or  (self.y0>otherElement.y0 and self.x0<otherElement.ey) \
            or  (self.ey>otherElement.y0 and self.ex<otherElement.ey) :
                rel = 3
        ImageElement.cached[cacheKey] = rel
        return  rel
    
    #
    # 检测当前 图形 是否 为 正 多变形
    #
    def getPolygonPoints(self)->list:
        try:
            return self.__nPolygonPoints
        except AttributeError as e:
            pass
        self.__nPolygonPoints = []
        if self.blackPixelCount==0:
            return self.__nPolygonPoints
        img = cv2.bitwise_not(self.image,mask=None)
        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)!=2 or len(contours[0])!=1:
            raise BaseException("???")
        contour = contours[0][0]
        epsilon = 0.01 * cv2.arcLength(contours[0][0], True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        for p in approx:
            self.__nPolygonPoints.append(p[0])
        M = cv2.moments(approx)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            self.__nPolygonCenter = (cX, cY)
        else:
            self.__nPolygonCenter = None    
        return self.__nPolygonPoints
    
    def isRegularPolygon(self)->int:
        polygonPoints = self.getPolygonPoints()
        if  len(polygonPoints)<=2 or self.__nPolygonCenter==None :
            return 0
        cx,cy = self.__nPolygonCenter
        r0 = -1
        for x,y in polygonPoints:
            r = math.sqrt((x-cx)**2+(y-cy)**2)
            #print("r = ",r)
            if r0 <0:
                r0 = r
            elif abs(r-r0)<3:
                continue
            elif abs(r-r0)/r>0.01:
                return False
        return True      
        

# END class ImageElement

#
# copied from https://github.com/spwhitt/cclabel
#
class UFarray:
    def __init__(self):
        # Array which holds label -> set equivalences
        self.P = []
    def makeLabel(self):
        r = len(self.P) #self.label
        self.P.append(r)
        return r
    # Makes all nodes "in the path of node i" point to root
    def setRoot(self, i, root):
        while self.P[i] < i:
            j = self.P[i]
            self.P[i] = root
            i = j
        self.P[i] = root

    # Finds the root node of the tree containing node i
    def findRoot(self, i):
        while self.P[i] < i:
            i = self.P[i]
        return i
    
    # Finds the root of the tree containing node i
    # Simultaneously compresses the tree
    def find(self, i):
        root = self.findRoot(i)
        self.setRoot(i, root)
        return root
    
    # Joins the two trees containing nodes i and j
    # Modified to be less agressive about compressing paths
    # because performance was suffering some from over-compression
    def union(self, i, j):
        if i != j:
            root = self.findRoot(i)
            rootj = self.findRoot(j)
            if root > rootj: root = rootj
            self.setRoot(j, root)
            self.setRoot(i, root)
    
    def flatten(self):
        for i in range(1, len(self.P)):
            self.P[i] = self.P[self.P[i]]
#End  __init__           

class Image1:
    def __init__(self,name:str,image):
        self.name = name
        self.image = image
        self._blackPixelCount = -1
        self._elementsHeight = -1
        self._elementsWidth = -1
    def getImageElements(self)->list:
        try:
            return self._imgElements  
        except AttributeError as e:
            pass
        self._imgElements = Image1.splitImage(self.image,self.name+"[%d]")
        return self._imgElements  
    
    def getImageElement(self,idx:int)->ImageElement:
        return self.getImageElements()[idx]     
     #
     # 将图片(按像素相连)分隔成多个元素, 相连的像素分在一个元素组中
     #
    def splitImage(image,nameFormat:str)->list:
        height ,width = image.shape
        uf = UFarray() # 
        uf.makeLabel() # 忽略 0
        labels = np.full((height, width), 0, np.int32)
        #   -------------
        #   | a | b | c |
        #   -------------
        #   | d | e |   |
        #   -------------
        #   |   |   |   |
        #   -------------
        for y, x in product(range(height), range(width)):
            # If the current pixel is 0, it's obviously not a component...
            if image[y,x] != 0:
                continue
            if y > 0 and image[y-1,x] == 0:
                # 上行(b)) 有,  
                labels[y,x] = labels[y-1,x]

            elif x+1 < width and y > 0 and image[y-1,x+1] == 0:
                # 右上(c) 有
                c = labels[ y-1,x+1]
                labels[y,x] = c   # = 右上(c)
                if x > 0 and image[y-1,x-1] == 0:
                    # 左上 也有
                    a = labels[y-1,x-1 ]
                    uf.union(c, a)  #  合并到  左上, 
    
                elif x > 0 and image[y,x-1] == 0:
                    # # 左(d)  也有
                    d = labels[y,x-1] #  合并到  左, 
                    uf.union(c, d)

            elif x > 0 and y > 0 and image[y-1,x-1] == 0:
                # 左上(a) 有 , 
                labels[y,x] = labels[y-1,x-1 ]

            elif x > 0 and image[y,x-1] == 0:
                # 左 (d)
                labels[y,x] = labels[y,x-1]
    
            else: 
                labels[y,x] = uf.makeLabel()
                #print("(%d,%d) 新生成 Label =%d" %(y,x,labels[x, y]))

        # 精简
        uf.flatten()        

        imgElementsByLabel = {}
        imgElements = []
        for y, x in product(range(height), range(width)):
            if labels[y,x]==0:
                continue
            label = uf.findRoot(labels[y,x])
            if label not in imgElementsByLabel:
                imgElementsByLabel[label] = ImageElement(image.shape)
                imgElements.append(imgElementsByLabel[label])
            imgElementsByLabel[label].addPixel(x,y)
        # 去掉 <10 的 点:
        for i in range(len(imgElements)-1,-1,-1):
            if imgElements[i].blackPixelCount<10:
                del imgElements[i]

        #imgElements[0].addPixel(0,0)    
        # 按元素 面积(像素个数)  从大到小 排序     184*184         
        imgElements.sort(key=lambda e:(e.getTotalPixel(),e.blackPixelCount),reverse=True)  
        #  按元素 前景像素个数  从大到小 排序   ?? 
        #imageParts.sort(key=lambda e:e.blackPixelCount,reverse=True)   
        if nameFormat!=None:
            for i in range(len(imgElements)):
                imgElements[i].name = nameFormat % i         
                #print("[%d] : getTotalPixel = %d, blackPixelCount=%d" %(i,imageParts[i].getTotalPixel(),imageParts[i].blackPixelCount))
        return imgElements       

    #
    # 整个图片作为一个元素
    #
    def asImgElement(self)->ImageElement:
        try:
            return self._asImgElement
        except AttributeError as e:
            pass
        elements = self.getImageElements()
        if len(elements)==1:
            self._asImgElement = elements[0]
            return elements[0]
        self._asImgElement = ImageElement(None,self.name)
        self._asImgElement.image = self.image
        if len(elements)==0:
            self._asImgElement.x0 = 0
            self._asImgElement.y0 = 0
            self._asImgElement.ex = 0
            self._asImgElement.ey = 0
        else:
            self._asImgElement.blackPixelCount = sum(map(lambda e:e.blackPixelCount,elements))
            self._asImgElement.x0 = min(map(lambda e:e.x0,elements))
            self._asImgElement.y0 = min(map(lambda e:e.y0,elements))
            self._asImgElement.ex = max(map(lambda e:e.ex,elements))
            self._asImgElement.ey = max(map(lambda e:e.ey,elements))
        return self._asImgElement    
    
    def getRotateImage(self,rotaMode:str):
        return self.asImgElement().getRotateImage(rotaMode)
    
    #
    #  images1Lst 中的前景像素 与 images2Lst 中的前景像素 的 个数 差值
    #  [E-02] AB 与 C 前景像素 相差 272/3596/33856
    #    GH 与 7 前景像素 相差 457/6023/33856   
    #  [E-03] AB 与 C 前景像素 相差 142/1848/33856
    #  [E-03] GH 与 2 前景像素 相差 534/2626/33856     534/33856=0.0158  534/2626=0.203
    #

    def countImagesDiff(images1Lst:list,images2Lst:list)->int:
        cacheKey = "ImagesDiff("+("".join(map(lambda v:v.name,images1Lst)))+","+("".join(map(lambda v:v.name,images2Lst)))+")"
        if cacheKey in Image1.cached:
            #print("使用缓存 ",cacheKey)
            return Image1.cached[cacheKey]
        count = 0
        blackCount = 0
        #n1 , n2 = len(images1Lst) ,len(images2Lst)
        height, width = images1Lst[0].image.shape
        for img in images1Lst:
            h, w = img.image.shape
            if height>h:
                height = h
            if width>w:
                width = w
        for img in images2Lst:
            h, w = img.image.shape
            if height>h:
                height = h
            if width>w:
                width = w

        for y,x in product(range(height), range(width)):
            v1 = False
            v2 = False
            for img in images1Lst:
                if img.image[y,x]==0:
                    v1 = True
                    break
            for img in images2Lst:
                if img.image[y,x]==0:
                    v2 = True
                    break
            if v1!=v2:
                count += 1  
                blackCount += 1
            elif v1:
                blackCount += 1
        Image1.cached[cacheKey] = (count ,blackCount, height*width)            
        return Image1.cached[cacheKey] #count ,blackCount, height*width 

    def countImagesXOR(imagesLst:list)->int:
        cacheKey = "ImagesXOR("+("".join(map(lambda v:v.name,imagesLst)))+")"
        if cacheKey in Image1.cached:
            return Image1.cached[cacheKey]
        count = 0
        blackCount = 0
        #n1 , n2 = len(images1Lst) ,len(images2Lst)
        height, width = imagesLst[0].image.shape
        for img in imagesLst:
            h, w = img.image.shape
            if height>h:
                height = h
            if width>w:
                width = w
        image = np.full((height, width), 255, np.uint8)  #        
        for y,x in product(range(height), range(width)):
            v = False
            hasBlack = False
            for img in imagesLst:
                if img.image[y,x]==0 :
                    v = not v
                    hasBlack = True
            if v:
                count += 1
                image[y,x] = 0
            if hasBlack:
                blackCount += 1    
        Image1.cached[cacheKey] = (count ,blackCount, height*width ,image)             
        return Image1.cached[cacheKey] #count ,blackCount, height*width ,image

    def isMatchedLRMerged(img1,img2,img3)->bool:
        cacheKey = "MatchedLRMerged("+img1.name+","+img2.name+","+img3.name+")"
        if not cacheKey in Image1.cached:
            Image1.cached[cacheKey] = Image1.__calc_isMatchedLRMerged(img1,img2,img3)
        return Image1.cached[cacheKey]

    # img1+img2 => img3
    def __calc_isMatchedLRMerged(img1,img2,img3)->bool:
        if len(img1.getImageElements())!=1 or  len(img2.getImageElements())!=1 or len(img3.getImageElements())!=1:
            return False
        e1 = img1.getImageElement(0)
        e2 = img2.getImageElement(0)
        e3 = img3.getImageElement(0)
        height1,width1  =  e1.getSize()
        height2,width2  =  e2.getSize()
        height3,width3  =  e3.getSize()
        #print("%s.width=%d,height=%d,%s.width=%d,height=%d,%s.width=%d,height=%d" % (img1.name,width1,height1,img2.name,width2,height2,img3.name,width3,height3))
        if abs(height3-height1)>1 or abs(height3-height2)>1 or (width1+width2-width3)>2 :
            return False
        n1 = e1.blackPixelCount
        n2 = e2.blackPixelCount
        n3 = e3.blackPixelCount
        #print("%s.blackPixelCount=%d,%s.blackPixelCount=%d,%s.blackPixelCount=%d : (%d+%d-%d)/%d = %f" % (img1.name,n1,img2.name,n2,img3.name,n3,n1,n2,n3,n3,abs (n1+n2-n3) / n3 ))
        if n3==0 or abs (n1+n2-n3) / n3 >0.1:
            return False
        #  Challenge Problem E-04 的 _img2 img3R 
        def cmpLRImage(img1,img2,total)->bool:
            diff = countImageDiff(img1,img2)
            if diff/total < 0.2: return True
            height,width = img1.shape
            for roll in [1,2,3]:
                img1X = img1[0:height,roll:width]  # img1 右移 一位 
                img2X = img2[0:height,0:width-roll]  # img2 
                diff = countImageDiff(img1X,img2X)
                #print("roll = %d : diff=%d/%d = %f  " %(roll,diff,total,diff/total))
                if diff/total < 0.2: return True
                img1X = img1[0:height,0:width-roll]  # img1 右移 一位 
                img2X = img2[0:height,roll:width]  # img2 
                diff = countImageDiff(img1X,img2X)
                #print("roll = %d : diff=%d/%d = %f  " %(roll,diff,total,diff/total))
                if diff/total < 0.2: return True
            return False

        _img1 = e1.image[e1.y0:e1.ey, e1.x0:e1.ex]
        _img2 = e2.image[e2.y0:e2.ey, e2.x0:e2.ex]
        img3L = e3.image[e3.y0:e3.ey, e3.x0:e3.x0+width1]
        img3R = e3.image[e3.y0:e3.ey, e3.ex-width2:e3.ex]
        #print("_img1.size = %d,%d" % _img1.shape )
        #print("_img2.size = %d,%d" % _img2.shape )
        #print("img3L.size = %d,%d" % img3L.shape )
        #print("img3R.size = %d,%d" % img3R.shape )
        return cmpLRImage(_img1,img3L,e1.blackPixelCount) and cmpLRImage(_img2,img3R,e2.blackPixelCount)
        return  False
        #img1R = e1.image[self.y0:self.ey, self.x0:self.ex]
        return n3>0 and abs (n1+n2-n3) / n3 <0.02
        

    def getSumImgElementsBlackPoints(self)->int:
        if self._blackPixelCount>=0:
            #print("使用缓存 ...")
            return self._blackPixelCount
        self._blackPixelCount = 0
        for e in self.getImageElements():
            self._blackPixelCount += e.blackPixelCount
        return self._blackPixelCount
    
    def getImgElementsHeight(self)->int:
        if self._elementsHeight>=0:
            return self._elementsHeight
        y0, ey = 10000, 0
        for e in self.getImageElements():
            y0 = min(y0,e.y0)
            ey = max(ey,e.ey)
        self._elementsHeight = ey-y0    
        return self._elementsHeight 
    def getImgElementsWidth(self)->int:
        if self._elementsWidth>=0:
            return self._elementsWidth
        x0, ex = 10000, 0
        for e in self.getImageElements():
            x0 = min(x0,e.x0)
            ex = max(ex,e.ex)
        self._elementsWidth = ex-x0       
        return self._elementsWidth      

    def allElementsInCenter(self,threshod:int=4)->bool:
        try:
            return self._allElementsInCenter
        except AttributeError as e:
            pass
        self._allElementsInCenter = ImageElement.allElementsInCenter(self.getImageElements())
        return self._allElementsInCenter
    
    def getElementsCenterDistanceXY(self)->list:
        try:
            return self._elementsCenterDistanceXY
        except AttributeError as e:
            pass
        self._elementsCenterDistanceXY = ImageElement.getElementsCenterDistanceXY(self.getImageElements())
        return self._elementsCenterDistanceXY
    
    def getAllElementsInLine(self)->int:
        try:
            return self._allElementsInLine
        except AttributeError as e:
            pass
        self._allElementsInLine = ImageElement.getAllElementsInLine(self.getImageElements())
        return self._allElementsInLine

#END class Image1

#
# 两个图片比较 结果常量
#
IMGTRANSMODE_NONE = None # 
IMGTRANSMODE_EQ = "EQUALS"  # UNCHANGED

IMGTRANSMODE_FLIPV = "FLIPV"  #   (以元素为中心)垂直翻转(上下)  flipMode==0
IMGTRANSMODE_FLIPH = "FLIPH"  #   (以元素为中心)水平翻转 ( 左右 ) flipMode==1
IMGTRANSMODE_FLIPVH = "FLIPVH"  #   (以元素为中心)水平翻转 ( 左右 ) flipMode==-1

#IMGTRANSMODE_WHOLEFLIPV = "WHOLEFLIPV"  #   (以整个图为中心)垂直翻转(上下)  flipMode==0
#IMGTRANSMODE_WHOLEFLIPH = "WHOLEFLIPH"  #   (以整个图为中心)水平翻转 ( 左右 ) flipMode==1
#IMGTRANSMODE_WHOLEFLIPVH = "WHOLEFLIPVH"  #   (以整个图为中心)水平翻转 ( 左右 ) flipMode==-1

IMGTRANSMODE_ROTATE090 = "ROTATE090" #  逆时针 旋转 90度
IMGTRANSMODE_ROTATE180 = "ROTATE180" #   旋转 180度
IMGTRANSMODE_ROTATE270 = "ROTATE270" #  逆时针 旋转 270度(-90度)  
IMGTRANSMODE_FILLED = "FILLED"
IMGTRANSMODE_UNFILLED = "UNFILLED"
IMGTRANSMODE_SIMILAR = "SIMILAR"


def getCVFlipMode(flipTransMode)->int:
    if flipTransMode==IMGTRANSMODE_FLIPV:
        return 0
    if flipTransMode==IMGTRANSMODE_FLIPH:
        return 1
    if flipTransMode==IMGTRANSMODE_FLIPVH:
        return -1
    raise BaseException("Invalid flipTransMode=" , flipTransMode)

#IMGTRANSMODE_ADDED = 11
#IMGTRANSMODE_REMOVED = 12
#IMGTRANSMODE_ZOOM = 15  # 放缩 
#
# 描述 两个图形元素的 变换规则, 例如
#       
class ImageElementTrans:
    def __init__(self,transMode,matched:bool,similar:float,matched2:bool,matched3:bool,similar2:float,scale):
        #self.elementIdx = elementIdx
        self.transMode = transMode  # IMGTRANSMODE_EQ 等
        self.matched = matched # similar > 给定的阈值, 即 表名 两个元素 满足 transMode 的变换规则 ; similar>=0.90
        self.matched2 = matched2 # 轮廓 匹配 , 且 内部像素匹配 ( 只在 !matched 时 有效 )
        self.matched3 = matched3 # 轮廓 匹配  (  内部像素匹配 肯不匹配 )
        self.similar = similar
        self.similar2 = similar2
        self.scale = scale

#END class ImageElementTrans
            
#
#   "两个图形" 构成的对象类,  如 图形A+图形B , 描述 他们直接的 相关
#  Images2("AB")
#    
# 成员 transElements 描述 两个图形 元素的 变换(FLIP,FILL , ROTATE 等 ) 关系 , 
#     例如 A 的 第0个元素,  垂直翻转 后 与图形 C 相等, 则有:
#       transElements[?] == (0,IMGTRANSMODE_FLIPV)
#     如果 A 中一个元素, C 中两个元素, 第一个元素 相等 )
#        transElements[0] ==  ImageElementTrans(IMGTRANSMODE_EQ,1)   描述第一个元素相等
#
class Images2:
    #
    # @param name : 图形Id, 如 "AB","AC","C1",...
    # 
    def __init__(self,agent,name:str):
        self.agent = agent
        self.name = name
        self.img1Id = name[0]   # 如 "A"
        self.img1 = agent.getImage1(self.img1Id)
        self.img2Id = name[1]   # 如 "C"
        self.img2 = agent.getImage1(self.img2Id)
        self.img1Elements = self.img1.getImageElements()  # A  ImageElement[]
        self.img2Elements = self.img2.getImageElements()  # C  ImageElement[]
        self.transElements = []  # 由 class ImageElementTrans 描述
        for _ in range(min(len(self.img1Elements),len(self.img2Elements))):
            self.transElements.append({})
        #    print("elementIdx=%d" % elementIdx)
        #self.transElementsParsed = {}  # key == elementIdx+transMode or elementIdx+transMode*
    #
    # @return 两个 Frame 的 最小 元素 个数
    #
    def getImgElementCount(self):
        return  len(self.transElements)
    
    #
    # img1 -> img2 : 元素 增加 个数
    #
    def getImgElementsCountDiff(self):
        return  len(self.img2Elements) - len(self.img1Elements)
    
    #
    #  img1 / img2 的 相像素 比例
    #
    def getBlackRatio(self):
        n1 = self.img1.getSumImgElementsBlackPoints()
        n2 = self.img2.getSumImgElementsBlackPoints()
        return  n1 if n2==0 else n1/n2
   
    def getAllImgElementTrans(self,elementIdx:int,transModeLst:list)->list:
        trans = []
        cacheTrans = self.transElements[elementIdx]
        for transMode in transModeLst: #Images2.CaseAllTransMode:
            if transMode not in cacheTrans:
                v = Images2.parseImeElementTrans(self.img1Elements[elementIdx],self.img2Elements[elementIdx],transMode)
                cacheTrans[transMode] = v
                if v.transMode!=transMode:
                    raise BaseException("???")
            trans.append(cacheTrans[transMode])
        return trans        
    #
    #  获取 img1Id 与 img2Id 两个图的 第 elementIdx 个元素 的
    #   变换规则
    # @return  返回 {
    #             IMGTRANSMODE_EQ : [similar,scale],  ( similar==0 时, 表示 不满足)
    #             IMGTRANSMODE_FLIPV  : [similar,scale], 
    #             IMGTRANSMODE_FLIPH  : [similar,scale],      
    #              ...
    #         }
    #      如果  IMGTRANSMODE_EQ 满足 的情况下, 就 不再 返回 IMGTRANSMODE_FLIPV,IMGTRANSMODE_FLIPH
    #    

    def getImgElementTrans(self,elementIdx:int,transMode:str) ->ImageElementTrans:
        if elementIdx>=len(self.transElements):
            return None
        cacheTrans = self.transElements[elementIdx]
        #Images2.parseImeElementTrans(self.img1Elements[elementIdx],self.img2Elements[elementIdx],cacheTrans,transModeOnly)    
        #if IMGTRANSMODE_EQ not in cacheTrans:
        #    v = Images2.parseImeElementTrans(self.img1Elements[elementIdx],self.img2Elements[elementIdx],IMGTRANSMODE_EQ)
        #    cacheTrans[IMGTRANSMODE_EQ] = v
        if not transMode in cacheTrans:
            v = Images2.parseImeElementTrans(self.img1Elements[elementIdx],self.img2Elements[elementIdx],transMode)
            cacheTrans[transMode] = v
            if v.transMode!=transMode:
                raise BaseException("???")
        return cacheTrans[transMode]
        #return cacheTrans
       
    #
    #  
    # Challenge D-02","BC","ROTAGE270" : B-ROTAGE270 与 C 相似度:  0.899999
    #
    SimilarMatchedThreshold = 0.85   
    Similar2MatchedThreshold = 0.9
#
#  判断 两个图形元素之间 的 变换规则, 
# 即: 图形元素 dstImgElement 是否 能由 srcImfElement 经过 转换获得
#  例如   srcImfElement 水平翻转后 与 dstImgElement 相同(或相似) 
#   则 返回 (IMGTRANSMODE_FLIPV,similar,scale)  ( 其中 scale 为 相似 比例 )
#      返回 None : 表明 两个图形 无 关系
#  @param srcImgElement , dstImgElement : 待检查的 两个图形元素
#  @param transModeOnly : 只检查 指定的 变换规则
#        
    def parseImeElementTrans(srcImgElement :ImageElement, dstImgElement:ImageElement,transMode:str) ->ImageElementTrans:
        #if( Agent._DEBUG ):
        #    print("parseImeElementTrans %s-%s for %s.... "%(srcImgElement.name, dstImgElement.name,transMode))
        if transMode==IMGTRANSMODE_EQ:
            similar,similar2,pixMatched,scale = srcImgElement.getImageElementSimilarScale(dstImgElement)
            matched3 = similar2>=Images2.Similar2MatchedThreshold
            return ImageElementTrans(transMode,similar>=Images2.SimilarMatchedThreshold,similar,matched3 and pixMatched,matched3,similar2,scale)
        if transMode==IMGTRANSMODE_FLIPV or transMode==IMGTRANSMODE_FLIPH or transMode==IMGTRANSMODE_FLIPVH:
            if srcImgElement.isBlackPixelRatioEquals(dstImgElement):
                #flipMode = 0 if transMode==IMGTRANSMODE_FLIPV else ( 1 if transMode==IMGTRANSMODE_FLIPH else -1)
                similar,similar2,pixMatched,scale = srcImgElement.getFlipedImage(transMode).getImageElementSimilarScale(dstImgElement)
                matched3 = similar2>=Images2.Similar2MatchedThreshold
                #print("flipTransMode=%s : 相似度=%f 比例=%f" %(flipTransMode,similar,scale))
                return  ImageElementTrans(transMode,similar>=Images2.SimilarMatchedThreshold,similar,matched3 and pixMatched,matched3,similar2,scale)    
            return  ImageElementTrans(transMode,False,0,False,False,0,0)
        if transMode==IMGTRANSMODE_FILLED:    
            if srcImgElement.getBlackPixelRatio()<0.3 and dstImgElement.getBlackPixelRatio()>0.4:
                similar,similar2,pixMatched,scale = srcImgElement.getFilledImage().getImageElementSimilarScale(dstImgElement)
                matched3 = similar2>=Images2.Similar2MatchedThreshold
                #print("FILLED: 相似度=%f 比例=%f" %(similar,scale))
                return  ImageElementTrans(transMode,similar>=Images2.SimilarMatchedThreshold,similar,matched3 and pixMatched,matched3,similar2,scale)    
            return  ImageElementTrans(transMode,False,0,False,False,0,0)
        if transMode==IMGTRANSMODE_UNFILLED:    
            if srcImgElement.getBlackPixelRatio()>0.4 and dstImgElement.getBlackPixelRatio()<0.3:
                similar,similar2,pixMatched,scale = dstImgElement.getFilledImage().getImageElementSimilarScale(srcImgElement)
                matched3 = similar2>=Images2.Similar2MatchedThreshold
                return  ImageElementTrans(transMode,similar>=Images2.SimilarMatchedThreshold,similar,matched3 and pixMatched,matched3,similar2,scale)    
            return  ImageElementTrans(transMode,False,0,False,False,0,0)

        if transMode==IMGTRANSMODE_ROTATE090 or transMode==IMGTRANSMODE_ROTATE180 or transMode==IMGTRANSMODE_ROTATE270:
            if srcImgElement.isBlackPixelRatioEquals(dstImgElement):
                rotateImg = srcImgElement.getRotateImage(transMode)
                if rotateImg==None:
                    return ImageElementTrans(transMode,False,0,False,False,0,0)
                similar,similar2,pixMatched,scale = rotateImg.getImageElementSimilarScale(dstImgElement)
                matched3 = similar2>=Images2.Similar2MatchedThreshold
                return  ImageElementTrans(transMode,similar>=Images2.SimilarMatchedThreshold,similar,matched3 and pixMatched,matched3,similar2,scale)  
            return  ImageElementTrans(transMode,False,0,False,False,0,0)


    def isImgElementTransMatched(self,elementIdx:int,transMode:str) ->bool:   
        return self.getImgElementTrans(elementIdx,transMode).matched
    
    def isBlackPixelRatioEquals(self,elementIdx:int,ratioThreadhold=0.05) ->bool:
        return self.img1Elements[elementIdx].isBlackPixelRatioEquals(self.img2Elements[elementIdx],ratioThreadhold)

    def isRoteteMatched(self,elementIdx:int,rotate:int)->bool:
        if( not self.isBlackPixelRatioEquals(elementIdx)) : return False
        if rotate<0 : rotate += 360
        rotateImg = self.img1Elements[elementIdx].getRotateImage("ROTATE%03d" %rotate )
        return  rotateImg.isEquals(self.img2Elements[elementIdx])        

    #
    # 判断 图片 是基于整个图 翻转
    #
    def isWholeImgElementFliped(self,elementIdx:int,flipTransMode:str)->bool:
        if not self.isImgElementTransMatched(elementIdx,flipTransMode): return False
        #flipMode = getFlipModeModeByTransMode(flipTransMode)
        img1 = self.img1Elements[elementIdx]
        img2 = self.img2Elements[elementIdx]
        x1,y1 = img1.getWholeFlipedCenterPoint(flipTransMode) # A 图 (基于整图)翻转后的 中心点 
        x2,y2 = img2.getCenter() # B 图  中心点 
        #print("%s图(基于整图)翻转后的 中心点 = (%f,%f) , %s图中心点 =  (%f,%f) " %(imgs1Name[0:1],xA0,yA0,imgs1Name[1:2],xB0,yB0))
        return abs(x1-x2)<3 and abs(y1-y2)<3

    def isWholeImgElementsFliped(self,flipTransMode:str)->bool:
        n = len(self.img1Elements)
        if n!=len(self.img2Elements) : return False
        for i in range(n):
            if not self.isWholeImgElementFliped(i,flipTransMode): return False
        return True
        
        #if transModeOnly==None or transModeOnly==IMGTRANSMODE_SIMILAR:# = "SIMILAR"   
        #    pass 

    #
    # 判断 从 startElementIdx - endElementIdx 的元素 相同 或 相似
    #
    def isImgElementsEqualsOrSimilar(self,startElementIdx=0,endElementIdx=0)->bool:
        if endElementIdx==0:
            endElementIdx = len(self.transElements)
        cacheKey = self.name+".isImgElementsEqualsOrSimilar("+str(startElementIdx)+","+str(endElementIdx)+")"
        if cacheKey in Images2.cached: return  Images2.cached[cacheKey]
        for elementIdx in range(startElementIdx,endElementIdx):
            transInfo = self.getImgElementTrans(elementIdx,IMGTRANSMODE_EQ)
            if not transInfo.matched:
                Images2.cached[cacheKey] = False
                return False
        Images2.cached[cacheKey] = True
        return True    
    
    def isImgElementsEquals(self,startElementIdx=0,endElementIdx=0)->bool:
        if endElementIdx==0:
            endElementIdx = len(self.transElements)
        cacheKey = self.name+".isImgElementsEquals("+str(startElementIdx)+","+str(endElementIdx)+")"
        if cacheKey in Images2.cached: return  Images2.cached[cacheKey]
        for elementIdx in range(startElementIdx,endElementIdx):
            transInfo = self.getImgElementTrans(elementIdx,IMGTRANSMODE_EQ)
            if not transInfo.matched or abs(transInfo.scale-1)>0.05 :
                Images2.cached[cacheKey] = False
                return False
        Images2.cached[cacheKey] = True    
        return True    
    
    def getImgElementsEqualsIdxMap(self):
        cacheKey = self.name+".imgElementsEqualsIdxMap"
        if cacheKey in Images2.cached:return  Images2.cached[cacheKey]
            #print("使用缓存...",cacheKey)
            #return  Images2.cached[cacheKey]
        #print("getImgElementsEqualsIdxMap...")
        if len(self.img1Elements)!=len(self.img2Elements):
            Images2.cached[cacheKey] = None
            return None
        idxMap = [] # img1Elements[?] 在 img2Elements 对应的位置
        for e1 in self.img1Elements:
            i = e1.getIndexOfEqElements(self.img2Elements,idxMap)
            if i<0: 
                Images2.cached[cacheKey] = None
                return None
            idxMap.append(i)
        Images2.cached[cacheKey] = idxMap
        return idxMap

    #
    # 判断 从 startElementIdx - endElementIdx 的元素 外形匹配 , D-09 的 B3
    #
    def isImgElementsOutterSharpMatched(self,startElementIdx=0,endElementIdx=0)->bool:
        if endElementIdx==0:
            endElementIdx = len(self.transElements)
        cacheKey = self.name+".isImgElementsOutterSharpMatched("+str(startElementIdx)+","+str(endElementIdx)+")"
        if cacheKey in Images2.cached: return  Images2.cached[cacheKey]
        for elementIdx in range(startElementIdx,endElementIdx):
            transInfo = self.getImgElementTrans(elementIdx,IMGTRANSMODE_EQ)
            if not transInfo.matched and  not transInfo.matched3 :
                Images2.cached[cacheKey] = False
                return False
        Images2.cached[cacheKey] = True    
        return True    
    
    #
    # Challenge Problem B-01 : AC,BC
    # 
    def isIncSameElements(self):
        #print("%d %d" %(len(self.img1Elements),len(self.img2Elements)))
        if len(self.img1Elements)>1 or len(self.img2Elements)<=1:
            return False
        e1 = self.img1Elements[0]
        for e2 in self.img2Elements:
            if not e1.isSimilar(e2):
                return False
        return True    
    
    def isDecSameElements(self):
        if len(self.img2Elements)!=1 or len(self.img1Elements)<=1:
            return False
        e2 = self.img2Elements[0]
        for e1 in self.img1Elements:
            if not e1.isSimilar(e2):
                return False
        return True    

    #
    #  两个图片 都是 两个元素,  位置 互换 , 
    # 待测试, C-09 AC
    #
    def  isImgSame2SwappedElements(self)->bool:
        if len(self.img1Elements)!=2 or len(self.img2Elements)!=2 :
            return
        idxMap = self.getImgElementsEqualsIdxMap() # img1Elements[?] 在 img2Elements 对应的位置
        #print("idxMap = %s" % idxMap)
        if idxMap==None: return False
        img1X1,img1Y1 = self.img1Elements[0].getCenter()
        img2X2,img2Y2 = self.img2Elements[idxMap[1]].getCenter()
        #print("%d,%d : %d,%d " % (img1X1,img1Y1,img2X2,img2Y2))
        if abs(img1X1-img2X2)>2 or abs(img1Y1-img2Y2)>2:
            return False

        img1X2,img1Y2 = self.img1Elements[1].getCenter()
        img2X1,img2Y1 = self.img2Elements[idxMap[0]].getCenter()
        #print("%d,%d : %d,%d " % (img1X2,img1Y2,img2X1,img2Y1))
        #print("图形1 = (%f,%f)  图形2 = (%f,%f) " %(img1X1,img1Y1,img2X2,img2Y2))
        #print("图形1 = (%f,%f)  图形2 = (%f,%f) " %(img1X2,img1Y2,img2X1,img2Y1))
        return abs(img1X2-img2X1)<=2 or abs(img1Y2-img2Y1)<=2

    def getImagePixelRatio(self)->list:
        img1BlackPoints = self.img1.getSumImgElementsBlackPoints()
        img2BlackPoints = self.img2.getSumImgElementsBlackPoints()
        return  -1000 if img2BlackPoints==0 else img1BlackPoints/img2BlackPoints

    def getXORImageElement(self)->ImageElement:
        return ImageElement.getXORImageElement([self.img1.asImgElement(),self.img2.asImgElement()])
    
    def getANDImage(self):
        try:
            return self._andImage
        except AttributeError as e:pass 
        self._andImage = cv2.bitwise_or(self.img1.image,self.img2.image,mask=None)
        #self._andImage = ImageElement.getANDImageElement([self.img1.asImgElement(),self.img2.asImgElement()]).image
        return self._andImage
    
    def getXORImage(self):
        try:
            return self._xorImage
        except AttributeError as e:pass 
        self._xorImage = cv2.bitwise_not(cv2.bitwise_xor(self.img1.image,self.img2.image,mask=None),mask=None)
        #self._andImage = ImageElement.getANDImageElement([self.img1.asImgElement(),self.img2.asImgElement()]).image
        return self._xorImage
    
    def getORImage(self):
        try:
            return self._orImage
        except AttributeError as e:pass 
        self._orImage = cv2.bitwise_and(self.img1.image,self.img2.image,mask=None)
        #self._andImage = ImageElement.getANDImageElement([self.img1.asImgElement(),self.img2.asImgElement()]).image
        return self._orImage
    
    def getAllImagesFilledFlags(self)->int:
        try:
            return self._allImagesFilledFlags
        except AttributeError as e:pass 
        flags = 0
        for imgElements in [self.img1Elements,self.img2Elements]:
            for e in imgElements:
                if e.isFilledImage():
                    flags |= 1
                else:
                    flags |= 2
                if flags==3: break    
            if flags==3: break        
        self._allImagesFilledFlags = flags    
        return flags 
    
    def getIncedElements(self)->int: 
        try:
            return self._incedElements
        except AttributeError as e:pass 
        self._incedElements = self.__getIncedElements()
        return self._incedElements
    
    def __getIncedElements(self)->int: 
        n1 = len(self.img1Elements)
        n2 = len(self.img2Elements)
        if n2<=n1: return None
        idxsInImg2Elements = []
        for e1 in self.img1Elements:
            i = e1.getIndexOfEqElements(self.img2Elements,idxsInImg2Elements,0.85)
            if i<0: return None
            idxsInImg2Elements.append(i)
        incedElements = []
        for i in range(n2):
            if indexOf(idxsInImg2Elements,i)<0: incedElements.append(self.img2Elements[i])
        return incedElements
        """
        def elementsEqs(a1,a2):
            for e1,e2 in zip(a1,a2):
            #    if not e1.isEquals(e2,0.85): return False
                # Challenge E-11 : G[4] H[4] : 0.87
            return True
        if elementsEqs(self.img1Elements,self.img2Elements[0:n1]):
            return self.img2Elements[n1:]
        if elementsEqs(self.img1Elements,self.img2Elements[n2-n1:]):
            return self.img2Elements[0:n2-n1]
        return None
        """    
    
    def isIncedSameElements(self,otherImgs)->bool:
        if len(self.img2Elements)-len(self.img1Elements) != len(otherImgs.img2Elements)-len(otherImgs.img1Elements): 
            return False
        thisInced = self.getIncedElements()
        if thisInced==None: return False
        otherInced = otherImgs.getIncedElements()
        if otherInced==None: return False
        return ImageElement.isElementsEquals2(thisInced,otherInced)

# End class Images2

#
# 描述 一行 或 一列的 的图片
#  如  行 Frame:
#      A   B   C
#      D   E  F
#      G   H  ?
#   列 Frame 
#      A   D   G
#      B   E   H
#      C   F   ?
#   左斜线 ()
#      G  E  C
#      A  F  H
#      D  B  ?   
#   右斜线
#      A  E  ?
#      C  D  H  
#      B  F  G
#      
# 
class Images3:
    #
    # @param  imgId1,imgId2,imgId3 : 一行 或 一列 或 某对角线 上的三个图片ID
    # @frmType :  1 : 行  ( 三个图片 在同行)
    #             2 : 列  ( 三个图片 在同列)
    #             3 : 斜线 ( 三个图片 在一个斜线上)
    # 
    def __init__(self,agent,name):
        self.agent = agent
        self.name = name
        self.imgId1 = name[0]   # 如 "A"
        self.img1 = agent.getImage1(self.imgId1)
        self.imgId2 = name[1]   # 如 "B"
        self.img2 = agent.getImage1(self.imgId2)
        self.imgId3 = name[2]   # 如 ""
        self.img3 = agent.getImage1(self.imgId3)
        self.img1Elements = self.img1.getImageElements() # A 的图片元素
        self.img2Elements = self.img2.getImageElements() # B 的图片元素
        self.img3Elements = self.img3.getImageElements() # C 的图片元素
        self.imgsElementsLst = [self.img1Elements,self.img2Elements,self.img3Elements]
        self.notEqImgElementIdx = -2
        #self.frmType = frmType
        
    #
    # 在 三个图形中 找到 与 otherImgId 相等或相似的 图片序号(0,1,2 之一) 
    #  元素个数相同 且  元素[i]==otherImg.元素[i]
    # @param otherImgId "A","B","1","2",... 等
    # 
    def getIndexOfEqualsImage(self,otherImgId:str,excludeIdxs:list=None) ->int:    
        otherimgElements = self.agent.getImageElements(otherImgId) 
        nElements = len(otherimgElements)
        for i in range(3):
            if indexOf(excludeIdxs,i)>=0 or len(self.imgsElementsLst[i])!=nElements: continue
            img2 =  self.agent.getImages2(self.name[i]+otherImgId)
            # Challenge Problem D-05 : G2 需要使用 getImgElementsEqualsIdxMap 判断
            if img2.isImgElementsEqualsOrSimilar() or img2.getImgElementsEqualsIdxMap()!=None:
                return i
        return -1
    
    #
    # 同 getIndexOfEqualsImage , 但 不考虑 增加的元素
    #
    def getIndexOfEqualsImageIgnoreInced(self,otherImgId:str,excludeIdxs:list=None) ->int:    
        otherimgElements = self.agent.getImageElements(otherImgId) 
        nElements = len(otherimgElements)
        for i in range(3):
            if indexOf(excludeIdxs,i)>=0 : continue
            n = min(len(self.imgsElementsLst[i]),nElements)
            if( self.agent.getImages2(self.name[i]+otherImgId).isImgElementsEqualsOrSimilar(0,n) ):
                return i
        return -1    

    #
    # 在 三个图形中 找到 与 otherImgId 外形相似的 图片序号(0,1,2 之一)
    # @param otherImgId "A","B","1","2",... 等
    # 
    def getIndexOfOutterSharpEqImage(self,otherImgId:str,excludeIdxs:list=None) ->int:    
        otherimgElements = self.agent.getImageElements(otherImgId) 
        nElements = len(otherimgElements)
        for i in range(3):
            if indexOf(excludeIdxs,i)<0 and len(self.imgsElementsLst[i])==nElements and self.agent.getImages2(self.name[i]+otherImgId).isImgElementsOutterSharpMatched():
                return i
        return -1
    
    #
 
    #
    # @return 1 : 图1+图2==图3
    #         2 : 图1-图2==图3
    #         0 : 图1 == 图2 == 图3
    #
    def compareImgPixelCount(self):
        img1BlackPoints = self.img1.getSumImgElementsBlackPoints()
        img2BlackPoints = self.img2.getSumImgElementsBlackPoints()
        img3BlackPoints = self.img3.getSumImgElementsBlackPoints()
        return _compare3(img1BlackPoints,img2BlackPoints,img3BlackPoints)
    
    
    def compareImgPixelHeight(self)->bool:
        h1 = self.img1.getImgElementsHeight()
        h2 = self.img2.getImgElementsHeight()
        h3 = self.img3.getImgElementsHeight()
        return _compare3(h1,h2,h3)
    
    def compareImgPixelWidth(self)->bool:
        w1 = self.img1.getImgElementsWidth()
        w2 = self.img2.getImgElementsWidth()
        w3 = self.img3.getImgElementsWidth()
        return _compare3(w1,w2,w3)
    
    #
    # 返回 A/B, B/C , A/C 的像素比例
    #
    def getImagePixelRatio(self)->list:
        img1BlackPoints = self.img1.getSumImgElementsBlackPoints()
        img2BlackPoints = self.img2.getSumImgElementsBlackPoints()
        img3BlackPoints = self.img3.getSumImgElementsBlackPoints()
        return [img1BlackPoints if img2BlackPoints==0 else img1BlackPoints/img2BlackPoints, \
                img2BlackPoints if img3BlackPoints==0 else img2BlackPoints/img3BlackPoints, \
                img1BlackPoints if img3BlackPoints==0 else img1BlackPoints/img3BlackPoints \
                ]
    
    #
    # 如果所有元素 X 族 中心点 相同, 返回该 中心点, 否则 返回 -1
    #
    def allElementsInCenterX(self)->int:
        x1 = ImageElement.allElementsInCenterX(self.img1Elements)
        if x1<0:
            return -1
        x2 = ImageElement.allElementsInCenterX(self.img2Elements)
        if x2<0 or abs(x1-x2)>2:
            return -1
        x3 = ImageElement.allElementsInCenterX(self.img3Elements)
        if x3<0 or abs(x1-x3)>2:
            return -1
        return (x1+x2+x3) / 3
    def allElementsInCenterY(self)->int:
        y1 = ImageElement.allElementsInCenterY(self.img1Elements)
        if y1<0:
            return -1
        y2 = ImageElement.allElementsInCenterY(self.img2Elements)
        if y2<0 or abs(y1-y2)>2:
            return -1
        y3 = ImageElement.allElementsInCenterY(self.img3Elements)
        if y3<0 or abs(y1-y3)>2:
            return -1
        return (y1+y2+y3) / 3

    #
    #   D-06 ABC
    #  ABC 中 ,  第 ? 组元素 不 相等
    #
    def getNotEqImgElementIdx(self)->int:
        if self.notEqImgElementIdx>-2:
            return self.notEqImgElementIdx
        #print("....")
        n = len(self.img1Elements)
        if n!=len(self.img2Elements) and n!=len(self.img3Elements):
            self.notEqImgElementIdx = -1
            return -1
        #imgAB = self.agent.getImage2(self.imgId1+self.imgId2)
        #imgAC = self.agent.getImage2(self.imgId1+self.imgId3)
        k = -1
        for i in range(n):
            similar1,_,_,scale1 = self.img1Elements[i].getImageElementSimilarScale(self.img2Elements[i])
            eq = similar1>=0.90 and abs(scale1-1)<0.05  # AB 的 第 i 个元素 相等
            if eq:
                similar2,_,_,scale2 = self.img1Elements[i].getImageElementSimilarScale(self.img3Elements[i])
                eq = similar2>=0.90 and abs(scale2-1)<0.05 # AC 的 第 i 个元素 相等
            if not eq:
                if k>=0:
                    self.notEqImgElementIdx = -1        
                    return -1
                k = i
        self.notEqImgElementIdx = k        
        return k
    
    #
    # get XOR Image
    #
    #def getXORImageElement(self,elementIdx:int)->ImageElement:
    #    return ImageElement.getXORImageElement([self.img1Elements[elementIdx],self.img2Elements[elementIdx],self.img3Elements[elementIdx]])
    def getXORImageElement(self)->ImageElement:
        return ImageElement.getXORImageElement([self.img1.asImgElement(),self.img2.asImgElement(),self.img3.asImgElement()])

    #
    # @return 1: A and B == C
    #         2: A xor B == C   
    #
    def getBitOPMatched(self)->int:
        try:
            return self._andMatched
        except AttributeError as e:pass 
        self._andMatched = 0 
        imgsAB = self.agent.getImages2(self.imgId1+self.imgId2) 
        diffRatio,diffCount,_ = countImageDiffRatio(imgsAB.getANDImage(),self.agent.getImage(self.imgId3))
        #print("[%s - %s] diffRatio=%f,diffCount=%d" %(imgsAB.name,self.imgId2,diffRatio,diffCount))
        if diffRatio<0.01 :
            self._andMatched = 1
            return self._andMatched
        diffRatio,diffCount,_ = countImageDiffRatio(imgsAB.getXORImage(),self.agent.getImage(self.imgId3))
        #print("[%s - %s] diffRatio=%f,diffCount=%d" %(imgsAB.name,self.imgId2,diffRatio,diffCount))
        if diffRatio<0.01 :
            self._andMatched = 2
            return self._andMatched
        
        diffRatio,diffCount,_ = countImageDiffRatio(imgsAB.getORImage(),self.agent.getImage(self.imgId3))
        #img3 = self.agent.getImage1(self.imgId3)
        #count3 = countImageDiffCaseNeighbor(imgsAB.getORImage(),self.agent.getImage(self.imgId3)) 
        #print("[%s - %s] diffRatio=%f,diffCount=%d %s.blackPixelCount = %d; %d" %(imgsAB.name,self.imgId2,diffRatio,diffCount,img3.name,img3.asImgElement().blackPixelCount,count3))
        #print("[OR-%s  %s] diffRatio=%f,diffCount=%d" %(imgsAB.name,self.imgId2,diffRatio,diffCount))
        if diffRatio<0.01 :
            self._andMatched = 3
            return self._andMatched
        elif diffRatio<0.07 :
            # Challenge E-02","ABC" 比较的结果 , 0.044187;  DEF : 0.064184 ; 使用 countImageDiffCaseNeighbor 
            countDiffCaseNeighbor = countImageDiffCaseNeighbor(imgsAB.getORImage(),self.agent.getImage(self.imgId3)) 
            #print("[OR-%s  %s] countDiffCaseNeighbor = %d" %(imgsAB.name,self.imgId2,countDiffCaseNeighbor))
            if countDiffCaseNeighbor<=30:   
                # Challenge E-02 : GH7 countDiffCaseNeighbor==26
                self._andMatched = 3
                return self._andMatched
        return self._andMatched    
    
    def getAllImagesFilledFlags(self)->int:
        try:
            return self._allImagesFilledFlags
        except AttributeError as e:pass 
        flags = 0
        for imgElements in [self.img1Elements,self.img2Elements,self.img3Elements]:
            for e in imgElements:
                if e.isFilledImage():
                    flags |= 1
                else:
                    flags |= 2
                if flags==3: break    
            if flags==3: break        
        self._allImagesFilledFlags = flags    
        return flags 
    
    #
    # 判断 第 elementIdx 项 元素 是否全相等
    #
    def isEqualsByElementIdx(self,elementIdx)->bool:
        cacheKey = self.name+".isEqualsByElementIdx("+str(elementIdx)+")"
        if not cacheKey in Images3.cached:
            Images3.cached[cacheKey] = self.img1Elements[elementIdx].isEqualsAllElements([self.img2Elements[elementIdx],self.img3Elements[elementIdx]])
        return Images3.cached[cacheKey]
    
    #
    # 元素个数相同, 且 只有 一个 元素 不同时, 返回 该元素序号,
    #   -1 : 全等 -2 : 元素个数不等; -3: 至少两个不等
    #   
    # 
    def getOnlyNotEqElementIdx(self)->int:
        try:
            return self._onlyNotEqElementIdx
        except AttributeError as e:pass 
        n = len(self.img1Elements)
        if n!=len(self.img2Elements) or n!=len(self.img3Elements):
            self._onlyNotEqElementIdx = -2
            return -2
        j = -1
        for i in range(n):
            if self.isEqualsByElementIdx(i): continue
            if j>=0 : 
               self._onlyNotEqElementIdx = -3
               return -3
            j = i
        return j
    
    def getImgElementEqualsIdxMap(self,elementIdx:int,otherImg3,otherImg3ElementIdx:int)->list:
        idxMap = [] 
        e2Lst = [otherImg3.img1Elements[otherImg3ElementIdx],otherImg3.img2Elements[otherImg3ElementIdx],otherImg3.img3Elements[otherImg3ElementIdx]]
        for e1 in [self.img1Elements[elementIdx],self.img2Elements[elementIdx],self.img3Elements[elementIdx]]:
            i = e1.getIndexOfEqElements(e2Lst,idxMap)
            if i<0: return None
            idxMap.append(i)
        return idxMap
    
    #
    # 三个元素使用 不同的 填充模式
    #
    def isDifferentFillMode(self,elementIdx:int)->bool:
        try:
            return self._isDifferentFillMode
        except AttributeError as e:pass 
        def _getFillMode(e):
            if e.isFilledImage(): return 1
            if e.isLinesFielldImage() : return 2
            return 0
        m1 = _getFillMode(self.img1Elements[elementIdx])
        m2 = _getFillMode(self.img2Elements[elementIdx])
        if m2==m1:
            self._isDifferentFillMode = False
            return False
        m3 = _getFillMode(self.img3Elements[elementIdx])
        self._isDifferentFillMode = m3!=m1 and m3!=m2
        return self._isDifferentFillMode


 
# END class Images3   
     
class AnswerScoreDetail:     
    def __init__(self,score:float,imgs1Name:str,imgs2Name:str,type:int,desc:str,step):
        self.score = score
        self.imgs1Name = imgs1Name
        self.imgs2Name = imgs2Name
        self.type = type
        self.desc = desc
        self.step = step
        self.flags = 0
    

#
#  某个答案 对应的 可行度得分
#  成员 int answer : 答案, 序号, 如 1,2, 等
#  成员 float score : 可行度分数 
#         
class AnswerScore:
    def __init__(self,answer,score=0):
        self.answer = answer
        self.score = score
        self.scoreDetails = [] 
    def addScore(self,score:float,imgs1Name:str,imgs2Name:str,type:int,desc:str,step=0):
        self.score += score
        self.scoreDetails.append( AnswerScoreDetail(score,imgs1Name,imgs2Name,type,desc,step))
        #((score,imgs1Name,imgs2Name,type,desc,"" if step==0 else "(附加分)"))    
    def execludeScores(self,accept):
        for d in  self.scoreDetails :
            if  accept(d) and (d.flags&1) ==0:
                d.flags |= 1
                self.score -= d.score
    def getMaxScoreAnswers(answers):
        #answers.sort(key=lambda e:e.score,reverse=True)
        maxScore = max(map(lambda s:s.score,answers))
        maxScoreAnswers = []
        for ansert in answers:
            if ansert.score==maxScore:
                maxScoreAnswers.append(ansert)
        return  maxScoreAnswers      

    
# END class AnswerScore
#
class Agent:
    _DEBUG = False
    _WARN = False
    def __init__(self):
        """
        The default constructor for your Agent. Make sure to execute any processing necessary before your Agent starts
        solving problems here. Do not add any variables to this signature; they will not be used by main().

        This init method is only called once when the Agent is instantiated
        while the Solve method will be called multiple times.
        """
        

    def  getImage(self,imageId:str):
        return self.getImage1(imageId).image
    
    def getImage1(self,imageId:str)->Image1:
        if len(imageId)!=1:
            raise BaseException("Invalie imageId = ",imageId)
        return self.images[imageId]
    #
    # @param imageId image id,  如 "A", "B", "C", "1", "2" 等
    # @return 返回 数组: [元素1,元素2,...]
    #
    def  getImageElements(self,imageId:str) ->list:
        return self.getImage1(imageId).getImageElements()
    
    #
    #  getImages2("A","C")
    #   返回 Images2
    #
    def getImages2(self,name:str)->Images2:
        if name in self.imagesFrame:
            #print("使用缓存 %s" % name)
            return self.imagesFrame[name]
        imgFrame = Images2(self,name)
        self.imagesFrame[name] = imgFrame
        return imgFrame
    
    #
    #  返回 一行 或 一列 或 一个 斜线 上 三个图片组成的 图片Frame
    #
    def  getImages3(self,name:str)->Images3:
        if name in self.imagesFrame:
            #print("使用缓存 %s" % name)
            return self.imagesFrame[name]   
        imgFrame = Images3(self,name)
        self.imagesFrame[name] = imgFrame
        return imgFrame 

    SCORETYPE2_TRANSMATCHED  = 0x01000
    SCORETYPE2_OUTTERMATCHED = 0x02000
    SCORETYPE2_FULLROTATE = 0x03000
    SCORETYPE2_ALLFILLED = 0x04000
    SCORETYPE2_ROTATE = 0x05000
    SCORETYPE2_PIXCHANED = 0x0600
    SCORETYPE2_OUTTERSIMILAR = 0x0700
    SCORETYPE2_VERTICES = 0x0700
    #
    #  计算 2x2 的图形 两行 或  两列 之间 属性匹配程度 的 得分
    #     原理是   图A -> 图B 之间某些元素使用了某一个转换规则(IMGTRANSMODE_EQ,IMGTRANSMODE_FLIPH,IMGTRANSMODE_FILLED 等)变换等到
    #       如果   图C -> 图? 之间 如果使用了 相同的 转换规则
    #        则    匹配程度 的 得分 越高
    #       
    #  例如 调用 self.calculateImages2MatchScore("AB","C1")
    #    根据    图A -> 图B 的 变换规则, 
    #     得到   图C -> 答案1 的 得分
    # @param imgs1Name,imgs2Name:  一行 或 一列 或 对角线 的 两个图 ,如 "AB","AC", "C1" ,"B1" 等
    # @param scoreAddTo 得分结果 累加到 scoreAddTo 中
    #    
    def calculateImages2MatchScore(self,imgs1Name,imgs2Name,scoreAddTo:AnswerScore,scoreWeight=1): 
        imgsFrm1 = self.getImages2(imgs1Name)
        imgsFrm2 = self.getImages2(imgs2Name)
        elementCountDiff1 = imgsFrm1.getImgElementsCountDiff()  # count(B) - count(A)
        elementCountDiff2 = imgsFrm2.getImgElementsCountDiff()  # count(?) - count(C)
        minElementCount = min(imgsFrm1.getImgElementCount(),imgsFrm2.getImgElementCount())
        scoreFactor = 1 if minElementCount<=1 else 1 / minElementCount
        if imgsFrm1.getImgElementCount()!=imgsFrm2.getImgElementCount()  or  len(imgsFrm1.img1Elements)!= len(imgsFrm1.img2Elements)  or  len(imgsFrm2.img1Elements)!= len(imgsFrm2.img2Elements):  # B-12 , 不等的情况下, 降低分数系数
            if elementCountDiff1==elementCountDiff2:
                # B-12
                scoreFactor /= 4
            else:
                # Challenge Problem B-05 的 权重太大
                scoreFactor /= 10
        #print("[%s-%s] scoreFactor=%f,minElementCount=%f = min(%d,%d); elementCountDiff1=%d,elementCountDiff2=%d" % (imgs1Name,imgs2Name,scoreFactor,minElementCount,imgsFrm1.getImgElementCount(),imgsFrm2.getImgElementCount(),elementCountDiff1,elementCountDiff2))
        caseCheckFilled = True
        caseCheckRota45 = True
        caseOuterSharpCmp = True
        caseVertices = True
        allElementsMatchedTransMode = []  # indexed by elementIdx
        for elementIdx in range( minElementCount ):
            elementsMatchedTransMode = []
            allElementsMatchedTransMode.append(elementsMatchedTransMode)
            eqTrans = imgsFrm1.getImgElementTrans(elementIdx,IMGTRANSMODE_EQ)
            if eqTrans.matched:  # 如果 两个 图片 相等, 不再判断 其他 转换:
                 # 不能加 or eqTrans.matched2, 否则 : B-06 的 AC 就 不会 比较 翻转了
                forAllTrans = [eqTrans]
            else:
                forAllTrans = imgsFrm1.getAllImgElementTrans(elementIdx,[IMGTRANSMODE_EQ,IMGTRANSMODE_FLIPV,IMGTRANSMODE_FLIPH,IMGTRANSMODE_FILLED,IMGTRANSMODE_UNFILLED]) 
            #
            # 以上 相等, 翻转, 填充 都不满足的情况下 才 考虑 旋转 的情况, 
            #
            caseRotate = True
            for transInfo in forAllTrans:
                if transInfo.matched or transInfo.matched2:
                    caseRotate = False
                    break
            if caseRotate:
                for transMode in [IMGTRANSMODE_ROTATE090,IMGTRANSMODE_ROTATE270,IMGTRANSMODE_ROTATE180]:
                    t = imgsFrm1.getImgElementTrans(elementIdx,transMode)
                    if t.matched:
                        forAllTrans.append(t)
                        break
            for transInfo in forAllTrans:  # r similar,scale
                #if Agent._DEBUG and imgs2Name=="C6" and imgs1Name=="AB":
                #    print("%s-%s: imgs1.transMode=%s,%s:%f,%s:%f " %(imgs1Name,imgs2Name,transInfo.transMode,transInfo.matched,transInfo.similar,transInfo.matched2,transInfo.similar2))
                if not transInfo.matched and not transInfo.matched2:
                    continue
                caseCheckRota45 = False
                caseOuterSharpCmp = False
                caseVertices = False
                #if Agent._DEBUG and imgs2Name=="C6":
                #    print("%s : 满足变换规则 %s" %(imgs1Name,transInfo.transMode))
                transInfo2 = imgsFrm2.getImgElementTrans(elementIdx,transInfo.transMode)
                #if Agent._DEBUG and imgs2Name=="B3":
                #    print("  %s : 检测是否满足变换规则 %s 结果matched = %s, similar=%f " %(imgs2Name,transInfo.transMode,transInfo2.matched,transInfo2.similar))
                if not transInfo2.matched and not transInfo2.matched2:
                    continue
                elementsMatchedTransMode.append(transInfo.transMode)
                score = 10 
                # 翻转, 的情况下, 如果 都是基于 整个 图 反转, 加分
                desc2 = ""
                if transInfo.transMode==IMGTRANSMODE_FLIPV or transInfo.transMode==IMGTRANSMODE_FLIPV or transInfo.transMode==IMGTRANSMODE_FLIPVH:
                    # 如果同时基于 整图 翻转, 加分
                    if imgsFrm1.isWholeImgElementFliped(elementIdx,transInfo.transMode) and  imgsFrm2.isWholeImgElementFliped(elementIdx,transInfo.transMode):
                        score += 3 
                        desc2 += "(基于整图翻转)"
                elif transInfo.transMode==IMGTRANSMODE_EQ:
                    # 正方形, 圆形 等对称图形, 如果 同时 基于整图翻转 , 加分
                    if (imgsFrm1.isWholeImgElementFliped(elementIdx,IMGTRANSMODE_FLIPV) and  imgsFrm2.isWholeImgElementFliped(elementIdx,IMGTRANSMODE_FLIPV))\
                    or (imgsFrm1.isWholeImgElementFliped(elementIdx,IMGTRANSMODE_FLIPH) and  imgsFrm2.isWholeImgElementFliped(elementIdx,IMGTRANSMODE_FLIPH)):
                        # B-05
                        score += 3 
                        desc2 += "(基于整图翻转)"
                if not transInfo.matched or not transInfo2.matched:
                    # Challenge B-07
                    scoreFactor *= 0.5
                    desc2 += "(仅轮廓相似)"
                scoreAddTo.addScore(score*scoreFactor*scoreWeight,imgs1Name,imgs2Name,Agent.SCORETYPE2_TRANSMATCHED,"元素%d匹配相同变换%s%s"%(elementIdx,transInfo.transMode,desc2))
                caseCheckFilled = False

        #
        # [Challenge Problem B-09]  判断外形匹配
        #
        if caseOuterSharpCmp:
            if imgsFrm1.isImgElementsOutterSharpMatched() and imgsFrm2.isImgElementsOutterSharpMatched():
                caseVertices = False
                score = 4
                desc2 = ""
                if imgsFrm1.getImgElementCount()==1 and imgsFrm2.getImgElementCount()==1:
                    def _getFillMode(e):
                        if e.isFilledImage(): return 1
                        if e.isLinesFielldImage() : return 2
                        return 0                
                    #fillModeA = _getFillMode(imgsFrm1.img1Elements[0])
                    if _getFillMode(imgsFrm1.img1Elements[0])==_getFillMode(imgsFrm2.img1Elements[0]) \
                    and _getFillMode(imgsFrm1.img2Elements[0])==_getFillMode(imgsFrm2.img2Elements[0]):
                            score += 1
                            desc2 = ",且填充模式一致"
                    #filledFlags1 =  imgsFrm1.getAllImagesFilledFlags()
                    #if (filledFlags1==1 or filledFlags1==2):
                    #    filledFlags2 =  imgsFrm2.getAllImagesFilledFlags()
                    #    if (filledFlags2==1 or filledFlags2==2):
                    #        score += 1
                    #        desc2 = ",且填充模式一致"
                scoreAddTo.addScore(score*scoreWeight,imgs1Name,imgs2Name,Agent.SCORETYPE2_OUTTERMATCHED,"外形相似"+desc2)
        #
        # Challenge Problem B-10 : 如果 整图 旋转  AB/C4 
        #
        if elementCountDiff1==elementCountDiff2  and elementCountDiff1==0 and imgsFrm1.getImgElementCount()==imgsFrm2.getImgElementCount() and minElementCount>1:
            checkAllRota = None
            for transMode in [IMGTRANSMODE_ROTATE090,IMGTRANSMODE_ROTATE180,IMGTRANSMODE_ROTATE270]:
                allEleMatched = True
                for elementIdx in range( minElementCount ):
                    if transMode not in allElementsMatchedTransMode[elementIdx]:
                        allEleMatched = False
                        break
                if allEleMatched:
                    checkAllRota = transMode
                    break    
            if checkAllRota!=None:
                #print("检查 %s/%s 是否满足整体旋转 %d ..." %(imgs1Name,imgs2Name,checkAllRota))        
                if imgsFrm1.img1.getRotateImage(checkAllRota).isEquals(imgsFrm1.img2.asImgElement()) and imgsFrm2.img1.getRotateImage(checkAllRota).isEquals(imgsFrm2.img2.asImgElement()):
                    scoreAddTo.addScore(3*scoreWeight,imgs1Name,imgs2Name,Agent.SCORETYPE2_FULLROTATE,"整图旋转")

       #
        #  Challenge Problem B-04 : 区答案 3,4
        #     
        if caseCheckFilled:
            filledFlags =  imgsFrm1.getAllImagesFilledFlags()
            if (filledFlags==1 or filledFlags==2) and filledFlags==imgsFrm2.getAllImagesFilledFlags():
                scoreAddTo.addScore(1*scoreWeight,imgs1Name,imgs2Name,Agent.SCORETYPE2_ALLFILLED, "两组元素全为填充图" if filledFlags==1 else "两组元素全为非填充图") 
                
        #
        # Challenge Problem B-02 : 区答案 1,2,6 
        #
        if caseCheckRota45 and elementCountDiff1==0 and elementCountDiff2==0 and  len(imgsFrm1.img1.getImageElements())==1:
            for rotaMode in ["ROTATE315","ROTATE045","ROTATE135","ROTATE225"]:
                rotaImg1 = imgsFrm1.img1.asImgElement().getRotateImage(rotaMode)
                if not imgsFrm1.img2.asImgElement().isSimilar(rotaImg1): continue #similarTh=0.90
                rotaImg2 = imgsFrm2.img1.asImgElement().getRotateImage(rotaMode)
                if  imgsFrm2.img2.asImgElement().isSimilar(rotaImg2):
                    scoreAddTo.addScore(10*scoreWeight,imgs1Name,imgs2Name,Agent.SCORETYPE2_ROTATE,"满足旋转"+rotaMode[6:]+"度")
                    break
                    
        #
        # 考虑 元素 增加 / 减少 的规则:
        #        
        if elementCountDiff1==elementCountDiff2 : # A-B 的元素增加 == ? -C 的元素增加
            #scoreDesc.append((1 * scoreWeight,"[%s-%s]两组元素增减个数相同"),imgs1Name,imgs2Name)
            if len(imgsFrm1.img1Elements)==len(imgsFrm2.img1Elements):  # A 与 C 的 图形元素 个数相同
                score = 3
                desc = "两组元素个数匹配,增减个数相同"
                # 再检查, 是否 变动的 元素 相同:
                if elementCountDiff1>0:   # A->B , C->? 图形元素增加 相同元素
                    imgFrameBD = self.getImages2(imgs1Name[1]+imgs2Name[1])  # Frame  B?
                    if imgFrameBD.isImgElementsEqualsOrSimilar(len(imgsFrm1.img1Elements)) : # C 与 答案 新加的元素 相同 
                        score += 7
                        desc += ",且增加了相同类型的元素"
                        #scoreAddTo.addScore(7* scoreWeight,imgs1Name,imgs2Name,"两组增加了相同类型的元素");   
                    elif imgsFrm1.isIncSameElements() and imgsFrm2.isIncSameElements():                 # Challenge Problems B-01 : AC-B6
                        score += 7
                        desc += ",且增加了同倍数的相同元素"
                        #scoreAddTo.addScore(7* scoreWeight,imgs1Name,imgs2Name,"两组增加了同倍数的相同元素");   
                elif elementCountDiff2<0: # C-? 图形元素减少
                    imgFrameAC = self.getImages2(imgs1Name[0]+imgs2Name[0])  # Frame  AC 
                    if imgFrameAC.isImgElementsEqualsOrSimilar(len(imgsFrm1.img2Elements)) : # C 与 答案 的元素 相同 
                        score += 7
                        desc += ",且减少了相同类型的元素"
                        #scoreAddTo.addScore(7* scoreWeight,imgs1Name,imgs2Name,"两组减少了相同类型的元素")
                    elif imgsFrm1.isIncSameElements() and imgsFrm2.isIncSameElements():                 
                        score += 7
                        desc += ",且减少了同倍数的相同元素"
                        #scoreAddTo.addScore(7* scoreWeight,imgs1Name,imgs2Name,"两组减少了同倍数的相同元素");       
                scoreAddTo.addScore(score* scoreWeight,imgs1Name,imgs2Name,score,desc);    
            else:  #A 与 C 的 图形元素 个数不同
                scoreAddTo.addScore(2* scoreWeight,imgs1Name,imgs2Name,Agent.SCORETYPE2_PIXCHANED,"两组元素增减个数相同")
                # Challenge Problem B-03 : 需要 较高的 分数, 

            """
            filledFlags =  ImageElement.getImagesFilledFlags(imgsFrm1.img1Elements) #[elementIdx].isFilledImage()
            if (filledFlags==1 or filledFlags==2 )  \
                and filledFlags==ImageElement.getImagesFilledFlags(imgsFrm2.img1Elements) \
                and filledFlags==ImageElement.getImagesFilledFlags(imgsFrm1.img2Elements) \
                and filledFlags==ImageElement.getImagesFilledFlags(imgsFrm2.img2Elements) :
                    scoreAddTo.addScore(1*scoreWeight,imgs1Name,imgs2Name,"两组元素全为填充图" if filledFlags==1 else "两组元素全为非填充图") 
            """            
        #
        # Challenge B-08
        #
        if  caseVertices and elementCountDiff1==0  and elementCountDiff2==0 and minElementCount==1 :
            #print("检测 多边形 ... %s-%s" %(imgs1Name,imgs2Name))
            verticesA = len(imgsFrm1.img1Elements[0].getPolygonPoints())
            verticesB = len(imgsFrm1.img2Elements[0].getPolygonPoints())
            if verticesA!=verticesB :
                verticesC = len(imgsFrm2.img1Elements[0].getPolygonPoints())
                if  verticesA!=verticesC and verticesA-verticesB==verticesC-len(imgsFrm2.img2Elements[0].getPolygonPoints()):
                    scoreAddTo.addScore(1*scoreWeight,imgs1Name,imgs2Name,Agent.SCORETYPE2_VERTICES,"图形顶点数变化趋势一致")
            


        #
        # Challenge Problem B-04 : 区答案 4,6
        #
        for elementIdx in range( minElementCount ):
            if len(allElementsMatchedTransMode[elementIdx])>0:
                continue
            eqTrans1 = imgsFrm1.getImgElementTrans(elementIdx,IMGTRANSMODE_EQ)
            eqTrans2 = imgsFrm2.getImgElementTrans(elementIdx,IMGTRANSMODE_EQ)
            if not eqTrans1.matched and not eqTrans2.matched and eqTrans1.matched3 and eqTrans2.matched3:  # 外轮廓匹配 , 且等
                r1 = imgsFrm1.img2Elements[elementIdx].blackPixelCount / imgsFrm1.img1Elements[elementIdx].blackPixelCount 
                r2 = imgsFrm2.img2Elements[elementIdx].blackPixelCount / imgsFrm2.img1Elements[elementIdx].blackPixelCount 
                #print("%s : matched=%s matched3=%s r1=%f ; %s : matched=%s matched3=%s r2=%f" %(imgsFrm1.name, eqTrans1.matched,eqTrans1.matched3,r1,imgsFrm2.name, eqTrans2.matched,eqTrans2.matched3,r2))
                if (r1>=2 and r2>=2) or (r1<=0.5 and r2<=0.5):
                    scoreAddTo.addScore(0.5*scoreFactor*scoreWeight,imgs1Name,imgs2Name,Agent.SCORETYPE2_OUTTERSIMILAR,"元素%d轮廓相似, 像素个数变化趋势相近"%(elementIdx))
            #if eqTrans1.matched!=eqTrans2.matched:

            #scoreFactor
        

    def calculateImages2MatchScore_2(self,imgs1Name,imgs2Name,scoreAddTo:AnswerScore,scoreWeight=1): 
        imgsFrm1 = self.getImages2(imgs1Name)
        imgsFrm2 = self.getImages2(imgs2Name)
        #
        # 判断像素 比例 变化规律
        #
        r1 = imgsFrm1.getImagePixelRatio()
        r2 = imgsFrm2.getImagePixelRatio()
        diff = abs(r1-r2)
        if  diff< 0.05:
            scoreAddTo.addScore(1,imgs1Name,imgs2Name,0,"两图片素个数变化率相差<0.05",1) 
        elif diff < 0.1:
            scoreAddTo.addScore(0.5,imgs1Name,imgs2Name,0,"两图片像素个数变化率相差<0.1",1) 
        elif diff < 0.15:
            scoreAddTo.addScore(0.2,imgs1Name,imgs2Name,0,"两图片像素个数变化率相差<0.15",1) 



    #END method calculateImages2MatchScore
    
    SCORETYPE3_EQ1               =     0x01000  # 相同组合
    SCORETYPE3_EQ1_ALL           =     0x00100  # 全相等
    SCORETYPE3_EQ1_ALLSIMILAR    =     0x00200  # 全相似
    SCORETYPE3_EQ1_1             =     0x00001   #A==C,D==F,G==3 
    SCORETYPE3_EQ2               =     0x02000  # ABC 相等, GHI 相等
    SCORETYPE3_ECCHANGE          =     0x03000  # 元素变化相同guilv1
    SCORETYPE3_ECCHANGE_EQ       =     0x00100  
    SCORETYPE3_ECCHANGE_MUTIPLE  =     0x00100   # 
    SCORETYPE3_EQ3               =     0x04000   # 某元素相同组合
    SCORETYPE3_SUBELEMENTS       =     0x04000  # 子集关系
    SCORETYPE3_SIZECHANGE        =     0x05000  # 面积变化
    SCORETYPE3_ROTATE            =     0x06000  # 旋转 
    SCORETYPE3_FLIP              =     0x07000 
    SCORETYPE3_EQ3               =     0x08000  #第一组中三个图片分别等于第二组中三个图片的每个元素
    SCORETYPE3_OUTEREQ1          =     0x09000  # 两组图形外形具有相同组合
    SCORETYPE3_INCEDSAME         =     0x0A000  #两组图形具有相同组合,并同时增加相同个数(%d)元素,对应序号=%
    SCORETYPE3_ECCHANGE2         =     0x0B000   #两组元素个数(不考虑次序的情况下)匹配
    SCORETYPE3_MERGE             =     0x0C000   # A+B==C  第%d和%d图片像素合并==第%d个图片  
    SCORETYPE3_MERGE_LR          =     0x00100   #   
    SCORETYPE3_BITOP             =     0x0D000
    SCORETYPE3_XOR               =     0x0E000
    SCORETYPE3_VERTICES          =     0x0F000
    SCORETYPE3_LRMOVED           =     0x10000 # C-09
    #
    # 计算 两帧 图片(如 ABC 与 GH1) 之间 属性匹配程度的 得分
    #    self.calculateImages3MatchScore("ABC") 
    # @param imgs1Name,imgs2Name:  一行 或 一列 或 对角线 的 三个图 ,如 "ABC","ADG", "GH1" ,"CF1" 等
    # @param scoreAddTo 得分结果 累加到 scoreAddTo 中
    #
    def calculateImages3MatchScore(self,imgs1Name:str,imgs2Name:str,scoreAddTo:AnswerScore,scoreWeight=1):
        #print("calculateImages3MatchScore %s %s" %(imgs1Name,imgs2Name))
        imgsFrm1 = self.getImages3(imgs1Name)
        imgsFrm2 = self.getImages3(imgs2Name)
        #A,B,C = imgsFrm1.imgId1,imgsFrm1.imgId2,imgsFrm1.imgId3
        #G,H,I = imgsFrm2.imgId1,imgsFrm2.imgId2,imgsFrm2.imgId3
        #
        # 两行( 或 两列) 图片是否是 同组合类型的图片
        #  例如: ABC    ==  圆    三角   正方
        #       GHI    == 三角   正方    圆
        #  这两行 同组合类型, 只是可能 次序不同
        #
        idxOfImgs2 = []  # 第二组图片 对应在 第一组 图片中的序号 
        for i in range(3):
            j = imgsFrm1.getIndexOfEqualsImage(imgs2Name[i],idxOfImgs2)  # 第二组图片的 第 i 个图片, 在 第一组 中对应的序号 ( 不存在时 j==-1 )
            if j<0:
                break
            idxOfImgs2.append(j)
        all6ImgEquals = False    
        caseAddOrSubEq = True  # 
        caseXorEq = True
        caseOuterSharpCmp = True  # 判断外形  
        caseXorCmp = True
        caseBitOPCmp = True  # a and b==c, a xor b==c 等
        caseCheckEqsIgnoreInced = True
        caseCheckLastElementEqs = True  # Basic Problem D-09 , 其他元素相等, 只有 最后一个元素 具有相同组合
        caseTransMatched = True
        caseWholeFliped = True
        caseCheckIncedEq = True #  A  G 的每个元素==A, Challenge Problem D-10 : ADG / CF7
        caseElementCountInc1Mathched = True 
        caseVertices = True # 检测 多变形
        caseMovedMerge = True  # C-09 : A+C => B 
        #  
        # 判断 两组 图片 为 相同组合 或 完成 相同
        # 例子:  
        #   C-02 (全相同) :   D-02-不同组合   
        #
        if len(idxOfImgs2)==3:
            #print("%s-%s : idxOfImgs2 = %s" %(imgs1Name,imgs2Name,idxOfImgs2))
            score = 7  # 10 降为 7 Challenge D-04 : [DEF-GH2] 
            scoreType = Agent.SCORETYPE3_EQ1
            desc = "两组图形具有相同组合"  
            #scoreAddTo.addScore(10 * scoreWeight ,imgs1Name,imgs2Name,"两组图形具有相同组合")
            caseAddOrSubEq = False
            caseXorEq = False
            caseOuterSharpCmp = False
            caseXorCmp = False
            caseBitOPCmp = False
            caseCheckEqsIgnoreInced = False
            caseCheckLastElementEqs = False
            caseTransMatched = False
            caseWholeFliped = False
            caseCheckIncedEq  = False
            caseElementCountInc1Mathched = False
            caseVertices = False
            caseMovedMerge = False
            if self.getImages2(imgs2Name[0:2]).isImgElementsEqualsOrSimilar() and self.getImages2(imgs2Name[1:3]).isImgElementsEqualsOrSimilar():
                # GH? 图形相同, ABC 也相同 ( 因为 同组合 ) : A==B and B==C
                if self.getImages2(imgs2Name[0:2]).isImgElementsEquals() and self.getImages2(imgs2Name[1:3]).isImgElementsEquals():
                    score += 12
                    desc = "两组图形6个全相同"  
                    scoreType |= Agent.SCORETYPE3_EQ1_ALL
                else:                    
                    score += 10
                    desc = "两组图形6个全相似"  
                    scoreType |= Agent.SCORETYPE3_EQ1_ALLSIMILAR
                #scoreAddTo.addScore(10 * scoreWeight ,imgs1Name,imgs2Name,"两组图形6个全相同")
                all6ImgEquals = True
            elif idxOfImgs2[0]==0 and idxOfImgs2[1]==1 and idxOfImgs2[2]==2:
                score += 3
                scoreType |= Agent.SCORETYPE3_EQ1_1
                desc +=",且 %s==%s,%s==%s,%s==%s " %(imgs1Name[0],imgs2Name[0],imgs1Name[1],imgs2Name[1],imgs1Name[2],imgs2Name[2])
                #
                # getAllElementsInLine  : Challenge C-10 区分 2,3
                # 
                if imgsFrm2.img1.getAllElementsInLine()==3 and imgsFrm2.img2.getAllElementsInLine()==3 and imgsFrm2.img3.getAllElementsInLine()==3:
                    frm1InLine1 = imgsFrm1.img1.getAllElementsInLine()
                    frm1InLine2 = 0 if frm1InLine1<=0 else imgsFrm1.img2.getAllElementsInLine()
                    frm1InLine3 = 0 if frm1InLine2<=0 else imgsFrm1.img3.getAllElementsInLine()
                    #print("[%s %s] %d %d %d" %(imgs1Name,imgs2Name,frm1InLine1,frm1InLine2,frm1InLine3))
                    if frm1InLine3>0 and frm1InLine3!=3:
                        score += 1
                        desc +=",且 %s中心在同一线上, %s中心在同一点上" %(imgs1Name,imgs2Name)

             # A==G and B==H and    
            scoreAddTo.addScore(score * scoreWeight ,imgs1Name,imgs2Name,scoreType,desc) # C-02
        elif    self.getImages2(imgs1Name[0:2]).isImgElementsEqualsOrSimilar() and self.getImages2(imgs1Name[1:3]).isImgElementsEqualsOrSimilar() \
            and self.getImages2(imgs2Name[0:2]).isImgElementsEqualsOrSimilar() and self.getImages2(imgs2Name[1:3]).isImgElementsEqualsOrSimilar():
            eq =  self.getImages2(imgs1Name[0:2]).isImgElementsEquals() and self.getImages2(imgs1Name[1:3]).isImgElementsEquals() \
                and self.getImages2(imgs2Name[0:2]).isImgElementsEquals() and self.getImages2(imgs2Name[1:3]).isImgElementsEquals()
            # ABC 相等, GHI 相等
            scoreAddTo.addScore((12 if eq else 10) * scoreWeight ,imgs1Name,imgs2Name,Agent.SCORETYPE3_EQ2,"每组图形全相等" if eq else "每组图形全相似")
            caseOuterSharpCmp = False
            caseCheckEqsIgnoreInced = False
            caseXorCmp = False
        #
        # 判断是否两组图片 元素 个数 全相同 , 或 个数的变换规律相同
        #
        elementCountIncAB = len(imgsFrm1.img2Elements) - len(imgsFrm1.img1Elements)  # A-<B 图片元素的增加量
        elementCountIncBC = len(imgsFrm1.img3Elements) - len(imgsFrm1.img2Elements)  # B->C 图片元素的增加量
        elementCountIncGH = len(imgsFrm2.img2Elements) - len(imgsFrm2.img1Elements)  # A-<B 图片元素的增加量
        elementCountIncHI = len(imgsFrm2.img3Elements) - len(imgsFrm2.img2Elements)  
        nElementIfSame = -1 # 如果 六个 图形 具有 相同 元素 个数,  nElementIfSame 将 >0
        nElementIfSame1 = -1 # 如果 第一组 三个 图形 具有 相同 元素 个数, nElementIfSame1  将 >0
        nElementIfSame2 = -1 # 如果 第二组 三个 图形 具有 相同 元素 个数, nElementIfSame1  将 >0
        #if "ABC"==imgs1Name and "GH3"==imgs2Name:
        #    print("[%s %s]elementCountIncAB=%d, elementCountIncBC=%d ; elementCountIncGH=%d, elementCountIncHI=%d" %(imgs1Name,imgs2Name,elementCountIncAB,elementCountIncBC,elementCountIncGH,elementCountIncHI))
        if elementCountIncAB==elementCountIncGH and elementCountIncBC==elementCountIncHI:  # 元素个数递增 相同
            if elementCountIncAB==0 and elementCountIncBC==0 :
                nElementIfSame1 = len(imgsFrm1.img1Elements)
                nElementIfSame2 = len(imgsFrm2.img1Elements)
            score = 4   
            scoreType = Agent.SCORETYPE3_ECCHANGE
            caseElementCountInc1Mathched = False
            if elementCountIncAB==0 and elementCountIncBC==0 and len(imgsFrm1.img1Elements)==len(imgsFrm2.img1Elements):  
                score += 1 
                scoreType |= Agent.SCORETYPE3_ECCHANGE_EQ
                desc = "两组图形元素个数相同"
                nElementIfSame =  len(imgsFrm1.img1Elements)   
            else:
                desc = "两组图形元素个数变化递增量相同"
            #
            #  Challenge Problem C-02 : 区分 答案 6/7
            #     
            if (imgsFrm1.allElementsInCenterY()>0  and imgsFrm2.allElementsInCenterY()>0):
                score += 1 
                desc  += ",且所有元素在同水平线上"
            elif (imgsFrm1.allElementsInCenterX()>0  and imgsFrm2.allElementsInCenterX()>0):
                score += 1 
                desc  += ",且所有元素在同垂直线上"
            if elementCountIncAB>0 and elementCountIncBC>0 :
                if  self.getImages2(imgs1Name[0:2]).isIncedSameElements(self.getImages2(imgs2Name[0:2])) and self.getImages2(imgs1Name[1:]).isIncedSameElements(self.getImages2(imgs2Name[1:])) :
                    score += 1 
                    # Challenge Problem E-11 : 添加了相同的元素, 区分 3,6
                    #  Challenge E-11 [ABC-GH3]两组图形元素个数变化递增量相同,且AB与GH增加了相同元素,BC与H3也增加了相同元素
                    desc  += ",且%s与%s增加了相同元素,%s与%s也增加了相同元素" %(imgs1Name[0:2],imgs2Name[0:2],imgs1Name[1:],imgs2Name[1:])
                elif  self.getImages2(imgs1Name[0:2]).isIncedSameElements(self.getImages2(imgs2Name[1:])) and self.getImages2(imgs1Name[1:]).isIncedSameElements(self.getImages2(imgs2Name[0:2])) :
                    score += 0.5
                    # Challenge Problem E-11 : 添加了相同的元素, 区分 3,6
                    #  Challenge E-11 [ABC-GH3]两组图形元素个数变化递增量相同,且AB与GH增加了相同元素,BC与H3也增加了相同元素
                    desc  += ",且%s与%s增加了相同元素,%s与%s也增加了相同元素" %(imgs1Name[0:2],imgs2Name[1:],imgs1Name[1:],imgs2Name[0:2])
                
                pass
                #abInced = self.getImages2(imgs1Name[0:2]).getIncedElements()
                #if abInced!=None 
            scoreAddTo.addScore(score * scoreWeight ,imgs1Name,imgs2Name,scoreType,desc)
            #
            # D-06 : 比较最后一个元素 的,  ABC - GH1
            #
            #print("%s/%s nElementIfSame1=%d,nElementIfSame2=%d" %(imgs1Name,imgs2Name,nElementIfSame1,nElementIfSame2))
            if nElementIfSame1>0 and nElementIfSame2>0:
                i1 = imgsFrm1.getNotEqImgElementIdx()
                #print("%s/%s i1=%d" %(imgs1Name,imgs2Name,i1))
                if i1>=0  :
                    i2 = imgsFrm2.getNotEqImgElementIdx()
                    #print("%s/%s i2=%d" %(imgs1Name,imgs2Name,i2))
                    if i2>0:
                        #print("%s/%s i1=%d,i2=%d" %(imgs1Name,imgs2Name,i1,i2))
                        if ImageElement.isElementsEqualsIgnoreOrder([imgsFrm1.img1Elements[i1],imgsFrm1.img2Elements[i1],imgsFrm1.img3Elements[i1]],[imgsFrm2.img1Elements[i2],imgsFrm2.img2Elements[i2],imgsFrm2.img3Elements[i2]]):
                            scoreAddTo.addScore(4 * scoreWeight ,imgs1Name,imgs2Name,Agent.SCORETYPE3_EQ3,"第一组的第%d元素与第二组的第%d元素相同组合" %(i1,i2))
                            pass
            #
            #  D-06  AB1 - BFG
            #             
            if elementCountIncAB==-1 and elementCountIncAB==elementCountIncBC :
                if(   ImageElement.isElementsContains(imgsFrm1.img2Elements,imgsFrm1.img1Elements) and ImageElement.isElementsContains(imgsFrm1.img3Elements,imgsFrm1.img2Elements)   \
                   and ImageElement.isElementsContains(imgsFrm2.img2Elements,imgsFrm2.img1Elements) and ImageElement.isElementsContains(imgsFrm2.img3Elements,imgsFrm2.img2Elements) \
                     ):
                    scoreAddTo.addScore(2 * scoreWeight ,imgs1Name,imgs2Name,Agent.SCORETYPE3_SUBELEMENTS,"子集关系")
            elif elementCountIncAB==1 and elementCountIncAB==elementCountIncBC :
                if(   ImageElement.isElementsContains(imgsFrm1.img1Elements,imgsFrm1.img2Elements) and ImageElement.isElementsContains(imgsFrm1.img2Elements,imgsFrm1.img3Elements)   \
                   and ImageElement.isElementsContains(imgsFrm2.img1Elements,imgsFrm2.img2Elements) and ImageElement.isElementsContains(imgsFrm2.img2Elements,imgsFrm2.img3Elements) \
                     ):
                    scoreAddTo.addScore(2 * scoreWeight ,imgs1Name,imgs2Name,Agent.SCORETYPE3_SUBELEMENTS,"子集关系")
                pass           
        elif len(imgsFrm1.img1Elements)>0 \
            and  len(imgsFrm1.img1Elements)>0 and len(imgsFrm2.img1Elements)>0 \
            and len(imgsFrm1.img2Elements)/len(imgsFrm1.img1Elements)==len(imgsFrm2.img2Elements)/len(imgsFrm2.img1Elements) \
            and len(imgsFrm1.img3Elements)/len(imgsFrm1.img1Elements)==len(imgsFrm2.img3Elements)/len(imgsFrm2.img1Elements) : 
            scoreAddTo.addScore(4 * scoreWeight ,imgs1Name,imgs2Name,Agent.SCORETYPE3_ECCHANGE|Agent.SCORETYPE3_ECCHANGE_MUTIPLE,"两组图形元素个数变化按同倍数递增")  #C-03 :
        #
        #  元素 比例的 ( C-02 的 答案案 1,4 就因为 比例 不同
        #            
        if nElementIfSame>0: #六个 图形 具有 相同 元素 个数
            # 判断 面积 是否 都 按 从 小->大 或 从 大->小 的 趋势
            for i in range(nElementIfSame):
                sizeA,sizeB,sizeC = imgsFrm1.img1Elements[i].getTotalPixel(),imgsFrm1.img2Elements[i].getTotalPixel(),imgsFrm1.img3Elements[i].getTotalPixel()
                sizeG,sizeH,sizeI = imgsFrm2.img1Elements[i].getTotalPixel(),imgsFrm2.img2Elements[i].getTotalPixel(),imgsFrm2.img3Elements[i].getTotalPixel()
                #print("[%s-%s] : %d %d %d , %d,%d,%d"%(imgs1Name,imgs2Name,sizeA,sizeB,sizeC,sizeG,sizeH,sizeI))
                sizeIncRatioAB, sizeIncRatioBC = (sizeB-sizeA)/sizeA , (sizeC-sizeB)/sizeB
                sizeIncRatioGH, sizeIncRatioHI = (sizeH-sizeG)/sizeG , (sizeI-sizeH)/sizeH
                if   sizeIncRatioAB>0.3 and sizeIncRatioGH>0.3 and sizeIncRatioBC>0.3 and sizeIncRatioHI>0.3 \
                   or sizeIncRatioAB>0.3 and sizeIncRatioGH>0.3 and sizeIncRatioBC>0.3 and sizeIncRatioHI>0.3 :
                    # 面积 同时 变化:
                    scoreAddTo.addScore(2 * scoreWeight ,imgs1Name,imgs2Name,Agent.SCORETYPE3_SIZECHANGE,"两组图形面积Delta变化趋势相同")
        else: # 两组元素
            caseOuterSharpCmp = False              
            
        
        if nElementIfSame>0 and not all6ImgEquals: # 六个 图形 具有 相同 元素 个数 (但不全等)
            #
            # 是否 匹配 旋转 属性:  
            # 例如  A (旋转90度)-> B (旋转90度)-> C
            #   且 G (旋转90度)-> H (旋转90度)-> I
            # 例子 Challenge D-04
            #     
            imgsAB = self.getImages2(imgs1Name[0:2])
            
            #print("%s : elementsCount = %d/%d " % (imgsAB.name,len(imgsAB.img1Elements),len(imgsAB.img2Elements)))
            
            scoreFac = 1/nElementIfSame
            for i in range(nElementIfSame):
                # 逆时针
                if    imgsAB.isImgElementTransMatched(i,IMGTRANSMODE_ROTATE090) \
                  and self.getImages2(imgs1Name[1:3]).isImgElementTransMatched(i,IMGTRANSMODE_ROTATE090) \
                  and self.getImages2(imgs2Name[0:2]).isImgElementTransMatched(i,IMGTRANSMODE_ROTATE090) \
                  and self.getImages2(imgs2Name[1:3]).isImgElementTransMatched(i,IMGTRANSMODE_ROTATE090) :
                    scoreAddTo.addScore( 10*scoreFac* scoreWeight,imgs1Name,imgs2Name,Agent.SCORETYPE3_ROTATE,"两组图形为90度旋转关系")
                    caseAddOrSubEq = False
                    caseXorEq = False
                    caseBitOPCmp = False
                    continue
                if   imgsAB.isImgElementTransMatched(i,IMGTRANSMODE_ROTATE270) \
                  and self.getImages2(imgs1Name[1:3]).isImgElementTransMatched(i,IMGTRANSMODE_ROTATE270) \
                  and self.getImages2(imgs2Name[0:2]).isImgElementTransMatched(i,IMGTRANSMODE_ROTATE270) \
                  and self.getImages2(imgs2Name[1:3]).isImgElementTransMatched(i,IMGTRANSMODE_ROTATE270) :
                    scoreAddTo.addScore( 10*scoreFac* scoreWeight,imgs1Name,imgs2Name,Agent.SCORETYPE3_ROTATE,"两组图形为-90度旋转关系")
                    caseAddOrSubEq = False
                    caseXorEq = False
                    caseBitOPCmp = False
                    continue
                imgsAC = self.getImages2(imgs1Name[0:1]+imgs1Name[2:3])
                if    (imgsAC.isImgElementTransMatched(i,IMGTRANSMODE_ROTATE090) and self.getImages2(imgs2Name[0:1]+imgs2Name[2:3]).isImgElementTransMatched(i,IMGTRANSMODE_ROTATE090) \
                        and imgsAB.isRoteteMatched(i,45) and self.getImages2(imgs2Name[0:2]).isRoteteMatched(i,45)  )\
                  or (imgsAC.isImgElementTransMatched(i,IMGTRANSMODE_ROTATE270) and self.getImages2(imgs2Name[0:1]+imgs2Name[2:3]).isImgElementTransMatched(i,IMGTRANSMODE_ROTATE270) \
                        and imgsAB.isRoteteMatched(i,-45) and self.getImages2(imgs2Name[0:2]).isRoteteMatched(i,-45)  ) :
                    caseAddOrSubEq = False
                    caseXorEq = False
                    caseBitOPCmp = False
                        #if imgsAB.isBlackPixelRatioEquals(i) and self.getImages2(imgs2Name[0:2]).isBlackPixelRatioEquals(i) : # todo 需要判断 满足 45 度的旋转 ,暂时 使用 isBlackPixelRatioEquals 代替
                        # Challenge D-02 :  [CDH-AE2]两组图形为45度旋转关系 : 判断错误, 
                        # Challenge D-04 :  [ABC-GH6] : 需要 5 分 区分 答案 2
                    scoreAddTo.addScore( 5*scoreFac* scoreWeight,imgs1Name,imgs2Name,Agent.SCORETYPE3_ROTATE,"两组图形为45度旋转关系")
                    continue
                #elif imgsAB.isImgElementTransMatched(i,IMGTRANSMODE_ROTATE270) 
        # END if nElementIfSame>0: #六个 图形 具有 相同 元素 个数
        
        #
        # Challenge D-06 : 第?个元素具有相同组合
        #
        if caseCheckLastElementEqs and elementCountIncAB==0 and elementCountIncBC==0 and elementCountIncGH==0 and elementCountIncHI==0:
            j1 = imgsFrm1.getOnlyNotEqElementIdx()
            j2 = -1 if j1<0 else imgsFrm2.getOnlyNotEqElementIdx()
            #if imgs2Name==""
            if j2>=0 and imgsFrm1.getImgElementEqualsIdxMap(j1,imgsFrm2,j2)!=None:
                scoreAddTo.addScore( 3 * scoreWeight,imgs1Name,imgs2Name,Agent.SCORETYPE3_EQ3,"%s的第%d个元素与%s的第%d个元素具有相同组合" %(imgs1Name,j1,imgs2Name,j2))
                pass
            # isEqualsAllElements
            pass

        """
        if (imgs1Name=="ABC" and imgs2Name.startswith("GH")) or (imgs1Name=="ADG" and imgs2Name.startswith("CF")) :
            #imgsCG = self.getImages2(imgs1Name[2]+imgs2Name[0])
            #imgsAI = self.getImages2(imgs1Name[0]+imgs2Name[2])
            #print("all6ImgEquals=%s, nElementIfSame=%d" %(all6ImgEquals,nElementIfSame))
            #print("检查 %s 与 %s ..." %(imgs1Name[0]+imgs1Name[2],imgs2Name[0]+imgs2Name[2]))
            self.calculateImages2MatchScore(imgs1Name[0]+imgs1Name[2],imgs2Name[0]+imgs2Name[2],scoreAddTo,0.2,True)
            # C-07 
        """    
        imgsAG = self.getImages2(imgs1Name[0]+imgs2Name[0])
        imgsBH = self.getImages2(imgs1Name[1]+imgs2Name[1])
        imgsCI = self.getImages2(imgs1Name[2]+imgs2Name[2])
        #
        # 检查 是否满足 翻转 ( C-07 )
        #
        if  caseWholeFliped:
            for flipMode in [IMGTRANSMODE_FLIPV,IMGTRANSMODE_FLIPH]:
                if imgsAG.isWholeImgElementsFliped(flipMode) and  imgsBH.isWholeImgElementsFliped(flipMode) and  imgsCI.isWholeImgElementsFliped(flipMode):
                    scoreAddTo.addScore( 10* scoreWeight,imgs1Name,imgs2Name,Agent.SCORETYPE3_FLIP,"两组图形%s翻转关系" %("水平" if flipMode==IMGTRANSMODE_FLIPH else "上下"))
                    break

        #
        # Challenge Problem D-10 : ADG / CF7
        #             
        if caseCheckIncedEq and len(imgsFrm1.img1Elements)==1 and elementCountIncAB==0 and elementCountIncBC==0 and elementCountIncGH==0 and elementCountIncHI==0 and len(imgsFrm2.img1Elements)>1:
            #print("%s-%s : 检查 第一组中三个图片分别等于第二组中三个图片的每个元素" %(imgsFrm1.name,imgsFrm2.name))
            if  imgsFrm1.img1Elements[0].isOuterSimilarAllElements(imgsFrm2.img1Elements) \
               and imgsFrm1.img2Elements[0].isOuterSimilarAllElements(imgsFrm2.img2Elements) \
               and imgsFrm1.img3Elements[0].isOuterSimilarAllElements(imgsFrm2.img3Elements) :
                  scoreAddTo.addScore( 5* scoreWeight,imgs1Name,imgs2Name,Agent.SCORETYPE3_EQ3,"第一组中三个图片分别等于第二组中三个图片的每个元素")
            pass            

        if caseOuterSharpCmp and nElementIfSame>0:
            # 判断 外形相似  : D-09
            idxOfImgs2 = []  # 第二组图片 对应在 第一组 图片中的序号 
            for i in range(3):
                j = imgsFrm1.getIndexOfOutterSharpEqImage(imgs2Name[i],idxOfImgs2)  # 第二组图片的 第 i 个图片, 在 第一组 中对应的序号 ( 不存在时 j==-1 )
                if j<0:
                    break
                idxOfImgs2.append(j)
            if len(idxOfImgs2)==3:
                caseVertices = False
                score = 3
                scorpType = Agent.SCORETYPE3_OUTEREQ1
                filledFlags1 =  imgsFrm1.getAllImagesFilledFlags()
                #print("%s - %s : filledFlags = %d,%d" %(imgsFrm1.name,imgsFrm2.name,filledFlags,imgsFrm2.getAllImagesFilledFlags() ))
                desc2 = ""
                if (filledFlags1==1 or filledFlags1==2) :
                    filledFlags2 =  imgsFrm2.getAllImagesFilledFlags()                                                             
                    #if filledFlags2==filledFlags1:
                    #    score += 1
                    #    desc2 = ",且两组元素全为填充图" if filledFlags1==1 else ",且两组元素全为非填充图"
                    #elif 
                    if filledFlags2==1 or filledFlags2==2:
                        score += 0.5
                        desc2 = ",且两组元素全为"+("填充图" if filledFlags2==1 else "非填充图")  # D-08 : 区分 1,4
                elif nElementIfSame==1 and imgsFrm1.isDifferentFillMode(0) and imgsFrm2.isDifferentFillMode(0):
                    score += 0.5
                    desc2 = ",且两组元素各使用不同的填充模式 " # Challenge D-07 : 区分 1,3
                if nElementIfSame==1:
                    # Challenge Problem C-12 : 答案 2,5,7 : 根据 高, 宽 匹配度判断:
                    def _whMatched(a1,a2):
                        a1 = sorted(a1)
                        a2 = sorted(a2)
                        for x1,x2 in zip(a1,a2):
                            if abs(x1-x2)>2: return False
                        return True
                    widths1 = [imgsFrm1.img1Elements[0].getWidth(),imgsFrm1.img2Elements[0].getWidth(),imgsFrm1.img3Elements[0].getWidth()]
                    widths2 = [imgsFrm2.img1Elements[0].getWidth(),imgsFrm2.img2Elements[0].getWidth(),imgsFrm2.img3Elements[0].getWidth()]
                    if _whMatched(widths1,widths2):
                        heights1 = [imgsFrm1.img1Elements[0].getWidth(),imgsFrm1.img2Elements[0].getWidth(),imgsFrm1.img3Elements[0].getWidth()]
                        heights2 = [imgsFrm2.img1Elements[0].getWidth(),imgsFrm2.img2Elements[0].getWidth(),imgsFrm2.img3Elements[0].getWidth()]
                        if _whMatched(heights1,heights2):
                            score += 0.5
                            desc2 = ",且两组元素大小匹配"
                    #print( "%s  widths=%s, heights=%s ; %s  widths=%s, heights=%s" %(imgs1Name,widths1,heights1,imgs2Name,widths2,heights2) )
                    pass
                
                scoreAddTo.addScore( score*scoreWeight,imgs1Name,imgs2Name,scorpType,"两组图形外形具有相同组合"+desc2) #D-09
            
        #
        #  考虑 是否新增了相同的元素  : D-2
        #   
        if  caseCheckEqsIgnoreInced:
            idxOfImgs2 = []  # 第二组图片 对应在 第一组 图片中的序号   idxOfImgs2[idxOfImg2] == idxOfImg1
            inced0 = 0
            for i in range(3):  # i : 
                j = imgsFrm1.getIndexOfEqualsImageIgnoreInced(imgs2Name[i],idxOfImgs2)  # 第二组图片的 第 i 个图片, 在 第一组 中对应的序号 ( 不存在时 j==-1 )
                if j<0 :
                    break
                if not ImageElement.isAllElementsEquals(imgsFrm1.imgsElementsLst[j]) or not ImageElement.isAllElementsEquals(imgsFrm2.imgsElementsLst[i]):
                    break
                inced = len(imgsFrm2.imgsElementsLst[i]) - len(imgsFrm1.imgsElementsLst[j]) 
                if inced==0: break
                if inced0==0:
                    inced0 = inced
                elif  inced!=inced0:
                    break
                idxOfImgs2.append(j)  # imgs1Name[j]==imgs2Name[i]
            if len(idxOfImgs2)==3:
                scoreAddTo.addScore( 10*scoreWeight,imgs1Name,imgs2Name,Agent.SCORETYPE3_INCEDSAME,"两组图形具有相同组合,并同时增加相同个数(%d)元素,对应序号=%s" %(inced0,idxOfImgs2))
                pass    

        #
        #  Challenge Problem D-12
        #         
        if  caseElementCountInc1Mathched:
            elementCounts1 = [len(imgsFrm1.img1Elements),len(imgsFrm1.img2Elements),len(imgsFrm1.img3Elements)]
            elementCounts1.sort()
            #print("elementCounts1 = ",elementCounts1)
            if elementCounts1[1]==elementCounts1[0]+1 and elementCounts1[2]==elementCounts1[0]+2:
                elementCounts2 = [len(imgsFrm2.img1Elements),len(imgsFrm2.img2Elements),len(imgsFrm2.img3Elements)]
                elementCounts2.sort()
                if elementCounts1[0]==elementCounts2[0] and elementCounts1[1]==elementCounts2[1] and elementCounts1[2]==elementCounts2[2]:
                    scoreAddTo.addScore( 1.5* scoreWeight,imgs1Name,imgs2Name,Agent.SCORETYPE3_ECCHANGE2,"两组元素个数(不考虑次序的情况下)匹配")
                    pass
            
        #
        #  Challenge Problem D-10 : A-C 相似, 
        #         
        #if caseTransMatched \
        #        and len(imgsFrm1.img1Elements)==1 and elementCountIncAB==elementCountIncGH and elementCountIncBC==0 and elementCountIncHI==0 \
        #        and imgsBH.isImgElementsEqualsOrSimilar() :
            #for transMode in [IMGTRANSMODE_ROTATE090,IMGTRANSMODE_ROTATE270,IMGTRANSMODE_ROTATE180]:
            #imgsAG.getAllImgElementTrans()
        #    pass         
        caseLRMerge  = caseAddOrSubEq  # 满足 左右合并,  Challenge Problem E-04

        #
        # 是否 匹配 相加 属性 或 相减
        #  即:  图片A + 图片B == 图片C  ( 按 bit 加 )
        #    且 图片I + 图片B == 图片C
        #  或:  图片A - 图片B == 图片C
        #    且 图片I - 图片B == 图片C
        #       
        # 例子: 相加的例子 E-01 , E-02, E-03 , C-06
        #      相加的例子 
        #     XOR 的例子: D-11
        #      
        # 
        if caseAddOrSubEq: # 处理 图片A + 图片B == 图片C 的情况
            # A+B==C
            # B+C==A
            # A+C==B
            for abc in ((0,1,2),(1,2,0),(0,2,1)):
                diffABC,_,totalPixcelABC = Image1.countImagesDiff([self.getImage1(imgs1Name[abc[0]]),self.getImage1(imgs1Name[abc[1]])],[self.getImage1(imgs1Name[abc[2]])])  
                if diffABC/totalPixcelABC<=0.02:
                    diffGHI,_,totalPixcelGHI = Image1.countImagesDiff([self.getImage1(imgs2Name[abc[0]]),self.getImage1(imgs2Name[abc[1]])],[self.getImage1(imgs2Name[abc[2]])])  
                    if diffGHI/totalPixcelGHI<=0.02:
                        scoreAddTo.addScore( 6 * scoreWeight,imgs1Name,imgs2Name,Agent.SCORETYPE3_MERGE,"第%d和%d图片像素合并==第%d个图片"%(abc[0]+1,abc[1]+1,abc[2]+1))
                        #caseXorEq = False    
                        #caseAndCmp = False
                        caseXorCmp = False
                        caseBitOPCmp = False
                        caseLRMerge = False
                        caseVertices = False
                    break

        #
        #  Challenge Problem E-04 :  [ABC-GH5]第2和3图片左右合并==第1个图片
        #             
        if caseLRMerge:
            for abc in ((0,1,2),(1,2,0),(0,2,1)):
                if Image1.isMatchedLRMerged(self.getImage1(imgs1Name[abc[0]]),self.getImage1(imgs1Name[abc[1]]),self.getImage1(imgs1Name[abc[2]])) \
                      and Image1.isMatchedLRMerged(self.getImage1(imgs2Name[abc[0]]),self.getImage1(imgs2Name[abc[1]]),self.getImage1(imgs2Name[abc[2]])) :
                    scoreAddTo.addScore( 6 * scoreWeight,imgs1Name,imgs2Name,Agent.SCORETYPE3_MERGE|Agent.SCORETYPE3_MERGE_LR,"第%d和%d图片左右合并==第%d个图片"%(abc[0]+1,abc[1]+1,abc[2]+1))
                    caseXorCmp = False
                    caseBitOPCmp = False
                    caseVertices = False
            #isMatchedLRMerged

        #
        # 多边形 Challenge E-05;
        #
        if caseVertices and len(imgsFrm1.img1Elements)==1 and  elementCountIncAB==0 and  elementCountIncBC==0 and len(imgsFrm2.img1Elements)==1 and elementCountIncGH==0 and elementCountIncHI==0:
            verticesA = len(imgsFrm1.img1Elements[0].getPolygonPoints())
            verticesB = len(imgsFrm1.img2Elements[0].getPolygonPoints())
            """
            print("检测 多边形 ... %s-%s : %d %d %d; %d %d %d" %(imgs1Name,imgs2Name,verticesA,verticesB,len(imgsFrm1.img3Elements[0].getPolygonPoints()),\
                                len(imgsFrm2.img1Elements[0].getPolygonPoints()),\
                                len(imgsFrm2.img2Elements[0].getPolygonPoints()),\
                                len(imgsFrm2.img3Elements[0].getPolygonPoints()),\
                                                                                                    )) 
            """                                                                                        
            if verticesB>verticesA \
                  and verticesB-verticesA==len(imgsFrm2.img2Elements[0].getPolygonPoints())-len(imgsFrm2.img1Elements[0].getPolygonPoints())\
                  and len(imgsFrm1.img3Elements[0].getPolygonPoints()) - verticesB==len(imgsFrm2.img3Elements[0].getPolygonPoints())-len(imgsFrm2.img2Elements[0].getPolygonPoints()):
                scoreAddTo.addScore(1*scoreWeight,imgs1Name,imgs2Name,Agent.SCORETYPE3_VERTICES,"图形顶点数变化趋势一致")

        #
        #  C -09:
        #     
        #print("caseMovedMerge %s:%s %s:%s"  %(imgs1Name,self.getImages2(imgs1Name[0]+imgs1Name[2]).isImgSame2SwappedElements(),imgs2Name,self.getImages2(imgs2Name[0]+imgs1Name[2]).isImgSame2SwappedElements()))
        if  caseMovedMerge and len(imgsFrm1.img1Elements)==2 and len(imgsFrm1.img2Elements)==1 and len(imgsFrm1.img3Elements)==2 \
                           and len(imgsFrm2.img1Elements)==2 and len(imgsFrm2.img2Elements)==1 and len(imgsFrm2.img3Elements)==2 \
                           and self.getImages2(imgs1Name[0]+imgs1Name[2]).isImgSame2SwappedElements()  \
                           and self.getImages2(imgs2Name[0]+imgs2Name[2]).isImgSame2SwappedElements()  \
                            :
            img = ImageElement.getElementsCenterAlignMerged(imgsFrm1.img1Elements)
            diff,_,_ = countImageDiffRatio(img,imgsFrm1.img2Elements[0].image)
            if diff<0.05 and countImageDiffRatio(ImageElement.getElementsCenterAlignMerged(imgsFrm2.img1Elements),imgsFrm2.img2Elements[0].image):
                scoreAddTo.addScore(5*scoreWeight,imgs1Name,imgs2Name,Agent.SCORETYPE3_LRMOVED,"元素左右移动变换")
                #print("caseMovedMerge %s %s............." %(imgs1Name,imgs2Name))
            pass

        #
        # 三个图形 异或 后, 相同  : D-09 ???
        #                       : E-08 
        #  ??? Challenge E-12 : 有可能误判
        #     
        if caseXorCmp:
            xorImg1 = imgsFrm1.getXORImageElement()   
            xorImg2 = imgsFrm2.getXORImageElement()   
            ratio,_,_ = countImageDiffRatio(xorImg1.image,xorImg2.image)
            #print("------%s - %s : xorImgDiffRatio = %s" %(xorImg1.name,xorImg2.name,ratio))
            if ratio<0.03:
                scoreAddTo.addScore( 1*scoreWeight ,imgs1Name,imgs2Name,Agent.SCORETYPE3_XOR,"两组图形每组XOR后的图形相似") 
                pass
                #caseBitOPCmp = False  
                #caseAddOrSubEq = False
            elif ratio<0.05: # E-08 : ABC-GH1 : 0.04
                scoreAddTo.addScore( 0.3*scoreWeight ,imgs1Name,imgs2Name,Agent.SCORETYPE3_XOR,"两组图形每组XOR后的图形相似") 
                #caseBitOPCmp = False 
                #caseAddOrSubEq = False
            
            #if ratio<0.07:
            #    scoreAddTo.addScore( 3 if ratio<0.03 else ( 2 if ratio<0.05 else 1),imgs1Name,imgs2Name,"两组图形每组XOR后的图形相似") 

        if caseBitOPCmp:
            # E-10  , 
            # E-11 : [DEF-GH5]两组图形 D bitand E == F 且 G bitand H==5
            bitopMatched = imgsFrm1.getBitOPMatched() 
            if bitopMatched>0 and imgsFrm2.getBitOPMatched()==bitopMatched:
                if bitopMatched==1:
                    scoreAddTo.addScore( 5*scoreWeight ,imgs1Name,imgs2Name,Agent.SCORETYPE3_BITOP|1,"两组图形 %s bitand %s == %s 且 %s bitand %s==%s" %(imgsFrm1.imgId1,imgsFrm1.imgId2,imgsFrm1.imgId3,imgsFrm2.imgId1,imgsFrm2.imgId2,imgsFrm2.imgId3)) 
                elif bitopMatched==2:
                    scoreAddTo.addScore( 5*scoreWeight ,imgs1Name,imgs2Name,Agent.SCORETYPE3_BITOP|2,"两组图形 %s bitxor %s == %s 且 %s bitxor %s==%s" %(imgsFrm1.imgId1,imgsFrm1.imgId2,imgsFrm1.imgId3,imgsFrm2.imgId1,imgsFrm2.imgId2,imgsFrm2.imgId3)) 
                elif bitopMatched==3:
                    scoreAddTo.addScore( 5*scoreWeight ,imgs1Name,imgs2Name,Agent.SCORETYPE3_BITOP|3,"两组图形 %s bitor %s == %s 且 %s bitor %s==%s" %(imgsFrm1.imgId1,imgsFrm1.imgId2,imgsFrm1.imgId3,imgsFrm2.imgId1,imgsFrm2.imgId2,imgsFrm2.imgId3)) 
        
        
        #
        #  Challenge Problem C-10 : 区分 答案 2,3 : 
        #   根据 包含关系 :
        #
        """
        if maxImgPixelRatio<0.05 \
               and len(imgsFrm1.img1.getImageElements())==len(imgsFrm2.img1.getImageElements()) \
               and len(imgsFrm1.img2.getImageElements())==len(imgsFrm2.img2.getImageElements()) \
               and len(imgsFrm1.img3.getImageElements())==len(imgsFrm2.img3.getImageElements()):
            def distanceMatched(elements1,elements2): # 判断 到 最小元素 的距离 按 相同 规律 
                n = len(elements1) 
                if n!=len(elements2): return False
                if n<=1 : return True
                x10, y10 =  elements1[n-1].getCenter()
                x20, y20 =  elements2[n-1].getCenter()
                for e1,e2 in zip(elements1,elements2):
                    x1, y1 =  e1.getCenter()
                    x2, y2 =  e2.getCenter()
                    dx1 = x1 - x10
                    dy1 = y1 - y10
                    dx2 = x2 - x20
                    dy2 = y2 - y20
                    if abs(dx1)<5 : dx1 = 0
                    if abs(dx2)<5 : dx2 = 0
                    if abs(dy1)<5 : dy1 = 0
                    if abs(dy2)<5 : dy2 = 0
                    print("%s : dx1=%f dy1=%f ; %s : dx2=%f dy2=%f " %(e1.name,dx1,dy1,e2.name,dx2,dy2))
                return False    
            if      distanceMatched(imgsFrm1.img1.getImageElements(),imgsFrm2.img1.getImageElements())  \
                and distanceMatched(imgsFrm1.img2.getImageElements(),imgsFrm2.img2.getImageElements())  \
                and distanceMatched(imgsFrm1.img3.getImageElements(),imgsFrm2.img3.getImageElements()):
                scoreAddTo.addScore(1*scoreWeight,imgs1Name,imgs2Name,"两图片元素位置(距离)有相同规律") 
        """

              
        #
        # XOR 的例子: D-11
        # 
        #if caseXorEq:
        #    pass # todo                
    # END method calculateImages3MatchScore 

    def calculateImages3MatchScore_2(self,imgs1Name:str,imgs2Name:str,scoreAddTo:AnswerScore,scoreWeight=1):

        imgsFrm1 = self.getImages3(imgs1Name)
        imgsFrm2 = self.getImages3(imgs2Name)

        caseXorCmp = True
        #caseBitOPCmp = True
        caseAddOrSubEq = True


        
        # E-04 : 答案 2 与 8 都一样
        if caseAddOrSubEq: # 处理 图片A.像素个数 + 图片B.像素个数 == 图片C 的情况  E-04   
            c1 = imgsFrm1.compareImgPixelCount() 
            if c1>0 and c1==imgsFrm2.compareImgPixelCount() :        
                caseAddOrSubEq = False    
                #caseXorEq = False 
                #caseAndCmp = False
                #  - E-04 : 答案 2 与 8 都一样 , 进一步考虑图片形状 
                if  ( imgsFrm1.compareImgPixelHeight()==0 and imgsFrm2.compareImgPixelHeight()==0 and imgsFrm1.compareImgPixelWidth()==c1 and imgsFrm2.compareImgPixelWidth()==c1  ) \
                 or ( imgsFrm1.compareImgPixelWidth()==0 and imgsFrm2.compareImgPixelWidth()==0  and imgsFrm1.compareImgPixelHeight()==c1 and imgsFrm2.compareImgPixelHeight()==c1):
                    scoreAddTo.addScore( 6 * scoreWeight,imgs1Name,imgs2Name,0,"前两图片像素个数相加或减==第三个图片,且宽高匹配",1)
                else:
                    scoreAddTo.addScore( 1 * scoreWeight,imgs1Name,imgs2Name,0,"前两图片像素个数相加或减==第三个图片",1)


        #
        #  比较 像素 变化规律: AB , BC , AC
        #    r1 in [A/B , B/C , A/C] 
        #    r2 in [G/H , H/I,  G/I]     
        #          
        maxImgPixelRatio = 0  
        for r1,r2 in zip(imgsFrm1.getImagePixelRatio(),imgsFrm2.getImagePixelRatio()):
            diff =  abs(r1 - r2)
            if diff>maxImgPixelRatio: maxImgPixelRatio = diff
        if  maxImgPixelRatio< 0.05:
            scoreAddTo.addScore(3*scoreWeight,imgs1Name,imgs2Name,0,"两图片像素变化率相差<0.05",1) 
        elif maxImgPixelRatio < 0.1:
            scoreAddTo.addScore(2*scoreWeight,imgs1Name,imgs2Name,0,"两图片像素变化率相差<0.1",1) 
        elif maxImgPixelRatio < 0.15:
            scoreAddTo.addScore(1*scoreWeight,imgs1Name,imgs2Name,0,"两图片像素变化率相差<0.15",1) 

        
        #
        #  Challenge D-09 目前暂时使用 像素 变换率判断, 
        #
        for r1,r2,id1,id2 in zip(imgsFrm1.getImagePixelRatio(),imgsFrm2.getImagePixelRatio(),[imgs1Name[0:2],imgs1Name[1:3],imgs1Name[0:1]+imgs1Name[2:3]],[imgs2Name[0:2],imgs2Name[1:3],imgs2Name[0:1]+imgs2Name[2:3]]):
            diff =  abs(r1 - r2)
            #if diff<0.15:
            #    print("diff = %f" %diff)
            if  diff< 0.05:
                scoreAddTo.addScore(1.5*scoreWeight,imgs1Name,imgs2Name,0,"两图片(%s与%s)像素个数变化率相差<0.05"%(id1,id2),1) 
            elif diff < 0.1:
                scoreAddTo.addScore(1*scoreWeight,imgs1Name,imgs2Name,0,"两图片(%s与%s)像素个数变化率相差<0.1"%(id1,id2),1) 
            elif diff < 0.15:
                scoreAddTo.addScore(0.5*scoreWeight,imgs1Name,imgs2Name,0,"两图片(%s与%s)像素个数变化率相差<0.15"%(id1,id2),1) 
        

    def _printAnswerScoreDetails(answerScore:AnswerScore)->None:
        if len(answerScore.scoreDetails)==0:
            print("答案 [%d] 总得分 = %.2f " % (answerScore.answer,answerScore.score))
        else:
            print("答案 [%d] 总得分 = %.2f , 其中 " % (answerScore.answer,answerScore.score))
            for d in answerScore.scoreDetails:
                print("  得分 %.2f 来自于: [%s-%s]%06X:%s %s %s" % (d.score,d.imgs1Name,d.imgs2Name,d.type,d.desc,"" if d.step==0 else "(附加分)","" if (d.flags&1)==0 else "(被排除)"))

    def _printAnswersScoreDetails(answersScore:list)->None:                
        for answerScore in answersScore:
            Agent._printAnswerScoreDetails(answerScore)
        
################################################################
# solve_2x2
#############################################################
    def solve_2x2(self):
        allAnswersScore = []
        #for answer in self.potential_answers:
        for answer in self.images:
            if not answer.isdigit(): continue
            answerScore = AnswerScore(int(answer))   
            self.calculateImages2MatchScore("AB","C"+answer,answerScore)  # 行比较 : 第一行 与 第 三 行
            self.calculateImages2MatchScore("AC","B"+answer,answerScore)
            #  self.calculateImages2MatchScore("BC","A"+answer,answerScore,0.5) # 对角线 ????
            #if Agent._DEBUG:
            #    Agent._printAnswerScoreDetails(answerScore)
            allAnswersScore.append(answerScore)
        answersScore = AnswerScore.getMaxScoreAnswers(allAnswersScore)
      
        if len(answersScore)>1:
            if Agent._WARN:
                print("答案 %s 的得分相同(%f),继续根据像素变换 判断 ..." %(",".join(map(lambda s:str(s.answer),answersScore)),answersScore[0].score))
            for answerScore in  answersScore:
                answer = str(answerScore.answer)
                self.calculateImages2MatchScore_2("AB","C"+answer,answerScore)  # 行比较 : 第一行 与 第 三 行
                self.calculateImages2MatchScore_2("AC","B"+answer,answerScore)
            answersScore = AnswerScore.getMaxScoreAnswers(answersScore)

        if Agent._DEBUG:
            Agent._printAnswersScoreDetails(allAnswersScore)
      
        if len(answersScore)>1 and Agent._WARN:
            print("[%s] 答案 = %s " %(self.problem.name,",".join(map(lambda s:str(s.answer),answersScore))))
        return answersScore[0].answer


###############################################################
#
#  solve_3x3 
# 
##################################################################        

    def solve_3x3(self):
        # 去除一些 单独 的规则:
        # (1) count(SCORETYPE3_XOR) >=5 : 排除 Challenge Problem E-12 的 XOR
        # (2) count(SCORETYPE3_EQ1) >=3  : 排除 Challenge Problem E-12 的  (??? 误排除了 Challenge C-10 [ADG-CF3])
        #                                  排除 Challenge Problem E-05 的
        #
        rulesChecks = [\
                (0xff000|Agent.SCORETYPE3_EQ1_1,Agent.SCORETYPE3_EQ1,3 ),\
                (0xff000,Agent.SCORETYPE3_XOR,5 ),\
                (0xff000,Agent.SCORETYPE2_VERTICES,4 )
                ]  
        allAnswersScore = []
        for answer in self.images:
            if not answer.isdigit():continue
            answerScore = AnswerScore(int(answer)) 
    #         A     B    C    
    #         D     E    F
    #         G     H    ?
    #  Frames:
    #        ABC    DEF  (行)
    #        ADG    BEH  (列)
    #        BFG    CDH  (对角线)  
    #       
    #  比较:
    #        ABC-GH?  DEF-GH?   ( 第一行 与 第 三 行,  第二行 与 第 三 行)
    #        ADG-CF?  BEH-CF?   (  第一列 与 第 三 列 ,第二列 与 第 三 列 )
    #        BFG-AE?  CDH-AE?   ( 对角线 )
    # 比较 对每个答案, 比较以上 行 或 列 或 对角线 直接的 相似度,变换规律等 属性 , 得到 对应答案的可行度分数
    #
            self.calculateImages3MatchScore("ABC","GH"+answer,answerScore)  # 行比较 : 第一行 与 第 三 行
            self.calculateImages3MatchScore("DEF","GH"+answer,answerScore)  # 行比较 : 第二行 与 第 三 行
            self.calculateImages3MatchScore("ADG","CF"+answer,answerScore)  # 列比较 : 第一列 与 第 三 列
            self.calculateImages3MatchScore("BEH","CF"+answer,answerScore)  # 列比较 : 第二列 与 第 三 列
            self.calculateImages3MatchScore("BFG","AE"+answer,answerScore,0.7)  # 对角线 
            self.calculateImages3MatchScore("CDH","AE"+answer,answerScore,0.7)  # 对角线
            # score += self.calculateImages2MatchScore("BC","A"+answer,0.5)
            for  rulesCheck in rulesChecks:
                n =  0
                def matchRuleByMasktype(answerScoreDetail):
                    return (answerScoreDetail.type&rulesCheck[0])==rulesCheck[1]
                for d in answerScore.scoreDetails:
                    if matchRuleByMasktype(d): n += 1
                if n==0 or n>=rulesCheck[2]:
                    continue
                for d in answerScore.scoreDetails:
                    answerScore.execludeScores(matchRuleByMasktype)

            allAnswersScore.append(answerScore)

        

        answersScore = AnswerScore.getMaxScoreAnswers(allAnswersScore)

        if len(answersScore)>1:
            if Agent._WARN:
                print("答案 %s 的得分相同(%f),继续根据像素变换 判断 ..." %(",".join(map(lambda s:str(s.answer),answersScore)),answersScore[0].score))
            for answerScore in  answersScore:
                answer = str(answerScore.answer)
                self.calculateImages3MatchScore_2("ABC","GH"+answer,answerScore)  # 行比较 : 第一行 与 第 三 行
                self.calculateImages3MatchScore_2("DEF","GH"+answer,answerScore)  # 行比较 : 第二行 与 第 三 行
                self.calculateImages3MatchScore_2("ADG","CF"+answer,answerScore)  # 列比较 : 第一列 与 第 三 列
                self.calculateImages3MatchScore_2("BEH","CF"+answer,answerScore)  # 列比较 : 第二列 与 第 三 列
                self.calculateImages3MatchScore_2("BFG","AE"+answer,answerScore,0.7)  # 对角线 
                self.calculateImages3MatchScore_2("CDH","AE"+answer,answerScore,0.7)  # 对角线
            answersScore = AnswerScore.getMaxScoreAnswers(answersScore)

        if Agent._DEBUG:
            Agent._printAnswersScoreDetails(allAnswersScore)
            
        if len(answersScore)>1 and Agent._WARN:
            print("[%s] 答案 = %s " %(self.problem.name,",".join(map(lambda s:str(s.answer),answersScore))))
            
        return answersScore[0].answer

    def prepareProblem(self, problem):
        self.images = load_problem_images(problem)
        self.imagesFrame = {}
        self.problem = problem
        ImageElement.cached = {}
        Image1.cached = {}
        Images2.cached = {}
        Images3.cached = {}

    def Solve(self, problem):
        """
        Primary method for solving incoming Raven's Progressive Matrices.

        Args:
            problem: The RavensProblem instance.

        Returns:
            int: The answer (1-6 for 2x2 OR 1-8 for 3x3) : Remember that the Autograder will have up to 2 additional images for answers.
            Return a negative number to skip a problem.
            Remember to return the answer [Key], not the name, as the ANSWERS ARE SHUFFLED in Gradescope.
        """

        """
        DO NOT use absolute file pathing to open files.
        
        Example: Read the 'A' figure from the problem using Pillow
            image_a = Image.open(problem.figures["A"].visualFilename)
            
        Example: Read the '1' figure from the problem using OpenCv
            image_1 = cv2.imread(problem.figures["1"].visualFilename)
            
        Don't forget to uncomment the imports as needed!
        """
        self.prepareProblem(problem)
        #print("\n")
        #print("--->This Problem: ", problem.name)

        if problem.problemType == "2x2":
            # if problem.name[-4:] == 'B-04':
                # print("Debug B12:", problem.name[-4:])
            return self.solve_2x2()
        elif problem.problemType == "3x3":
            return self.solve_3x3()

        return -1

#
#####################################
#

# END class Agent    




