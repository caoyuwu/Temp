import cv2
import numpy as np
import os
from itertools import product

#from CV2Utils import CV2Utils 

APPPATH = os.path.dirname(__file__)

THRESHOLD = 1

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
        ravens_image = cv2.cvtColor(cv2.imread(img.visualFilename), cv2.COLOR_BGR2GRAY)
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
        return (self.ex+self.x0)/2,(self.ey+self.y0)/2
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
    
    def isBlackPixelRatioEquals(self,otherImgElement,ratioThreadhold=0.05):
        return abs(self.getBlackPixelRatio()-otherImgElement.getBlackPixelRatio()) <= ratioThreadhold;
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
            # Challenge Problems B-01 三角形 垂直 方向 可能 为 7
            #if w1<=5 and w2<=5:
            if w1<=7 and w2<=7:
                continue
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

    def getTransImage(self,transMode:str):
        if transMode==IMGTRANSMODE_FLIPV or transMode==IMGTRANSMODE_FLIPH or transMode==IMGTRANSMODE_FLIPVH :
            return self.getFlipedImage(transMode)
        if transMode==IMGTRANSMODE_ROTATE1 or transMode==IMGTRANSMODE_ROTATE2 or transMode==IMGTRANSMODE_ROTATE3:    
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
        #print("rotaMode=",rotaMode )
        if rotaMode==IMGTRANSMODE_ROTATE1:
            rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE # ROTATE_90_CLOCKWISE #rotaMode*90
        elif rotaMode==IMGTRANSMODE_ROTATE2:
            rotateCode = cv2.ROTATE_180
        elif rotaMode==IMGTRANSMODE_ROTATE3:
            rotateCode = cv2.ROTATE_90_CLOCKWISE
        else:
            raise BaseException("Invalid rotaMode=%d" % rotaMode)
        if rotaMode in self.transformImgs:
            #print("使用缓存图片...")
            return self.transformImgs.get(rotaMode)
        #print("rotateCode = %d"% rotateCode)
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
    
    def getFilledImage(self):
        imgKey = IMGTRANSMODE_FILLED
        if imgKey in self.transformImgs:
            #print("使用缓存图片...")
            return self.transformImgs.get(imgKey)
        
        imgElement = ImageElement(self.image.shape,self.name+"-"+imgKey)
        #imgElement.image = self.image.copy()
        img = imgElement.image 
        for y in range(self.y0,self.ey):
            for x in range(self.getStartPointX(y),self.getEndPointX(y)+1):
                img[y,x] = 0
        for x in range(self.x0,self.ex):
            startY = self.getStartPointY(x)
            if startY<0:
                continue
            endY = self.getEndPointY(x)
            for y in range(self.y0,startY):
                img[y,x] = 255
            for y in range(endY+1,self.ey):
                img[y,x] = 255
        # todo 还可能有些 凹进去的点 没被设置
          # cv2.floodFill         
        imgElement.update()    
        self.transformImgs[imgKey] = imgElement
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
    #  1: 含 全填充 ; 2: 含未全填充
    #     
    def getImagesFilledFlags(imgElements:list)->int:
        flags = 0
        for e in imgElements:
            if e.isFilledImage():
                flags |= 1
            else:
                flags |= 2
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

IMGTRANSMODE_ROTATE1 = "ROTAGE90" #  逆时针 旋转 90度
IMGTRANSMODE_ROTATE2 = "ROTAGE180" #   旋转 180度
IMGTRANSMODE_ROTATE3 = "ROTAGE270" #  逆时针 旋转 270度(-90度)  
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

        if transMode==IMGTRANSMODE_ROTATE1 or transMode==IMGTRANSMODE_ROTATE2 or transMode==IMGTRANSMODE_ROTATE3:
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
        for elementIdx in range(startElementIdx,endElementIdx):
            transInfo = self.getImgElementTrans(elementIdx,IMGTRANSMODE_EQ)
            if not transInfo.matched:
                return False
        return True    
    
    def isImgElementsEquals(self,startElementIdx=0,endElementIdx=0)->bool:
        if endElementIdx==0:
            endElementIdx = len(self.transElements)
        for elementIdx in range(startElementIdx,endElementIdx):
            transInfo = self.getImgElementTrans(elementIdx,IMGTRANSMODE_EQ)
            if not transInfo.matched or abs(transInfo.scale-1)>0.05 :
                return False
        return True    
    
    #
    # 判断 从 startElementIdx - endElementIdx 的元素 外形匹配 , D-09 的 B3
    #
    def isImgElementsOutterSharpMatched(self,startElementIdx=0,endElementIdx=0)->bool:
        if endElementIdx==0:
            endElementIdx = len(self.transElements)
        for elementIdx in range(startElementIdx,endElementIdx):
            transInfo = self.getImgElementTrans(elementIdx,IMGTRANSMODE_EQ)
            if not transInfo.matched and  not transInfo.matched3 :
                return False
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
        
        img1X1,img1Y1 = self.img1Elements[0].getCenter()
        img2X2,img2Y2 = self.img2Elements[1].getCenter()

        if abs(img1X1-img2X2)>2 or abs(img1Y1-img2Y2)>2:
            return False

        img1X2,img1Y2 = self.img1Elements[1].getCenter()
        img2X1,img2Y1 = self.img2Elements[0].getCenter()

        print("图形1 = (%f,%f)  图形2 = (%f,%f) " %(img1X1,img1Y1,img2X2,img2Y2))
        print("图形1 = (%f,%f)  图形2 = (%f,%f) " %(img1X2,img1Y2,img2X1,img2Y1))
        return abs(img1X2-img2X1)<=2 or abs(img1Y2-img2Y1)<=2

    def getImagePixelRatio(self)->list:
        img1BlackPoints = self.img1.getSumImgElementsBlackPoints()
        img2BlackPoints = self.img2.getSumImgElementsBlackPoints()
        return  -1000 if img2BlackPoints==0 else img1BlackPoints/img2BlackPoints

    def getXORImageElement(self)->ImageElement:
        return ImageElement.getXORImageElement([self.img1.asImgElement(),self.img2.asImgElement()])
    
    def getANDImage(self)->ImageElement:
        try:
            return self._andImage
        except AttributeError as e:pass 
        self._andImage = cv2.bitwise_or(self.img1.image,self.img2.image,mask=None)
        #self._andImage = ImageElement.getANDImageElement([self.img1.asImgElement(),self.img2.asImgElement()]).image
        return self._andImage

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
            if indexOf(excludeIdxs,i)<0 and len(self.imgsElementsLst[i])==nElements and self.agent.getImages2(self.name[i]+otherImgId).isImgElementsEqualsOrSimilar():
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
    #
    def getANDMatched(self)->int:
        try:
            return self._andMatched
        except AttributeError as e:pass  
        imgsAB = self.agent.getImages2(self.imgId1+self.imgId2) 
        diffRatio,diffCount,_ = countImageDiffRatio(imgsAB.getANDImage(),self.agent.getImage(self.imgId3))
        #print("[%s - %s] diffRatio=%f,diffCount=%d" %(imgsAB.name,self.imgId2,diffRatio,diffCount))
        return diffRatio<0.01                               
 
# END class Images3   
     
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
    def addScore(self,score:float,imgs1Name:str,imgs2Name:str,desc:str):
        self.score += score
        self.scoreDetails.append((score,imgs1Name,imgs2Name,desc))    
    def getMaxScoreAnswers(answers):
        answers.sort(key=lambda e:e.score,reverse=True)
        maxScoreAnswers = []
        for ansert in answers:
            if len(maxScoreAnswers)==0 or ansert.score==maxScoreAnswers[0].score:
                maxScoreAnswers.append(ansert)
        return  maxScoreAnswers       
# END class AnswerScore
#
class Agent:
    _DEBUG = False
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
    def calculateImages2MatchScore(self,imgs1Name,imgs2Name,scoreAddTo:AnswerScore,scoreWeight=1,for3X3=False): 
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
                for transMode in [IMGTRANSMODE_ROTATE1,IMGTRANSMODE_ROTATE3,IMGTRANSMODE_ROTATE2]:
                    t = imgsFrm1.getImgElementTrans(elementIdx,transMode)
                    if t.matched:
                        forAllTrans.append(t)
                        break
            for transInfo in forAllTrans:  # r similar,scale
                #if Agent._DEBUG and imgs2Name=="C6" and imgs1Name=="AB":
                #    print("%s-%s: imgs1.transMode=%s,%s:%f,%s:%f " %(imgs1Name,imgs2Name,transInfo.transMode,transInfo.matched,transInfo.similar,transInfo.matched2,transInfo.similar2))
                if not transInfo.matched and not transInfo.matched2:
                    continue
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
                scoreAddTo.addScore(score*scoreFactor*scoreWeight,imgs1Name,imgs2Name,"元素%d匹配相同变换%s%s"%(elementIdx,transInfo.transMode,desc2))
                caseCheckFilled = False

        #
        # Challenge Problem B-10 : 如果 整图 旋转  AB/C4 
        #
        if elementCountDiff1==elementCountDiff2  and elementCountDiff1==0 and imgsFrm1.getImgElementCount()==imgsFrm2.getImgElementCount() and minElementCount>1:
            checkAllRota = None
            for transMode in [IMGTRANSMODE_ROTATE1,IMGTRANSMODE_ROTATE2,IMGTRANSMODE_ROTATE3]:
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
                    scoreAddTo.addScore(3*scoreWeight,imgs1Name,imgs2Name,"整图旋转")

        if for3X3:
            return
                            
        #
        # 考虑 元素 增加 / 减少 的规则:
        #        
        if elementCountDiff1==elementCountDiff2 : # A-B 的元素增加 == ? -C 的元素增加
            #scoreDesc.append((1 * scoreWeight,"[%s-%s]两组元素增减个数相同"),imgs1Name,imgs2Name)
            if len(imgsFrm1.img1Elements)==len(imgsFrm2.img1Elements):  # A 与 C 的 图形元素 个数相同
                scoreAddTo.addScore(3* scoreWeight,imgs1Name,imgs2Name,"两组元素个数匹配,增减个数相同");    
                # 再检查, 是否 变动的 元素 相同:
                if elementCountDiff1>0:   # A->B , C->? 图形元素增加 相同元素
                    imgFrameBD = self.getImages2(imgs1Name[1]+imgs2Name[1])  # Frame  B?
                    if imgFrameBD.isImgElementsEqualsOrSimilar(len(imgsFrm1.img1Elements)) : # C 与 答案 新加的元素 相同 
                        scoreAddTo.addScore(7* scoreWeight,imgs1Name,imgs2Name,"两组增加了相同类型的元素");   
                    elif imgsFrm1.isIncSameElements() and imgsFrm2.isIncSameElements():                 # Challenge Problems B-01 : AC-B6
                        scoreAddTo.addScore(7* scoreWeight,imgs1Name,imgs2Name,"两组增加了同倍数的相同元素");   
                elif elementCountDiff2<0: # C-? 图形元素减少
                    imgFrameAC = self.getImages2(imgs1Name[0]+imgs2Name[0])  # Frame  AC 
                    if imgFrameAC.isImgElementsEqualsOrSimilar(len(imgsFrm1.img2Elements)) : # C 与 答案 的元素 相同 
                        scoreAddTo.addScore(7* scoreWeight,imgs1Name,imgs2Name,"两组减少了相同类型的元素")
                    elif imgsFrm1.isIncSameElements() and imgsFrm2.isIncSameElements():                 
                        scoreAddTo.addScore(7* scoreWeight,imgs1Name,imgs2Name,"两组减少了同倍数的相同元素");       
            else:
                scoreAddTo.addScore(2* scoreWeight,imgs1Name,imgs2Name,"两组元素增减个数相同")
                # Challenge Problem B-03 : 需要 较高的 分数, 

        #
        # 判断像素 比例 变化规律
        #
        
        r1 = imgsFrm1.getImagePixelRatio()
        r2 = imgsFrm2.getImagePixelRatio()
        diff = abs(r1-r2)
        if  diff< 0.05:
            scoreAddTo.addScore(3,imgs1Name,imgs2Name,"两图片素个数变化率相差<0.05") 
        elif diff < 0.1:
            scoreAddTo.addScore(2,imgs1Name,imgs2Name,"两图片像素个数变化率相差<0.1") 
        elif diff < 0.15:
            scoreAddTo.addScore(1,imgs1Name,imgs2Name,"两图片像素个数变化率相差<0.15") 

        #
        #  Challenge Problem B-04 : 区答案 3,4
        #     
        if caseCheckFilled:
            filledFlags =  ImageElement.getImagesFilledFlags(imgsFrm1.img1Elements) #[elementIdx].isFilledImage()
            if (filledFlags==1 or filledFlags==2 )  \
                and filledFlags==ImageElement.getImagesFilledFlags(imgsFrm2.img1Elements) \
                and filledFlags==ImageElement.getImagesFilledFlags(imgsFrm1.img2Elements) \
                and filledFlags==ImageElement.getImagesFilledFlags(imgsFrm2.img2Elements) :
                    scoreAddTo.addScore(1*scoreWeight,imgs1Name,imgs2Name,"两组元素全为填充图" if filledFlags==1 else "两组元素全为非填充图") 

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
                    scoreAddTo.addScore(0.5*scoreFactor*scoreWeight,imgs1Name,imgs2Name,"元素%d轮廓相似, 相似个数变化趋势相近"%(elementIdx))
            #if eqTrans1.matched!=eqTrans2.matched:

            #scoreFactor




    #END method calculateImages2MatchScore
    
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
        caseAddOrSubEq = True 
        caseXorEq = True
        caseOuterSharpCmp = True  # 判断外形  
        caseXorCmp = True
        caseAndCmp = True
        caseCheckEqsIgnoreInced = True
        #  
        # 判断 两组 图片 为 相同组合 或 完成 相同
        # 例子:  
        #   C-02 (全相同) :   D-02-不同组合   
        #
        if len(idxOfImgs2)==3:
            #print("%s-%s : idxOfImgs2 = %s" %(imgs1Name,imgs2Name,idxOfImgs2))
            score = 10
            desc = "两组图形具有相同组合"  
            #scoreAddTo.addScore(10 * scoreWeight ,imgs1Name,imgs2Name,"两组图形具有相同组合")
            caseAddOrSubEq = False
            caseXorEq = False
            caseOuterSharpCmp = False
            caseXorCmp = False
            caseAndCmp = False
            caseCheckEqsIgnoreInced = False
            if self.getImages2(imgs2Name[0:2]).isImgElementsEqualsOrSimilar() and self.getImages2(imgs2Name[1:3]).isImgElementsEqualsOrSimilar():
                # GH? 图形相同, ABC 也相同 ( 因为 同组合 )
                if self.getImages2(imgs2Name[0:2]).isImgElementsEquals() and self.getImages2(imgs2Name[1:3]).isImgElementsEquals():
                    score += 12
                    desc = "两组图形6个全相同"  
                else:                    
                    score += 10
                    desc = "两组图形6个全相似"  
                #scoreAddTo.addScore(10 * scoreWeight ,imgs1Name,imgs2Name,"两组图形6个全相同")
                all6ImgEquals = True
            scoreAddTo.addScore(score * scoreWeight ,imgs1Name,imgs2Name,desc) # C-02
        elif    self.getImages2(imgs1Name[0:2]).isImgElementsEqualsOrSimilar() and self.getImages2(imgs1Name[1:3]).isImgElementsEqualsOrSimilar() \
            and self.getImages2(imgs2Name[0:2]).isImgElementsEqualsOrSimilar() and self.getImages2(imgs2Name[1:3]).isImgElementsEqualsOrSimilar():
            eq =  self.getImages2(imgs1Name[0:2]).isImgElementsEquals() and self.getImages2(imgs1Name[1:3]).isImgElementsEquals() \
                and self.getImages2(imgs2Name[0:2]).isImgElementsEquals() and self.getImages2(imgs2Name[1:3]).isImgElementsEquals()
            # ABC 相等, GHI 相等
            scoreAddTo.addScore((12 if eq else 10) * scoreWeight ,imgs1Name,imgs2Name,"每组图形全相等" if eq else "每组图形全相似")
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
        if elementCountIncAB==elementCountIncGH and elementCountIncBC==elementCountIncHI:  # 元素个数递增 相同
            if elementCountIncAB==0 and elementCountIncBC==0 :
                nElementIfSame1 = len(imgsFrm1.img1Elements)
                nElementIfSame2 = len(imgsFrm2.img1Elements)
            score = 4   
            if elementCountIncAB==0 and elementCountIncBC==0 and len(imgsFrm1.img1Elements)==len(imgsFrm2.img1Elements):  
                score += 1 
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
            scoreAddTo.addScore(score * scoreWeight ,imgs1Name,imgs2Name,desc)
            #
            # D-06 : 比较最后一个元素 的,  ABC - GH1
            #
            #print("%s/%s nElementIfSame1=%d,nElementIfSame2=%d" %(imgs1Name,imgs2Name,nElementIfSame1,nElementIfSame2))
            if nElementIfSame1>0 and nElementIfSame2>0:
                i1 = imgsFrm1.getNotEqImgElementIdx()
                #print("%s/%s i1=%d" %(imgs1Name,imgs2Name,i1))
                if i1>=0  :
                    i2=imgsFrm2.getNotEqImgElementIdx()
                    #print("%s/%s i2=%d" %(imgs1Name,imgs2Name,i2))
                    if i2>0:
                        #print("%s/%s i1=%d,i2=%d" %(imgs1Name,imgs2Name,i1,i2))
                        if ImageElement.isElementsEqualsIgnoreOrder([imgsFrm1.img1Elements[i1],imgsFrm1.img2Elements[i1],imgsFrm1.img3Elements[i1]],[imgsFrm2.img1Elements[i2],imgsFrm2.img2Elements[i2],imgsFrm2.img3Elements[i2]]):
                            scoreAddTo.addScore(4 * scoreWeight ,imgs1Name,imgs2Name,"第一组的第%d元素与第二组的第%d元素相同组合" %(i1,i2))
                            pass
            #
            #  D-06  AB1 - BFG
            #             
            if elementCountIncAB==-1 and elementCountIncAB==elementCountIncBC :
                if(   ImageElement.isElementsContains(imgsFrm1.img2Elements,imgsFrm1.img1Elements) and ImageElement.isElementsContains(imgsFrm1.img3Elements,imgsFrm1.img2Elements)   \
                   and ImageElement.isElementsContains(imgsFrm2.img2Elements,imgsFrm2.img1Elements) and ImageElement.isElementsContains(imgsFrm2.img3Elements,imgsFrm2.img2Elements) \
                     ):
                    scoreAddTo.addScore(2 * scoreWeight ,imgs1Name,imgs2Name,"子集关系")
            elif elementCountIncAB==1 and elementCountIncAB==elementCountIncBC :
                if(   ImageElement.isElementsContains(imgsFrm1.img1Elements,imgsFrm1.img2Elements) and ImageElement.isElementsContains(imgsFrm1.img2Elements,imgsFrm1.img3Elements)   \
                   and ImageElement.isElementsContains(imgsFrm2.img1Elements,imgsFrm2.img2Elements) and ImageElement.isElementsContains(imgsFrm2.img2Elements,imgsFrm2.img3Elements) \
                     ):
                    scoreAddTo.addScore(2 * scoreWeight ,imgs1Name,imgs2Name,"子集关系")
                pass           
        elif len(imgsFrm1.img1Elements)>0 \
            and  len(imgsFrm1.img1Elements)>0 and len(imgsFrm2.img1Elements)>0 \
            and len(imgsFrm1.img2Elements)/len(imgsFrm1.img1Elements)==len(imgsFrm2.img2Elements)/len(imgsFrm2.img1Elements) \
            and len(imgsFrm1.img3Elements)/len(imgsFrm1.img1Elements)==len(imgsFrm2.img3Elements)/len(imgsFrm2.img1Elements) : 
            scoreAddTo.addScore(4 * scoreWeight ,imgs1Name,imgs2Name,"两组图形元素个数变化按同倍数递增")  #C-03 :
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
                    scoreAddTo.addScore(2 * scoreWeight ,imgs1Name,imgs2Name,"两组图形面积Delta变化趋势相同")
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
                if    imgsAB.isImgElementTransMatched(i,IMGTRANSMODE_ROTATE1) \
                  and self.getImages2(imgs1Name[1:3]).isImgElementTransMatched(i,IMGTRANSMODE_ROTATE1) \
                  and self.getImages2(imgs2Name[0:2]).isImgElementTransMatched(i,IMGTRANSMODE_ROTATE1) \
                  and self.getImages2(imgs2Name[1:3]).isImgElementTransMatched(i,IMGTRANSMODE_ROTATE1) :
                    scoreAddTo.addScore( 10*scoreFac* scoreWeight/nElementIfSame,imgs1Name,imgs2Name,"两组图形为90度旋转关系")
                    caseAddOrSubEq = False
                    caseXorEq = False
                    caseAndCmp = False
                    continue
                if   imgsAB.isImgElementTransMatched(i,IMGTRANSMODE_ROTATE3) \
                  and self.getImages2(imgs1Name[1:3]).isImgElementTransMatched(i,IMGTRANSMODE_ROTATE3) \
                  and self.getImages2(imgs2Name[0:2]).isImgElementTransMatched(i,IMGTRANSMODE_ROTATE3) \
                  and self.getImages2(imgs2Name[1:3]).isImgElementTransMatched(i,IMGTRANSMODE_ROTATE3) :
                    scoreAddTo.addScore( 10*scoreFac* scoreWeight/nElementIfSame,imgs1Name,imgs2Name,"两组图形为-90度旋转关系")
                    caseAddOrSubEq = False
                    caseXorEq = False
                    caseAndCmp = False
                    continue
                imgsAC = self.getImages2(imgs1Name[0:1]+imgs1Name[2:3])
                if (imgsAC.isImgElementTransMatched(i,IMGTRANSMODE_ROTATE1) and self.getImages2(imgs2Name[0:1]+imgs2Name[2:3]).isImgElementTransMatched(i,IMGTRANSMODE_ROTATE1) )\
                       or (imgsAC.isImgElementTransMatched(i,IMGTRANSMODE_ROTATE3) and self.getImages2(imgs2Name[0:1]+imgs2Name[2:3]).isImgElementTransMatched(i,IMGTRANSMODE_ROTATE3)) :
                    caseAddOrSubEq = False
                    caseXorEq = False
                    caseAndCmp = False
                    # 
                    if imgsAB.isBlackPixelRatioEquals(i) and self.getImages2(imgs2Name[0:2]).isBlackPixelRatioEquals(i) : # todo 需要判断 满足 45 度的旋转 ,暂时 使用 isBlackPixelRatioEquals 代替
                        # Challenge D-02 :  [CDH-AE2]两组图形为45度旋转关系 : 判断错误, 暂时 降为 4 分
                        scoreAddTo.addScore( 4*scoreFac* scoreWeight/nElementIfSame,imgs1Name,imgs2Name,"两组图形为45度旋转关系")
                    continue
                #elif imgsAB.isImgElementTransMatched(i,IMGTRANSMODE_ROTATE3) 
        # END if nElementIfSame>0: #六个 图形 具有 相同 元素 个数
        
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
        for flipMode in [IMGTRANSMODE_FLIPV,IMGTRANSMODE_FLIPH]:
            if imgsAG.isWholeImgElementsFliped(flipMode) and  imgsBH.isWholeImgElementsFliped(flipMode) and  imgsCI.isWholeImgElementsFliped(flipMode):
                scoreAddTo.addScore( 10* scoreWeight,imgs1Name,imgs2Name,"两组图形%s翻转关系" %("水平" if flipMode==IMGTRANSMODE_FLIPH else "上下"))
                break


        if caseOuterSharpCmp and nElementIfSame>0:
            # 判断 外形相似  : D-09
            idxOfImgs2 = []  # 第二组图片 对应在 第一组 图片中的序号 
            for i in range(3):
                j = imgsFrm1.getIndexOfOutterSharpEqImage(imgs2Name[i],idxOfImgs2)  # 第二组图片的 第 i 个图片, 在 第一组 中对应的序号 ( 不存在时 j==-1 )
                if j<0:
                    break
                idxOfImgs2.append(j)
            if len(idxOfImgs2)==3:
                scoreAddTo.addScore( 3,imgs1Name,imgs2Name,"两组图形外形具有相同组合") #D-09
            
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
                scoreAddTo.addScore( 10,imgs1Name,imgs2Name,"两组图形具有相同组合,并同时增加相同个数(%d)元素,对应序号=%s" %(inced0,idxOfImgs2))
                pass    
        #
        # 三个图形 异或 后, 相同  : D-09
        #     
        if caseXorCmp:
            xorImg1 = imgsFrm1.getXORImageElement()   
            xorImg2 = imgsFrm2.getXORImageElement()   
            ratio,_,_ = countImageDiffRatio(xorImg1.image,xorImg2.image)
            #print("------%s - %s : xorImgDiffRatio = %s" %(xorImg1.name,xorImg2.name,ratio))
            if ratio<0.03:
                scoreAddTo.addScore( 3 ,imgs1Name,imgs2Name,"两组图形每组XOR后的图形相似") 
                #caseAndCmp = False  
                #caseAddOrSubEq = False
            elif ratio<0.05:
                scoreAddTo.addScore( 1 ,imgs1Name,imgs2Name,"两组图形每组XOR后的图形相似") 
                #caseAndCmp = False 
                #caseAddOrSubEq = False
            
            #if ratio<0.07:
            #    scoreAddTo.addScore( 3 if ratio<0.03 else ( 2 if ratio<0.05 else 1),imgs1Name,imgs2Name,"两组图形每组XOR后的图形相似") 

        if caseAndCmp:
            andMatched = imgsFrm1.getANDMatched() 
            if andMatched>0 and imgsFrm2.getANDMatched()==andMatched:
                if andMatched==1:
                    scoreAddTo.addScore( 5 ,imgs1Name,imgs2Name,"两组图形 %s 位与 %s == %s 且 %s 位与 %s==%s" %(imgsFrm1.imgId1,imgsFrm1.imgId2,imgsFrm1.imgId3,imgsFrm2.imgId1,imgsFrm2.imgId2,imgsFrm2.imgId3)) 
        
        #
        # 是否 匹配 相加 属性 或 相减
        #  即:  图片A + 图片B == 图片C
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
                        scoreAddTo.addScore( 6 * scoreWeight,imgs1Name,imgs2Name,"第%d和%d图片像素合并==第%d个图片"%(abc[0]+1,abc[1]+1,abc[2]+1))
                    caseAddOrSubEq = False    
                    caseXorEq = False    
                    caseAndCmp = False
                    break
        
        # E-04 : 答案 2 与 8 都一样
        if caseAddOrSubEq: # 处理 图片A.像素个数 + 图片B.像素个数 == 图片C 的情况  E-04   
            c1 = imgsFrm1.compareImgPixelCount() 
            if c1>0 and c1==imgsFrm2.compareImgPixelCount() :        
                caseAddOrSubEq = False    
                caseXorEq = False 
                caseAndCmp = False
                #  - E-04 : 答案 2 与 8 都一样 , 进一步考虑图片形状
                if  ( imgsFrm1.compareImgPixelHeight()==0 and imgsFrm2.compareImgPixelHeight()==0 and imgsFrm1.compareImgPixelWidth()==c1 and imgsFrm2.compareImgPixelWidth()==c1  ) \
                 or ( imgsFrm1.compareImgPixelWidth()==0 and imgsFrm2.compareImgPixelWidth()==0  and imgsFrm1.compareImgPixelHeight()==c1 and imgsFrm2.compareImgPixelHeight()==c1):
                    scoreAddTo.addScore( 6 * scoreWeight,imgs1Name,imgs2Name,"前两图片像素个数相加或减==第三个图片,且宽高匹配")
                else:
                    scoreAddTo.addScore( 1 * scoreWeight,imgs1Name,imgs2Name,"前两图片像素个数相加或减==第三个图片")

        #
        #  比较 像素 变化规律: AB , BC , AC
        #             
        for r1,r2,id1,id2 in zip(imgsFrm1.getImagePixelRatio(),imgsFrm2.getImagePixelRatio(),[imgs1Name[0:2],imgs1Name[1:3],imgs1Name[0:1]+imgs1Name[2:3]],[imgs2Name[0:2],imgs2Name[1:3],imgs2Name[0:1]+imgs2Name[2:3]]):
            diff =  abs(r1 - r2)
            #if diff<0.15:
            #    print("diff = %f" %diff)
            if  diff< 0.05:
                scoreAddTo.addScore(3,imgs1Name,imgs2Name,"两图片(%s与%s)像素个数变化率相差<0.05"%(id1,id2)) 
            elif diff < 0.1:
                scoreAddTo.addScore(2,imgs1Name,imgs2Name,"两图片(%s与%s)像素个数变化率相差<0.1"%(id1,id2)) 
            elif diff < 0.15:
                scoreAddTo.addScore(1,imgs1Name,imgs2Name,"两图片(%s与%s)像素个数变化率相差<0.15"%(id1,id2)) 
        #
        # XOR 的例子: D-11
        # 
        if caseXorEq:
            pass # todo                
    # END method calculateImages3MatchScore 

    def _printAnswerScoreDetails(answerScore:AnswerScore)->None:
        if len(answerScore.scoreDetails)==0:
            print("答案 [%d] 总得分 = %.2f " % (answerScore.answer,answerScore.score))
        else:
            print("答案 [%d] 总得分 = %.2f , 其中 " % (answerScore.answer,answerScore.score))
            for scoreDetail in answerScore.scoreDetails:
                print("  得分 %.2f 来自于: [%s-%s]%s" % scoreDetail)
        
################################################################
# solve_2x2
#############################################################
    def solve_2x2(self):
        answers = []
        #for answer in self.potential_answers:
        for answer in self.images:
            if not answer.isdigit():
                continue
            answerScore = AnswerScore(int(answer))   
            self.calculateImages2MatchScore("AB","C"+answer,answerScore)  # 行比较 : 第一行 与 第 三 行
            self.calculateImages2MatchScore("AC","B"+answer,answerScore)
            #  self.calculateImages2MatchScore("BC","A"+answer,answerScore,0.5) # 对角线 ????
            if Agent._DEBUG:
                Agent._printAnswerScoreDetails(answerScore)
            answers.append(answerScore)
        answers = AnswerScore.getMaxScoreAnswers(answers)
        return answers[0].answer


###############################################################
#
#  solve_3x3 
# 
##################################################################        

    def solve_3x3(self):
        answers = []
        for answer in self.images:
            if not answer.isdigit():
                continue
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
            if Agent._DEBUG:
                Agent._printAnswerScoreDetails(answerScore)
            answers.append(answerScore)
        answers = AnswerScore.getMaxScoreAnswers(answers)
        return answers[0].answer

    def prepareProblem(self, problem):
        self.images = load_problem_images(problem)
        self.imagesFrame = {}
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




