import cv2
import numpy as np
import os


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
    potential_answers = {}

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
        if key.isalpha():
            images[key] = ravens_image
        elif key.isdigit():
            potential_answers[key] = ravens_image

    return images, potential_answers

    #
    # return 0 : v1==v2==v3
    #        1 : v1+v2==v3
    #        2 : v1-v2==v3 
    #
def _compare3(v1,v2,v3)->bool:
    if v2>0 and abs((v1+v2-v3) / v3 )<0.01:
        return 1
    if v1>0 and abs((v2+v3-v1) / v1 )<0.01:
        return 2
    if v3>0 and abs((v1-v3) / v3 )<0.01 and abs((v2-v3) / v3 )<0.01:
        return 0
    return -1
#
#  images1Lst 中的前景像素 与 images2Lst 中的前景像素 的 个数 差值
#  [E-02] AB 与 C 前景像素 相差 272/3596/33856
#    GH 与 7 前景像素 相差 457/6023/33856   
#  [E-03] AB 与 C 前景像素 相差 142/1848/33856
#  [E-03] GH 与 2 前景像素 相差 534/2626/33856     534/33856=0.0158  534/2626=0.203
#

def countImagesDiff(images1Lst:list,images2Lst:list)->int:
    count = 0
    blackCount = 0
    #n1 , n2 = len(images1Lst) ,len(images2Lst)
    height, width = images1Lst[0].shape
    for img in images1Lst:
        h, w = img.shape
        if height>h:
            height = h
        if width>w:
            width = w
    for img in images2Lst:
        h, w = img.shape
        if height>h:
            height = h
        if width>w:
            width = w

    for y in range(height):
        for x in range(width):
            v1 = False
            v2 = False
            for img in images1Lst:
                if img[y,x]==0:
                    v1 = True
                    break
            for img in images2Lst:
                if img[y,x]==0:
                    v2 = True
                    break
            if v1!=v2:
                count += 1  
                blackCount += 1
            elif v1:
                blackCount += 1
                  
    return count ,blackCount, height*width 

def countImagesXOR(imagesLst:list)->int:
    count = 0
    blackCount = 0
    #n1 , n2 = len(images1Lst) ,len(images2Lst)
    height, width = imagesLst[0].shape
    for img in imagesLst:
        h, w = img.shape
        if height>h:
            height = h
        if width>w:
            width = w
    image = np.full((height, width), 255, np.uint8)  #        
    for y in range(height):
        for x in range(width):
            v = False
            hasBlack = False
            for img in imagesLst:
                if img[y,x]==0 :
                    v = not v
                    hasBlack = True
            if v:
                count += 1
                image[y,x] = 0
            if hasBlack:
                blackCount += 1     
    return count ,blackCount, height*width ,image

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
def getHLineSegments(image,y,fromX=0,endX=0):
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
               setments.append((x1,x))
               x1 = -1   
    if x1>=0:
        setments.append((x1,x))
    return setments

#
# 获取 垂直线(坐标==x) 方向 上的线段  
#
def getVLineSegments(image,x,fromY=0,endY=0):
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
               setments.append((y1,y))
               y1 = -1   
    if y1>=0:
        setments.append((y1,y))
    return setments

def  toRelativeCenterPoint(image,x,y):
    height,width = image.shape
    return x-width/2, y-height/2

def indexOf(list,value):
    if list==None:
        return -1
    try:
        return list.index(value)
    except ValueError:    
        return -1

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
        #self.match = "none"
        #  "DELETED",  
        #self.transform = "not matched"
        #self.match_weight = 0
    def update(self):
        self.blackPixelCount= 0
        self.transformImgs = {} # 缓存 翻转, 旋转, 填充 等 变换
        self.hLineSegs = {}  # 缓存 水平 线段
        self.vLineSegs = {} # 缓存 垂直 线段
        height, width = self.image.shape
        for y in range(height):
            for x in range(width):
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
  
    def getHLineSegments(self,y):
        if y in self.hLineSegs:
            #print("使用缓存")
            return self.hLineSegs[y]
        v = getHLineSegments(self.image,y,self.x0,self.ex)
        self.hLineSegs[y] = v
        return v
    def getVLineSegments(self,x):
        if x in self.vLineSegs:
            #print("使用缓存")
            return self.vLineSegs[x]
        v = getVLineSegments(self.image,x,self.y0,self.ey)
        self.vLineSegs[x] = v
        return v
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
    #  返回   (图1 与  图2 的 相似度(0-1.0), 图1/图2 图形大小比例)
    #
    def getImageElementSimilarScale(self,otherImgElement):
        hwMatched,scale = self.isImageShapeHWSimilar(otherImgElement)
        if not hwMatched: # 两个图的 长 宽 比例 不一致, 认为他们 不 相似
            #if( Agent._DEBUG ):
            #    print("scaleH=%f,scaleW=%f 图1(h=%d,w=%d),图2(h=%d,w=%d)" %(scale,w1 /  w2,h1,w1,h2,w2))
            return 0 ,0
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
        comSampleCount = 50    
        #
        # 水平方法抽取 50 个样本 线段
        # 垂直方法抽取 50 个样本 线段
        #  计算这些字段 相似度
        # ( 不检测所有点 只是 为了效率) 
        # 
        #dy = smallElement
        hlinesY = ImageElement.getSamplePointForImgSimilarDetect(smallElement.y0,smallElement.ey,comSampleCount)
        vlinesX = ImageElement.getSamplePointForImgSimilarDetect(smallElement.x0,smallElement.ex,comSampleCount)
        totalLines = 0     # 总 检测的 线段 数
        matchedLines = 0   # 其中 匹配的 线段数
        maxLineCount = max(len(vlinesX),len(hlinesY))
        for i in range(maxLineCount):
            if i<len(hlinesY): # 水平线
                hlineY1 = hlinesY[i]
                hlineY2 = largeElement.y0+int((hlineY1-smallElement.y0)*largeScaleH+0.5)
                totalLines += 1
                if ImageElement.isImageElementLineSegmentSimilar(smallElement,largeElement,hlineY1,hlineY2,largeScaleW,0):
                    matchedLines += 1
            if i<len(vlinesX): # 垂直线
                vlineX1 = vlinesX[i]
                vlineX2 = largeElement.x0+int((vlineX1-smallElement.x0)*largeScaleW+0.5)
                totalLines += 1
                if ImageElement.isImageElementLineSegmentSimilar(smallElement,largeElement,vlineX1,vlineX2,largeScaleH,1):
                    matchedLines += 1
            # 前 10 组 线 匹配程度<70% 返回 0        
            if (i==10 or i==20) and matchedLines/totalLines < 0.7:
                return 0,scale
        #if Agent._DEBUG:    
        #    print("matchedLines/totalLines  = %d/%d" %(matchedLines,totalLines ))    
        return matchedLines/totalLines ,scale      
    
    #
    #  @param hvType :0 水平线, 1:垂直线
    #   scale 是 imageElement2 / imageElement1 的 图形放缩比例
    #  @param imageElement1 小图, imageElement2 : 大图
    #    
    def isImageElementLineSegmentSimilar(imageElement1,imageElement2,linePos1:int,linePos2:int,scale:float,hvType):   
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
        #print("nSegments = %d, nSegments2 = %d" %(nSegments,len(lineSegments2))) 
        CenterPointDelta = 3
        if nSegments!=len(lineSegments2):
            #
            # 处理 特除情况, 在 边框线 , 
            #
            if hvType==0: # 水平线
                minP, maxP = imageElement1.y0,imageElement1.ey
            else:
                minP, maxP = imageElement1.x0,imageElement1.ex     
            if nSegments==1 and len(lineSegments2)==2 and scale>0 and (linePos1<minP+5 or linePos1>maxP-5):
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
            #    print("%s 位置=%d,%d : 线段数 不等 :%d!=%d 小图区间=(%d %d) 大图区间=(%d %d),minP=%d,maxP=%d " %("水平线" if hvType==0 else "垂直线",linePos1,linePos2,nSegments,len(lineSegments2),imgElement1Pos,imgElement1PosEnd,imgElement2Pos,imgElement2PosEnd,minP, maxP ))
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
            dw = abs(w2-w1*scale)
            if dw>4 and dw/w2>0.05: # 线段宽度
                #if( Agent._DEBUG ):
                #    print("%s 位置=%d,%d : 宽度不等 %f!=%f "%("水平线" if hvType==0 else "垂直线",linePos1,linePos2,w1*scale,w2 ) )
                return False
        return True         
             
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
    
    def getFlipedImage(self,flipMode:int):
        imgKey = IMGTRANSMODE_FLIPVH if flipMode==-1 else ( IMGTRANSMODE_FLIPV if flipMode==0 else IMGTRANSMODE_FLIPH )
        if imgKey in self.transformImgs:
            #print("使用缓存图片...")
            return self.transformImgs.get(imgKey)
        imgElement = ImageElement(self.image.shape,self.name+"-"+imgKey)
        img = imgElement.image #np.full(self.image.shape, 255, np.uint8)
        imgRegion = self.image[self.y0:self.ey, self.x0:self.ex]
        imgRegion = cv2.flip(imgRegion,flipMode ) 
        img[self.y0:self.ey, self.x0:self.ex] = imgRegion
        imgElement.update()
        self.transformImgs[imgKey] = imgElement
        return imgElement
    
    def getWholeFlipedImage(self,flipMode:int):
        imgKey = IMGTRANSMODE_WHOLEFLIPVH if flipMode==-1 else ( IMGTRANSMODE_WHOLEFLIPV if flipMode==0 else IMGTRANSMODE_WHOLEFLIPH )
        if imgKey in self.transformImgs:
            #print("使用缓存图片...")
            return self.transformImgs.get(imgKey)
        imgElement = ImageElement(None,self.name+"-"+imgKey)
        imgElement.image = cv2.flip(self.image,flipMode ) 
        imgElement.update()
        self.transformImgs[imgKey] = imgElement
        return imgElement
    
    #
    # get
    #
    def  getWholeFlipedCenterPoint(self,flipMode:int):
        height,width = self.image.shape
        wholeX0,wholeY0  = width/2,height/2
        x0,y0 = (self.x0+self.ex)/2,(self.y0+self.ey)/2
        x1,y1 = x0-wholeX0, y0-wholeY0
        if flipMode==-1 : 
            x1 = -x1
            y1 = -y1
        elif flipMode==0 : # 垂直翻转(上下)
            y1 = -y1
        elif flipMode==1 : # 水平翻转 ( 左右 )
            x1 = -x1
        return  int(x1+wholeX0+0.5),int(y1+wholeY0+0.5)


    
    def getRotateImage(self,rotaMode:int):
        #print("rotaMode=",rotaMode )
        if rotaMode==1:
            imgKey = IMGTRANSMODE_ROTATE1
            rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE # ROTATE_90_CLOCKWISE #rotaMode*90
        elif rotaMode==2:
            imgKey = IMGTRANSMODE_ROTATE2
            rotateCode = cv2.ROTATE_180
        elif rotaMode==3:
            imgKey = IMGTRANSMODE_ROTATE3
            rotateCode = cv2.ROTATE_90_CLOCKWISE
        else:
            raise BaseException("Invalid rotaMode=%d" % rotaMode)
        if imgKey in self.transformImgs:
            #print("使用缓存图片...")
            return self.transformImgs.get(imgKey)
        #print("rotateCode = %d"% rotateCode)
        imgElement = ImageElement(self.image.shape,self.name+"-"+imgKey)
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
        self.transformImgs[imgKey] = imgElement
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
            for x in range(self.getStartPointX(y)+1,self.getEndPointX(y)):
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

    #
    # 将两个 图片元素 合并:
    #
    def  mergImgElements() ->list:
        pass
    
    def getSumImgElementsBlackPonts(imgElements:list)->int:
        n = 0
        for e in imgElements:
            n += e.blackPixelCount
        return n    
    def getImgElementsHeight(imgElements:list)->int:
        y0, ey = 10000, 0
        for e in imgElements:
            y0 = min(y0,e.y0)
            ey = max(ey,e.ey)
        return ey-y0   
    def getImgElementsWidth(imgElements:list)->int:
        x0, ex = 10000, 0
        for e in imgElements:
            x0 = min(x0,e.x0)
            ex = max(ex,e.ex)
        return ex-x0  

    def newImageElement(image,x0,y0,flagAdded):
        height, width = image.shape
        #newImage = np.full(image.shape, 255, np.uint8)
        newImageEle = ImageElement(image.shape)
        newImageEle.addPixel(x0,y0)
        #newImage[y0,x0] = 0    
        flagAdded[y0,x0] = True
        #print("newImageElement - %d,%d"%(x0,y0))
        checkPoints = [(y0,x0)]
        while checkPoints:
            y,x = checkPoints.pop()
            # 检查 周围点
            uY = y-1 if y>0 else y
            dY = min(y+2,height) 
            lX = x-1 if x>0 else x
            rX = min(x+2, width)
            for yi in range(uY,dY,1):
                for xi in range(lX,rX,1):
                    if not flagAdded[yi,xi] and image[yi,xi]==0:
                        #print(" ... xi=%d,yi=%d",xi,yi)
                        flagAdded[yi,xi] = True
                        newImageEle.addPixel(xi,yi)
                        checkPoints.append((yi,xi))
        return newImageEle      
    
    
     #
     # 将图片(按像素相连)分隔成多个元素, 相连的像素分在一个元素组中
     #
    def splitImage(image,nameFormat:str) ->list:
        #image0 = image.copy()
        height, width = image.shape
        flagAdded = np.full((height, width), False, np.bool)
        imageParts = []
        #while True:
        #    addedCount = 0
        #    if addedCount
        for y in range(height):
            for x in range(width):
                if not flagAdded[y,x]:
                    if image[y,x]==0 :
                        part = ImageElement.newImageElement(image,x,y,flagAdded)
                        if  part.blackPixelCount>10:  # 孤点 (噪音) 去掉
                            imageParts.append(part)
                    else:
                        flagAdded[y,x] = True     
        # 按元素 面积(像素个数)  从大到小 排序              
        imageParts.sort(key=lambda e:e.getTotalPixel(),reverse=True)   
        if nameFormat!=None:
            for i in range(len(imageParts)):
                imageParts[i].name = nameFormat % i         
        return imageParts        

    # END class ImageElement


    

#
# 两个图片比较 结果常量
#
IMGTRANSMODE_NONE = None # 
IMGTRANSMODE_EQ = "EQUALS"  # UNCHANGED

IMGTRANSMODE_FLIPV = "FLIPV"  #   (以元素为中心)垂直翻转(上下)  flipMode==0
IMGTRANSMODE_FLIPH = "FLIPH"  #   (以元素为中心)水平翻转 ( 左右 ) flipMode==1
IMGTRANSMODE_FLIPVH = "FLIPVH"  #   (以元素为中心)水平翻转 ( 左右 ) flipMode==-1

IMGTRANSMODE_WHOLEFLIPV = "WHOLEFLIPV"  #   (以整个图为中心)垂直翻转(上下)  flipMode==0
IMGTRANSMODE_WHOLEFLIPH = "WHOLEFLIPH"  #   (以整个图为中心)水平翻转 ( 左右 ) flipMode==1
IMGTRANSMODE_WHOLEFLIPVH = "WHOLEFLIPVH"  #   (以整个图为中心)水平翻转 ( 左右 ) flipMode==-1

IMGTRANSMODE_ROTATE1 = "ROTAGE90" #  旋转 90度
IMGTRANSMODE_ROTATE2 = "ROTAGE180" #  旋转 180度
IMGTRANSMODE_ROTATE3 = "ROTAGE270" #  旋转 270度(-90度)  
IMGTRANSMODE_FILLED = "FILLED"
IMGTRANSMODE_UNFILLED = "UNFILLED"
IMGTRANSMODE_SIMILAR = "SIMILAR"

# ImageElement.getFlipTransMode
def getFlipTransMode(flipMode:int)->str:
    if flipMode==0:
        return  IMGTRANSMODE_FLIPV
    if flipMode==1:
        return  IMGTRANSMODE_FLIPH
    if flipMode==-1:
        return  IMGTRANSMODE_FLIPVH
    raise BaseException("Invalid flipMode=%d" % flipMode)

def getFlipModeModeByTransMode(flipTransMode:str)->int:
    if flipTransMode==IMGTRANSMODE_FLIPV:
        return 0
    if flipTransMode==IMGTRANSMODE_FLIPH:
        return 1
    if flipTransMode==IMGTRANSMODE_FLIPVH:
        return -1
    return -2

#IMGTRANSMODE_ADDED = 11
#IMGTRANSMODE_REMOVED = 12
#IMGTRANSMODE_ZOOM = 15  # 放缩 
#
# 描述 两个图形元素的 变换规则, 例如
#       
class ImageElementTrans:
    def __init__(self,transMode,matched:bool,similar:float,scale):
        #self.elementIdx = elementIdx
        self.transMode = transMode  # IMGTRANSMODE_EQ 等
        self.matched = matched # similar > 给定的阈值, 即 表名 两个元素 满足 transMode 的变换规则
        self.similar = similar
        self.scale = scale
            
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
        self.img1Id = name[0:1]   # 如 "A"
        self.img2Id = name[1:2]   # 如 "C"
        self.img1Elements = agent.getImageElements(self.img1Id)  # A  ImageElement[]
        self.img2Elements = agent.getImageElements(self.img2Id)  # C  ImageElement[]
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
    # img1 -> img1 : 元素 增加 个数
    #
    def getImgElementsCountDiff(self):
        return  len(self.img2Elements) - len(self.img1Elements)
   
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
            similar,scale = srcImgElement.getImageElementSimilarScale(dstImgElement)
            return ImageElementTrans(transMode,similar>=0.85,similar,scale)
        if transMode==IMGTRANSMODE_FLIPV or transMode==IMGTRANSMODE_FLIPH or transMode==IMGTRANSMODE_FLIPVH:
            if srcImgElement.isBlackPixelRatioEquals(dstImgElement):
                flipMode = 0 if transMode==IMGTRANSMODE_FLIPV else ( 1 if transMode==IMGTRANSMODE_FLIPH else -1)
                similar,scale = srcImgElement.getFlipedImage(flipMode).getImageElementSimilarScale(dstImgElement)
                #print("flipTransMode=%s : 相似度=%f 比例=%f" %(flipTransMode,similar,scale))
                return  ImageElementTrans(transMode,similar>=0.90,similar,scale)    
            return  ImageElementTrans(transMode,False,0,0)
        if transMode==IMGTRANSMODE_FILLED:    
            if srcImgElement.getBlackPixelRatio()<0.3 and dstImgElement.getBlackPixelRatio()>0.4:
                similar,scale = srcImgElement.getFilledImage().getImageElementSimilarScale(dstImgElement)
                #print("FILLED: 相似度=%f 比例=%f" %(similar,scale))
                return  ImageElementTrans(transMode,similar>=0.90,similar,scale)    
            return  ImageElementTrans(transMode,False,0,0)
        if transMode==IMGTRANSMODE_UNFILLED:    
            if srcImgElement.getBlackPixelRatio()>0.4 and dstImgElement.getBlackPixelRatio()<0.3:
                similar,scale = dstImgElement.getFilledImage().getImageElementSimilarScale(srcImgElement)
                return  ImageElementTrans(transMode,similar>=0.90,similar,scale)    
            return  ImageElementTrans(transMode,False,0,0)

        if transMode==IMGTRANSMODE_ROTATE1 or transMode==IMGTRANSMODE_ROTATE2 or transMode==IMGTRANSMODE_ROTATE3:
            if  transMode==IMGTRANSMODE_ROTATE1:
                rotateMode = 1
            elif transMode==IMGTRANSMODE_ROTATE2:
                rotateMode = 2
            elif transMode==IMGTRANSMODE_ROTATE3:
                rotateMode = 3
            if srcImgElement.isBlackPixelRatioEquals(dstImgElement):
                rotateImg = srcImgElement.getRotateImage(rotateMode)
                if rotateImg==None:
                    return ImageElementTrans(transMode,False,0,0)
                similar,scale = rotateImg.getImageElementSimilarScale(dstImgElement)
                return  ImageElementTrans(transMode,similar>=0.90,similar,scale)  
            return  ImageElementTrans(transMode,False,0,0)


    def isImgElementTransMatched(self,elementIdx:int,transMode:str) ->bool:   
        return self.getImgElementTrans(elementIdx,transMode).matched
    
    def isBlackPixelRatioEquals(self,elementIdx:int,ratioThreadhold=0.05) ->bool:
        return self.img1Elements[elementIdx].isBlackPixelRatioEquals(self.img2Elements[elementIdx],ratioThreadhold)

    #
    # 判断 图片 是基于整个图 翻转
    #
    def isWholemgElementFliped(self,elementIdx:int,flipTransMode:str):
        if not self.isImgElementTransMatched(elementIdx,flipTransMode):
            return False
        flipMode = getFlipModeModeByTransMode(flipTransMode)
        img1 = self.img1Elements[elementIdx]
        img2 = self.img2Elements[elementIdx]
        x1,y1 = img1.getWholeFlipedCenterPoint(flipMode) # A 图 (基于整图)翻转后的 中心点 
        x2,y2 = img2.getCenter() # B 图  中心点 
        #print("%s图(基于整图)翻转后的 中心点 = (%f,%f) , %s图中心点 =  (%f,%f) " %(imgs1Name[0:1],xA0,yA0,imgs1Name[1:2],xB0,yB0))
        return abs(x1-x2)<3 and abs(y1-y2)<3

    """            
    def parseImeElementTrans2(srcImgElement, dstImgElement,cacheTrans,transModeOnly):
        #if( Agent._DEBUG ):
        #    print("parseImeElementTrans %s-%s for %s.... "%(srcImgElement.name, dstImgElement.name,transModeOnly))
        if IMGTRANSMODE_EQ not in cacheTrans and (transModeOnly=="*" or transModeOnly==IMGTRANSMODE_EQ):
            similar,scale = srcImgElement.getImageElementSimilarScale(dstImgElement)
            if similar>=0.85:  # ??? C-11 : 相似度 0.88
                cacheTrans.put(IMGTRANSMODE_EQ,(similar,scale))
            else:
                cacheTrans.put(IMGTRANSMODE_EQ,(0,scale))
            if transModeOnly!="*": # 不再比较其他 规则
                return None
            #if transModeOnly==None and similar>: 
            
        #
        # 其他情况 目前只考虑 相同大小的图形
        # 
        hwMatched,scale = srcImgElement.isImageShapeHWSimilar(dstImgElement)
        if not hwMatched or abs(scale-1)>0.08:
            #  [B-07 - 52] scale == 1.06
            #print("hwMatched=%s scale=%f" %(hwMatched,scale))
            return None
        #getBlackPixelRatio
        for flipMode,flipTransMode in zip([0,1],[IMGTRANSMODE_FLIPV,IMGTRANSMODE_FLIPH]):
            if transModeOnly=="*" or transModeOnly==flipTransMode: # = "FLIPV"  #   垂直翻转(上下)  flipMode==0
                if srcImgElement.isBlackPixelRatioEquals(dstImgElement):
                    similar,scale = srcImgElement.getFlipedImage(flipMode).getImageElementSimilarScale(dstImgElement)
                    #print("flipTransMode=%s : 相似度=%f 比例=%f" %(flipTransMode,similar,scale))
                    if similar>=0.90: 
                        return  (flipTransMode,similar,scale)    
                if transModeOnly!="*": # 不再比较其他 规则
                    return None
        #if transModeOnly==None or transModeOnly==IMGTRANSMODE_FLIPVH:# = "FLIPVH"  #  垂直+ 水平翻转 ( 左右 ) flipMode==1
        #    pass
        if transModeOnly=="*" or transModeOnly==IMGTRANSMODE_FILLED:# = "FILLED"
            #print("图1 前景像素比=%f, 图1 前景像素比=%f" %(srcImgElement.getBlackPixelRatio(),dstImgElement.getBlackPixelRatio()))
            if srcImgElement.getBlackPixelRatio()<0.3 and dstImgElement.getBlackPixelRatio()>0.4:
                similar,scale = srcImgElement.getFilledImage().getImageElementSimilarScale(dstImgElement)
                #print("FILLED: 相似度=%f 比例=%f" %(similar,scale))
                if similar>=0.90: 
                    return  (IMGTRANSMODE_FILLED,similar,scale)    
            if transModeOnly!="*": # 不再比较其他 规则
                return None
        if transModeOnly=="*" or transModeOnly==IMGTRANSMODE_UNFILLED:# = "UNFILLED"
            if srcImgElement.getBlackPixelRatio()>0.4 and dstImgElement.getBlackPixelRatio()<0.3:
                similar,scale = dstImgElement.getFilledImage().getImageElementSimilarScale(srcImgElement)
                if similar>=0.90: 
                    return  (IMGTRANSMODE_UNFILLED,similar,scale)    
            if transModeOnly!="*": # 不再比较其他 规则
                return None
        return None   
    """
    """    
        # 没发现例子中有 旋转 的变换规则  暂时先不考虑
        for rotateMode, rotateTransMode in zip([1,2,3],[IMGTRANSMODE_ROTATE1,IMGTRANSMODE_ROTATE2,IMGTRANSMODE_ROTATE3]):   
            if transModeOnly=="*" or transModeOnly==rotateTransMode:# = "ROTAGE90" #  旋转 90度
                if srcImgElement.isBlackPixelRatioEquals(dstImgElement):
                    similar,scale = srcImgElement.getRotateImage(rotateMode).getImageElementSimilarScale(dstImgElement)
                    if similar>=0.90: 
                        return  (rotateTransMode,similar,scale)  
                if transModeOnly!="*": # 不再比较其他 规则
                    return None
    """            
        
        #if transModeOnly==None or transModeOnly==IMGTRANSMODE_SIMILAR:# = "SIMILAR"   
        #    pass 

    #
    # 判断 从 startElementIdx - endElementIdx 的元素 相同 或 相似
    #
    def isImgElementsEquals(self,startElementIdx=0,endElementIdx=0):
        if endElementIdx==0:
            endElementIdx = len(self.transElements)
        for elementIdx in range(startElementIdx,endElementIdx):
            transInfo = self.getImgElementTrans(elementIdx,IMGTRANSMODE_EQ)
            if not transInfo.matched:
                return False
        return True    


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
        self.imgId1 = name[0:1]   # 如 "A"
        self.imgId2 = name[1:2]   # 如 "B"
        self.imgId3 = name[2:3]   # 如 ""
        self.img1Elements = agent.getImageElements(self.imgId1) # A 的图片元素
        self.img2Elements = agent.getImageElements(self.imgId2) # B 的图片元素
        self.img3Elements = agent.getImageElements(self.imgId3) # B 的图片元素
        self.imgsElementsLst = [self.img1Elements,self.img2Elements,self.img3Elements]
        #self.frmType = frmType
        
    #
    # 在 三个图形中 找到 与 otherImgId 相等或相似的 图片序号(0,1,2 之一)
    # @param otherImgId "A","B","1","2",... 等
    # 
    def getIndexOfEqualsImage(self,otherImgId:str,excludeIdxs:list=None) ->int:    
        otherimgElements = self.agent.getImageElements(otherImgId) 
        nElements = len(otherimgElements)
        for i in range(3):
            if indexOf(excludeIdxs,i)<0 and len(self.imgsElementsLst[i])==nElements and self.agent.getImages2(self.name[i:i+1]+otherImgId).isImgElementsEquals():
                return i
        return -1
 
    #
    # @return 1 : 图1+图2==图3
    #         2 : 图1-图2==图3
    #         0 : 图1 == 图2 == 图3
    #
    def compareImgPixelCount(self):
        img1BlackPoints = ImageElement.getSumImgElementsBlackPonts(self.img1Elements)
        img2BlackPoints = ImageElement.getSumImgElementsBlackPonts(self.img2Elements)
        img3BlackPoints = ImageElement.getSumImgElementsBlackPonts(self.img3Elements)
        return _compare3(img1BlackPoints,img2BlackPoints,img3BlackPoints)
        
    
    def compareImgPixelHeight(self)->bool:
        h1 = ImageElement.getImgElementsHeight(self.img1Elements)
        h2 = ImageElement.getImgElementsHeight(self.img2Elements)
        h3 = ImageElement.getImgElementsHeight(self.img3Elements)
        return _compare3(h1,h2,h3)
    
    def compareImgPixelWidth(self)->bool:
        w1 = ImageElement.getImgElementsWidth(self.img1Elements)
        w2 = ImageElement.getImgElementsWidth(self.img2Elements)
        w3 = ImageElement.getImgElementsWidth(self.img3Elements)
        return _compare3(w1,w2,w3)
    

 
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
        self.images = {}
        self.potential_answers = {}
        self.imagesEles = {}
        self.imagesFrame = {} 

    def  getImage(self,imageId:str):
        if imageId>="1" and imageId<="9":
            return self.potential_answers[imageId]
        else:
            return  self.images[imageId]    
    #
    # @param imageId image id,  如 "A", "B", "C", "1", "2" 等
    # @return 返回 数组: [元素1,元素2,...]
    #
    def  getImageElements(self,imageId:str) ->list:
        imgElemets = self.imagesEles.get(imageId)
        if imgElemets!=None :
            return imgElemets
        if imageId>="1" and imageId<="9":
            image =  self.potential_answers[imageId]
        else:
            image =  self.images[imageId]    
        imgElemets = ImageElement.splitImage(image,imageId+"[%d]")
        self.imagesEles[imageId] = imgElemets
        return imgElemets
    
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
    def calculateImages2MatchScore(self,imgs1Name,imgs2Name,scoreAddTo:AnswerScore,scoreWeight=1): 
        imgsFrm1 = self.getImages2(imgs1Name)
        imgsFrm2 = self.getImages2(imgs2Name)
        for elementIdx in range( min(imgsFrm1.getImgElementCount(),imgsFrm2.getImgElementCount()) ):
            eqTrans = imgsFrm1.getImgElementTrans(elementIdx,IMGTRANSMODE_EQ)
            if eqTrans.matched:  # 如果 两个 图片 相等, 不再判断 其他 转换:
                forAllTrans = [eqTrans]
            else:
                forAllTrans = imgsFrm1.getAllImgElementTrans(elementIdx,[IMGTRANSMODE_EQ,IMGTRANSMODE_FLIPV,IMGTRANSMODE_FLIPH,IMGTRANSMODE_FILLED,IMGTRANSMODE_UNFILLED]) 
            #
            # 以上 相等, 翻转, 填充 都不满足的情况下 才 考虑 旋转 的情况, 
            #
            caseRotate = True
            for transInfo in forAllTrans:
                if transInfo.matched:
                    caseRotate = False
                    break
            if caseRotate:
                for transMode in [IMGTRANSMODE_ROTATE1,IMGTRANSMODE_ROTATE3,IMGTRANSMODE_ROTATE2]:
                    t = imgsFrm1.getImgElementTrans(elementIdx,transMode)
                    if t.matched:
                        forAllTrans.append(t)
                        break
            for transInfo in forAllTrans:  # r similar,scale
                if not transInfo.matched:
                    continue
                #if Agent._DEBUG:
                #    print("%s : 满足变换规则 %s" %(imgs1Name,transInfo.transMode))
                transInfo2 = imgsFrm2.getImgElementTrans(elementIdx,transInfo.transMode)
                #if Agent._DEBUG:
                #    print("  %s : 检测是否满足变换规则 %s 结果 = %s" %(imgs2Name,transInfo.transMode,transInfo2.matched))
                if not transInfo2.matched:
                    continue
                score = 10 
                # 翻转, 的情况下, 如果 都是基于 整个 图 反转, 加分
                desc2 = ""
                if transInfo.transMode==IMGTRANSMODE_FLIPV or transInfo.transMode==IMGTRANSMODE_FLIPV or transInfo.transMode==IMGTRANSMODE_FLIPVH:
                    # 如果同时基于 整图 翻转, 加分
                    if imgsFrm1.isWholemgElementFliped(elementIdx,transInfo.transMode) and  imgsFrm2.isWholemgElementFliped(elementIdx,transInfo.transMode):
                        score += 3 
                        desc2 += "(基于整图翻转)"
                elif transInfo.transMode==IMGTRANSMODE_EQ:
                    # 正方形, 圆形 等对称图形, 如果 同时 基于整图翻转 , 加分
                    if (imgsFrm1.isWholemgElementFliped(elementIdx,IMGTRANSMODE_FLIPV) and  imgsFrm2.isWholemgElementFliped(elementIdx,IMGTRANSMODE_FLIPV))\
                    or (imgsFrm1.isWholemgElementFliped(elementIdx,IMGTRANSMODE_FLIPV) and  imgsFrm2.isWholemgElementFliped(elementIdx,IMGTRANSMODE_FLIPV)):
                        # B-05
                        score += 3 
                        desc2 += "(基于整图翻转)"
                scoreAddTo.addScore(score*scoreWeight,imgs1Name,imgs2Name,"元素%d匹配相同变换%s%s"%(elementIdx,transInfo.transMode,desc2))
                    
        #
        # 考虑 元素 增加 / 减少 的规则:
        #        
        elementCountDiff1 = imgsFrm1.getImgElementsCountDiff()  # count(B) - count(A)
        elementCountDiff2 = imgsFrm2.getImgElementsCountDiff()  # count(?) - count(C)
        if elementCountDiff1==elementCountDiff2 : # A-B 的元素增加 == ? -C 的元素增加
            #scoreDesc.append((1 * scoreWeight,"[%s-%s]两组元素增减个数相同"),imgs1Name,imgs2Name)
            if len(imgsFrm1.img1Elements)==len(imgsFrm2.img1Elements):  # A 与 C 的 图形元素 个数相同
                scoreAddTo.addScore(3* scoreWeight,imgs1Name,imgs2Name,"两组元素个数匹配,增减个数相同");    
                # 再检查, 是否 变动的 元素 相同:
                if elementCountDiff1>0:   # A->B , C->? 图形元素增加
                    imgFrameBD = self.getImages2(imgs1Name[1:2]+imgs2Name[1:2])  # Frame  B?
                    if imgFrameBD.isImgElementsEquals(len(imgsFrm1.img1Elements)) : # C 与 答案 新加的元素 相同 
                        scoreAddTo.addScore(7* scoreWeight,imgs1Name,imgs2Name,"两组增加了相同类型的元素");     
                elif elementCountDiff2<0: # C-? 图形元素减少
                    imgFrameAC = self.getImages2(imgs1Name[0:1]+imgs2Name[0:1])  # Frame  AC 
                    if imgFrameAC.isImgElementsEquals(len(imgsFrm1.img2Elements)) : # C 与 答案 的元素 相同 
                        scoreAddTo.addScore(7* scoreWeight,imgs1Name,imgs2Name,"两组减少了相同类型的元素")
            else:
                scoreAddTo.addScore(1* scoreWeight,imgs1Name,imgs2Name,"两组元素增减个数相同")
            
    #END method calculateImages2MatchScore
    
    #
    # 计算 两帧 图片(如 ABC 与 GH1) 之间 属性匹配程度的 得分
    #    self.calculateImages3MatchScore("ABC") 
    # @param imgs1Name,imgs2Name:  一行 或 一列 或 对角线 的 三个图 ,如 "ABC","ADG", "GH1" ,"CF1" 等
    # @param scoreAddTo 得分结果 累加到 scoreAddTo 中
    #
    def calculateImages3MatchScore(self,imgs1Name,imgs2Name,scoreAddTo:AnswerScore,scoreWeight=1):
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
            j = imgsFrm1.getIndexOfEqualsImage(imgs2Name[i:i+1],idxOfImgs2)  # 第二组图片的 第 i 个图片, 在 第一组 中对应的序号 ( 不存在时 j==-1 )
            if j<0:
                break
            idxOfImgs2.append(j)
        all6ImgEquals = False    
        caseAddOrSubEq = True 
        caseXorEq = True
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
            if self.getImages2(imgs2Name[0:2]).isImgElementsEquals() and self.getImages2(imgs2Name[1:3]).isImgElementsEquals():
                # GH? 图形相同, ABC 也相同 ( 因为 同组合 )
                score += 10
                desc = "两组图形6个全相同"
                #scoreAddTo.addScore(10 * scoreWeight ,imgs1Name,imgs2Name,"两组图形6个全相同")
                all6ImgEquals = True
            scoreAddTo.addScore(score * scoreWeight ,imgs1Name,imgs2Name,desc) # C-02
        elif    self.getImages2(imgs1Name[0:2]).isImgElementsEquals() and self.getImages2(imgs1Name[1:3]).isImgElementsEquals() \
            and self.getImages2(imgs2Name[0:2]).isImgElementsEquals() and self.getImages2(imgs2Name[1:3]).isImgElementsEquals():
            # ABC 相等, GHI 相等
            scoreAddTo.addScore(10 * scoreWeight ,imgs1Name,imgs2Name,"每组图形全相等")
        #
        # 判断是否两组图片 元素 个数 全相同 , 或 个数的变换规律相同
        #
        elementCountIncAB = len(imgsFrm1.img2Elements) - len(imgsFrm1.img1Elements)  # A-<B 图片元素的增加量
        elementCountIncBC = len(imgsFrm1.img3Elements) - len(imgsFrm1.img2Elements)  # B->C 图片元素的增加量
        elementCountIncGH = len(imgsFrm2.img2Elements) - len(imgsFrm2.img1Elements)  # A-<B 图片元素的增加量
        elementCountIncHI = len(imgsFrm2.img3Elements) - len(imgsFrm2.img2Elements)  
        nElementIfSame = -1 # 如果 六个 图形 具有 相同 元素 个数,  nElementIfSame 将 >0
        if elementCountIncAB==elementCountIncGH and elementCountIncBC==elementCountIncHI:
            score = 4   
            if elementCountIncAB==0 and elementCountIncBC==0:
                score += 1 
                desc = "两组图形元素个数相同"
                nElementIfSame =  len(imgsFrm1.img1Elements)   
            else:
                desc = "两组图形元素个数变化递增两相同"
            scoreAddTo.addScore(score * scoreWeight ,imgs1Name,imgs2Name,desc)
        elif len(imgsFrm1.img1Elements)>0 \
            and len(imgsFrm1.img2Elements)/len(imgsFrm1.img1Elements)==len(imgsFrm2.img2Elements)/len(imgsFrm2.img1Elements) \
            and   len(imgsFrm1.img3Elements)/len(imgsFrm1.img1Elements)==len(imgsFrm2.img3Elements)/len(imgsFrm2.img1Elements) : 
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
        
        if nElementIfSame>0 and not all6ImgEquals: # 六个 图形 具有 相同 元素 个数 (但不全等)
            #
            # 是否 匹配 旋转 属性:  
            # 例如  A (旋转90度)-> B (旋转90度)-> C
            #   且 G (旋转90度)-> H (旋转90度)-> I
            # 例子 Challenge D-04
            #     
            imgsAB = self.getImages2(imgs1Name[0:2])
            
            #print("%s : elementsCount = %d/%d " % (imgsAB.name,len(imgsAB.img1Elements),len(imgsAB.img2Elements)))
            
            for i in range(nElementIfSame):
                if imgsAB.isImgElementTransMatched(i,IMGTRANSMODE_ROTATE1) \
                  and self.getImages2(imgs1Name[1:3]).isImgElementTransMatched(i,IMGTRANSMODE_ROTATE1) \
                  and self.getImages2(imgs2Name[0:2]).isImgElementTransMatched(i,IMGTRANSMODE_ROTATE1) \
                  and self.getImages2(imgs2Name[1:3]).isImgElementTransMatched(i,IMGTRANSMODE_ROTATE1) :
                    scoreAddTo.addScore( 10* scoreWeight/nElementIfSame,imgs1Name,imgs2Name,"两组图形为90度旋转关系")
                    caseAddOrSubEq = False
                    caseXorEq = False
                    continue
                if imgsAB.isImgElementTransMatched(i,IMGTRANSMODE_ROTATE3) \
                  and self.getImages2(imgs1Name[1:3]).isImgElementTransMatched(i,IMGTRANSMODE_ROTATE3) \
                  and self.getImages2(imgs2Name[0:2]).isImgElementTransMatched(i,IMGTRANSMODE_ROTATE3) \
                  and self.getImages2(imgs2Name[1:3]).isImgElementTransMatched(i,IMGTRANSMODE_ROTATE3) :
                    scoreAddTo.addScore( 10* scoreWeight/nElementIfSame,imgs1Name,imgs2Name,"两组图形为-90度旋转关系")
                    caseAddOrSubEq = False
                    caseXorEq = False
                    continue
                imgsAC = self.getImages2(imgs1Name[0:1]+imgs1Name[2:3])
                if (imgsAC.isImgElementTransMatched(i,IMGTRANSMODE_ROTATE1) and self.getImages2(imgs2Name[0:1]+imgs2Name[2:3]).isImgElementTransMatched(i,IMGTRANSMODE_ROTATE1) )\
                       or (imgsAC.isImgElementTransMatched(i,IMGTRANSMODE_ROTATE3) and self.getImages2(imgs2Name[0:1]+imgs2Name[2:3]).isImgElementTransMatched(i,IMGTRANSMODE_ROTATE3)) :
                    caseAddOrSubEq = False
                    caseXorEq = False
                    if imgsAB.isBlackPixelRatioEquals(i) and self.getImages2(imgs2Name[0:2]).isBlackPixelRatioEquals(i) : # todo 需要判断 满足 45 度的旋转 ,暂时 使用 isBlackPixelRatioEquals 代替
                        scoreAddTo.addScore( 10* scoreWeight/nElementIfSame,imgs1Name,imgs2Name,"两组图形为45度旋转关系")
                    continue
                #elif imgsAB.isImgElementTransMatched(i,IMGTRANSMODE_ROTATE3) 
        # END if nElementIfSame>0: #六个 图形 具有 相同 元素 个数
        
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
            diffABC,_,totalPixcelABC = countImagesDiff([self.getImage(imgs1Name[0:1]),self.getImage(imgs1Name[1:2])],[self.getImage(imgs1Name[2:3])])  
            if diffABC/totalPixcelABC<=0.02:
                diffGHI,_,totalPixcelGHI = countImagesDiff([self.getImage(imgs2Name[0:1]),self.getImage(imgs2Name[1:2])],[self.getImage(imgs2Name[2:3])])  
                if diffGHI/totalPixcelGHI<=0.02:
                    scoreAddTo.addScore( 6 * scoreWeight,imgs1Name,imgs2Name,"前两图片像素合并==第三个图片")
                caseAddOrSubEq = False    
                caseXorEq = False    
            
        if caseAddOrSubEq: # 处理 图片A - 图片B == 图片C 的情况   
            diffCBA,_,totalPixcelCBA = countImagesDiff([self.getImage(imgs1Name[2:3]),self.getImage(imgs1Name[1:2])],[self.getImage(imgs1Name[0:1])])          
            if diffCBA/totalPixcelCBA<=0.02: #E-04
                diffIHG,_,totalPixcelIHG = countImagesDiff([self.getImage(imgs2Name[2:3]),self.getImage(imgs2Name[1:2])],[self.getImage(imgs2Name[0:1])])  
                if diffIHG/totalPixcelIHG<=0.02:
                    scoreAddTo.addScore( 6 * scoreWeight,imgs1Name,imgs2Name,"前两图片像素相减==第三个图片")
                caseAddOrSubEq = False    
                caseXorEq = False  
        
        
        # E-04 : 答案 2 与 8 都一样
        if caseAddOrSubEq: # 处理 图片A.像素个数 + 图片B.像素个数 == 图片C 的情况  E-04   
            c1 = imgsFrm1.compareImgPixelCount() 
            if c1>0 and c1==imgsFrm2.compareImgPixelCount() :        
                caseAddOrSubEq = False    
                caseXorEq = False 
                #  - E-04 : 答案 2 与 8 都一样 , 进一步考虑图片形状
                if  ( imgsFrm1.compareImgPixelHeight()==0 and imgsFrm2.compareImgPixelHeight()==0 and imgsFrm1.compareImgPixelWidth()==c1 and imgsFrm2.compareImgPixelWidth()==c1  ) \
                 or ( imgsFrm1.compareImgPixelWidth()==0 and imgsFrm2.compareImgPixelWidth()==0  and imgsFrm1.compareImgPixelHeight()==c1 and imgsFrm2.compareImgPixelHeight()==c1):
                    scoreAddTo.addScore( 6 * scoreWeight,imgs1Name,imgs2Name,"前两图片像素个数相加或减==第三个图片,且宽高匹配")
                else:
                    scoreAddTo.addScore( 1 * scoreWeight,imgs1Name,imgs2Name,"前两图片像素个数相加或减==第三个图片")
                    
        #
        # XOR 的例子: D-11
        # 
        if caseXorEq:
            pass # todo                
    # END method calculateImages3MatchScore 

################################################################
# solve_2x2
#############################################################
    def solve_2x2(self):
        answers = []
        for answer in self.potential_answers:
            answerScore = AnswerScore(int(answer))   
            self.calculateImages2MatchScore("AB","C"+answer,answerScore)  # 行比较 : 第一行 与 第 三 行
            self.calculateImages2MatchScore("AC","B"+answer,answerScore)
            #  self.calculateImages2MatchScore("BC","A"+answer,answerScore,0.5) # 对角线 ????
            if Agent._DEBUG:
                print("答案 [%d] 总得分 = %.2f , 其中:" % (answerScore.answer,answerScore.score))
                for scoreDetail in answerScore.scoreDetails:
                    print("  得分 %.2f 来自于: [%s-%s]%s" % scoreDetail)
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
        for answer in self.potential_answers:
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
                print("答案 [%d] 总得分 = %.2f , 其中:" % (answerScore.answer,answerScore.score))
                for scoreDetail in answerScore.scoreDetails:
                    print("  得分 %.2f 来自于: [%s-%s]%s" % scoreDetail)
            answers.append(answerScore)
        answers = AnswerScore.getMaxScoreAnswers(answers)
        return answers[0].answer

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
        answer = -1
        self.images, self.potential_answers = load_problem_images(problem)
        self.imagesEles = {}
        self.imagesFrame = {} 
        #print("\n")
        print("--->This Problem: ", problem.name)

        if problem.problemType == "2x2":
            # if problem.name[-4:] == 'B-04':
                # print("Debug B12:", problem.name[-4:])
            answer = self.solve_2x2()
        elif problem.problemType == "3x3":
            answer = self.solve_3x3()
            pass

        return answer

#
#####################################
#

# END class Agent    




