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



def calculate_delta(p1, p2):
    """
    Calculates the Euclidean distance between two images.

    Args:
        p1, p2: Two image matrices to compare.

    Returns:
        float: The calculated Euclidean distance between the two images.
    """
    distance = np.sqrt(np.sum(np.power(p1 - p2, 2)))
    return distance

def countDiff(image1,image2):
    count = 0
    height, width = image1.shape
    for y in range(height):
        for x in range(width):
            if (image1[y,x]==0) != (image2[y,x]==0):
                count += 1
    return count


###
#  类似于  np.mean(cv2.absdiff(image1,image2))
###
def countDiffRadio(image1,image2):
    height, width = image1.shape
    return float(countDiff(image1,image2)) / (height*width)

####
# 
####
def countPixcel(image):
    count = 0
    height, width = image.shape
    for y in range(height):
        for x in range(width):
            if (image[y,x]==0) :
                count += 1
    return count

def countPixcelRadio(image):
    height, width = image.shape
    return float(countPixcel(image)) / (height*width)

def find_best_match(delta_dict):
    """
    Finds the best matching answer based on the smallest delta value.

    Args:
        delta_dict: A dictionary of delta values for each answer key.

    Returns:
        int: The answer key corresponding to the correct image.
    """
    index = int(min(delta_dict, key=delta_dict.get))
    return index

#
#  y 行 最后一个 黑点
#
def endBlackPointX(image,y):
    height, width = image.shape
    for x in range(width-1,-1,-1):
        if image[y,x]==0:
             return x
    return -1     

#
# 将 imag 填充 变为 实心 图 , 例如 B-09.2 => B-09.1
# 
def fillImage(image):
    #image = _image.copy()
    height, width = image.shape
    for y in range(height):
        #
        # status==0 :  白色区域, 非填充 (第 1,3,5 ..次出现)
        #         1 :  左黑色区域 , (第 1,3,5 ..次出现)
        #         2 :  白色区域, 填充  (第 2,4,6 ..次出现)
        #         3 :  黑色区域  (第 2,4,6 ..次出现)
         
        status = 0
        for x in range(endBlackPointX(image,y)):
            v = image[y,x]
            if status==0: #  白色区域, 非填充
                if v==0 :
                    status = 1 
            elif status==1: 
                if v!=0 :
                    image[y,x] = 0
                    status = 2  
            elif status==2:
                if v==0 :
                   status = 3
                else:
                   image[y,x] = 0       
            elif status==3: #黑色区域  (第 2,4,6 ..次出现)
                if v!=0 :  
                    status = 0       
                    
def cloneFillImage(image):
    filledImage = image.copy()
    fillImage(filledImage)
    return filledImage           

#
#  从图像 image1 中 去除掉 image2 同位置 图形的部分, 例如
#   "B-10 中的 C.png"  减去  "B-10 中的 A.png"  得到一个小放过
#
def  imageSubstract(image1,image2):
    image = image1.copy()
    height, width = image.shape
    for y in range(height):
        for x in range(width):
            if image2[y,x]==0:
                image[y,x] = 255
    return image       

#
# 将 两个 图像 合并
#
def  imageAdd(image1,image2):
    return cv2.bitwise_and(image1,image2,mask=None)

#
# 图片的相连区域
# 
class ImageElement:
    def __init__(self,image,x0,y0):
        self.image = image
        self.x0 = x0
        self.y0 = y0
        self.ex = x0+1
        self.ey = y0+1
    def addPixcel(self,x,y):
        self.image[y,x] = 0
        if self.ex < x+1:
            self.ex = x+1
        if self.ey < y+1:
            self.ey = y+1
        if self.x0 > x:
            self.x0 = x
        if self.y0 > y:
            self.y0 = y
    # 返回 图片元素的 高, 宽
    def  getSize(self):
        return ((self.ey-self.y0),(self.ex-self.x0))
    #
    # 将 图片元素重新合并 后的 图片
    #



def newImageElement(image,x0,y0,flagAdded):
    #height, width = image.shape
    newImage = np.full(image.shape, 255, np.uint8)
    newImageEle = ImageElement(newImage,x0,y0)
    newImage[y0,x0] = 0    
    flagAdded[y0,x0] = True
    #print("newImageElement - %d,%d"%(x0,y0))
    checkPoints = [(y0,x0)]
    while checkPoints:
        y,x = checkPoints.pop()
        # 检查 周围点
        for yi in range(y-1,y+2,1):
            for xi in range(x-1,x+2,1):
                if not flagAdded[yi,xi] and image[yi,xi]==0:
                    #print(" ... xi=%d,yi=%d",xi,yi)
                    flagAdded[yi,xi] = True
                    newImageEle.addPixcel(xi,yi)
                    checkPoints.append((yi,xi))
    return newImageEle      


 #
 # 将图片(按像素相连)分隔成多个元素, 相连的按像 分在一个元素组中
 #
def splitImage(image):
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
                    part = newImageElement(image,x,y,flagAdded)
                    imageParts.append(part)
                else:
                    flagAdded[y,x] = True     
                
    return imageParts        

#
# 判断 两图片元素是否相等(或相似)
#
def  isImageElementEquals(imageElement1,imageElement2,threadhold):
    h1,w1 = imageElement1.getSize()
    h2,w2 = imageElement2.getSize()
    #
    # 元素的 大小 需要相等, 允许 +/- 2 的误差
    #
    if h1<h2-2 or h1>h2+2 or w1<w2-2 or  w1>w2+2:
        return False,-1
    x1 = imageElement1.x0
    y1 = imageElement1.y0
    x2 = imageElement2.x0
    y2 = imageElement2.y0
    #
    # 坐标位置 需要相等, 允许 +/- 3 的误差
    #
    if y1<y2-3 or y1>y2+3 or x1<x2-3 or  x1>x2+3:
        return False,-1
    h = h1 if h1>h2 else h2
    w = w1 if w1>w2 else w2
    img1 = imageElement1.image
    img2 = imageElement2.image
    countDiff = 0
    for yi in range(h):
        for xi in range(w):
            if (img1[y1+yi,x1+xi]==0) != (img2[y2+yi,x2+xi]==0):
                countDiff += 1
    r =  float(countDiff) / (w*h)
    return r<=threadhold,r

def indexOfImageElement(imageElement,inElements,threadhold):
    index = 0
    for e in inElements:
        #eq,diffR = isImageElementEquals(imageElement,e,threadhold)
        eq,_ = isImageElementEquals(imageElement,e,threadhold)
        #print("%d: eq=%s diffR=%f " % (index,eq,diffR))
        if eq  :
            return index
        index += 1
    return -1    

def  mergeImageElementsAsImage(imgElements):
    if len(imgElements)==1 :
        return imgElements[0].image
    if len(imgElements)==0 :
        return None
    image = cv2.bitwise_and(imgElements[0].image,imgElements[1].image,mask=None)
    if len(imgElements)==2 :
        return image
    for imgEle in  imgElements[2:]:
       image = cv2.bitwise_and(image,imgEle.image,mask=None) 
    return  image               

#
#  从元素组 中 减去 一个元素, 再将剩下的元素 合并为一个 新 图片
#
def  getImageAfterRemoveImageElement(imgElements,removeElement):
    iRemove = indexOfImageElement(removeElement,imgElements,SAMEIMG_THRESHOLD)
    if iRemove<0 :
        return None,False
    newElements = []
    for i in range(len(imgElements)):
        if i!=iRemove:
            newElements.append(imgElements[i])
    return mergeImageElementsAsImage(newElements),True        

#
# 将一个 图片元素 添加到 元素组, 并 合并为一个 新 图片
#
def  getImageAfterAddImageElement(imgElements,addElement):
    newElements = []
    for i in range(len(imgElements)):
        newElements.append(imgElements[i])
    newElements.append(addElement)    
    return mergeImageElementsAsImage(newElements)        

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

    """
      return {
         "1" :  1.png - ?.png
         "2" : 2/pmg - ?.png
       }
    """
    def fill_delta_dict(self, base_image):
        """
        Helper function to calculate deltas for all potential answers against a base image.

        Args:
            base_image: The base image to compare against potential answers.

        Returns:
            delta_dict: A dictionary with delta values for each potential answer.
        """
        delta_dict = {}
        for key, img in self.potential_answers.items():
            delta_dict[key] = calculate_delta(base_image, img)
        return delta_dict
    
    #
    # @param imageId image id,  如 "A", "B", "C", "1", "2" 等
    #
    def  getImageElements(self,imageId):
        imgElemets = self.imagesEles.get(imageId)
        if imgElemets!=None :
            return imgElemets
        if imageId>="1" and imageId<="9":
            image =  self.potential_answers[imageId]
        else:
            image =  self.images[imageId]    
        imgElemets = splitImage(image)
        self.imagesEles[imageId] = imgElemets
        return imgElemets
    
    #
    # 获取 imageId1 - imageId2 后的图片
    # 例如: B-10 问题中 :
    #    图片 "C"-"A" 返回结果为 一个实心方块
    # 如果 imageId2 的 元素个数 >= imageId1 的 元素个数 返回 None
    # @return 返回 相减 后的 图片元素
    #
    def imageElementsSubtract(self,imageId1,imageId2):
        imgElements1 = self.getImageElements(imageId1)
        imgElements2 = self.getImageElements(imageId2)
        if len(imgElements1) != len(imgElements2)+1 :
            return None
        indexSubed = []
        for i in range(len(imgElements1)):
            indexSubed.append(False)
        for imgElement2 in imgElements2:
            i = indexOfImageElement(imgElement2,imgElements1,SAMEIMG_THRESHOLD)
            if i<0:
                return None  
            indexSubed[i] = True      
        for i in range(len(imgElements1)):
            if not indexSubed[i]:
                return  imgElements1[i]   
        return None
    #
    # 从可选答案中 选 与 image 最匹配的 图片:
    #  返回 : 最匹配图 及 最匹配图 与 image 的 差值
    #
    def findBestMatchAnswerImage2(self,image):
        #if image==None:
        #    return -1, 1000.0
        if isinstance(image,str):
            image = self.images[image]
        bestMatch = -1
        bestMatchDelta = 0
        for key, img in self.potential_answers.items():
            #delta = calculate_delta(image,img)
            #diffRadio = countDiffRadio(image,img)
            """
              ??? 使用 countDiffRadio 更合理
               发现了有些情况下, countDiffRadio calculate_delta 的结果 不是同一个趋势:
                 B-10 中 C-A+B 的结果图片 查找答案时, 
                  与答案 2 比较 :  calculate_delta==255.663842 , countDiffRadio==0.128574
                  与答案 3 比较 :  calculate_delta==379.743334 , countDiffRadio==0.035799
                答案 3 与   C-A+B 应该更相似,  但使用 calculate_delta 方式判断时, 结果 是 2
                countDiffRadio 类似于 np.mean(cv2.absdiff(image1,image2))
            """
            delta = countDiffRadio(image,img)
            #if Agent._DEBUG:
            #    print("  ...Image %s : delta=%f, 图形 欧氏距离=%f" % (key,delta,calculate_delta(image,img)))
            if bestMatch==-1 or bestMatchDelta>delta:
                bestMatch = int(key)
                bestMatchDelta = delta
                #print("Image %s = %d" % (key,bestMatchDelta))
        return  bestMatch,bestMatchDelta       

    #
    # 从可选答案中 选 与 image 最匹配的 图片:
    #        
    def findBestMatchAnswerImage(self,image):
        return self.findBestMatchAnswerImage2(image)[0]
    

    #
    # 从可选答案中 选 与 image 最匹配的 图片:
    # 当 threshold>=0 时, 如果所有答案 与 image 的 值 >threshold
    # 返回 -1
    #        
    def tryFindBestMatchAnswerImage(self,image,threshold):
        bestMatch,matchedDelta = self.findBestMatchAnswerImage2(image)
        if threshold>=0 and matchedDelta>threshold:
            return -1
        return bestMatch

    """
          A 的反转比较 : 考虑四种可能
            (1) 图片A 垂直翻转(上下)  与  图片B 比较   如果相似, 则 将 C 按同样方法翻转后获取答案
            (2) 图片A 垂直翻转(上下)  与  图片C 比较
            (3) 图片A 水平翻转 ( 左右 ) 与  图片B 比较
            (3) 图片A 水平翻转 ( 左右 ) 与  图片C 比较
         B-04, B-05 等问题  使用到该方法求解  
    """
    def try_solve_2x2_byFlip(self):
        SAMEIMG_THRESHOLD_Flip = 0.03
        FLIPANSERT_THRESHOLD = 0.1
        ansert = -1
        ansertDelta = 100.0
        ansertByFlip = 0
        ansertFromImg = "?"
        ansertCmpImg = "?"
        for flipMode in [0,1]:
            flipped_A = cv2.flip(self.images["A"],flipMode ) 
            #
            # 翻转 后 图不能相同 ( B-11,B-12)
            #
            diffRadio = countDiffRadio(flipped_A, self.images["A"])
            if  diffRadio<SAMEIMG_THRESHOLD:
                continue
            for compareTo in ["B","C"]:
                # A 的 反转 与 B 或 C 比较
                diffRadio = countDiffRadio(flipped_A, self.images[compareTo]) 
                #if Agent._DEBUG:
                #    print("[try_solve_2x2_byFlip]A %s 的图片 与 %s : diffRadio = %f" % ("垂直翻转(上下)" if flipMode==0 else "A水平翻转(左右)", compareTo, diffRadio ))
                if diffRadio<SAMEIMG_THRESHOLD_Flip : #SAMEIMG_THRESHOLD:
                    ansertImg = "C" if compareTo=="B" else "B"
                    a,d = self.findBestMatchAnswerImage2(cv2.flip(self.images[ansertImg], flipMode)) 
                    #if Agent._DEBUG:
                    #    print("   ...  根据 %s 翻转后 查询得到 结果 %d (delta=%f)  " %(ansertImg,a,d))
                    if d<FLIPANSERT_THRESHOLD and d<ansertDelta:
                        ansertByFlip = flipMode
                        ansertFromImg = ansertImg
                        ansertCmpImg = compareTo
                        ansert = a
                        ansertDelta = d
        if Agent._DEBUG and ansert>0:
            print("[try_solve_2x2_byFlip]A %s 的图片 与 %s 相似, 使用 %s 的翻转图查结果" % ("垂直翻转(上下)" if ansertByFlip==0 else "A水平翻转(左右)",ansertCmpImg,ansertFromImg ))        
        return  ansert
            

    #
    # 求解 类似 B-09 之类的问题
    #  算法说明:
    #    将 图片A 转换为 实心图, 与 B 或 C 比较 , 如果 相似, 则 使用 另一图片( C 或 B) 变为 实心图 求解
    #  ( 例如 B-09 的 图片 A, 变为 实心图 ,与 B 相似 )  
    #
    def try_solve_2x2_byFilled(self):
        THRESHOLD_FOR_FILLEDSOLVE = 0.05  # 5% 的误差
        filledA = cloneFillImage(self.images["A"])
        # 判断 A 填充后的图形 是否与 B 相似
        diffRadio = countDiffRadio(filledA, self.images["B"])
        #if Agent._DEBUG:
        #    print("[try_solve_2x2_byFilled]: filledA 与 B 的 diffRadio=%f " %(diffRadio))
        if diffRadio < THRESHOLD_FOR_FILLEDSOLVE:
            if Agent._DEBUG:
                print("B 图 与 A的填充图相似(diffRadio=%f), 使用 C 的填充图查结果 "%diffRadio)
            return  self.findBestMatchAnswerImage(cloneFillImage(self.images["C"]))
        #diffImg = cv2.absdiff(filledA, self.images["C"])
        #avgDiff = np.mean(diffImg) 
        # 判断 A 填充后的图形 是否与 C 相似
        diffRadio = countDiffRadio(filledA, self.images["C"])
        if Agent._DEBUG:
            print("[try_solve_2x2_byFilled]: filledA 与 C 的 diffRadio=%f " %(diffRadio))
        if diffRadio < THRESHOLD_FOR_FILLEDSOLVE:
            print("C 图 与 A的填充图相似(diffRadio=%f), 使用 B 的填充图查结果 "%diffRadio)
            return  self.findBestMatchAnswerImage(cloneFillImage(self.images["B"]))
        ###
        # ??? todo B 或 C 填充后 与 A  相似的情况下,  求 D
        ###
        return -1
    

    #
    # 使用  image1Id +/- ( image2Id - image3Id ) 获取到的图片 求解
    # @param op1 "+" or "-"
    # 
    def try_solve_byImgElementChange1(self,image1Id,op1,image2Id,image3Id):
        imgEle = self.imageElementsSubtract(image2Id,image3Id) 
        if imgEle==None:
            return -1,0
        if imgEle!=None:
            if op1=="-":
                image,_ok = getImageAfterRemoveImageElement(self.getImageElements(image1Id),imgEle)
            elif  op1=="+":
                image,_ok = getImageAfterAddImageElement(self.getImageElements(image1Id),imgEle),True
            else:
                 return -1,0   
            if _ok:
                answer,delta = self.findBestMatchAnswerImage2(image)
                if answer>=0 :
                    if Agent._DEBUG:
                        print("通过 图形元素增减规律: %s %s (%s-%s) 求解为 %d( delta=%f) "%(image1Id,op1,image2Id,image3Id,answer,delta))
                    return answer,delta
        return -1        

    #######
    #   根据 图形元素 增减 求解
    #   B-10 : B +(C-A)
    #   B-11 : C-(A-B)
    #   case-1 : C - (A-B)    
    #   case-2 : B - (A-C)
    #   case-3 : C + (B-A)
    #   case-4 : B + (C-A)
    # 算法说明:( 以 情况-1 为例 )
    #    如果 图片 A 与 B 之间 只相差 一个 元素, 则认为 答案 与 C 之间 也可能 只相差 一个相同的 元素  
    #      将 图片 C 加上 或减去 同样元素 来查找 答案 
    #  实现说明:
    #   通过 将 A 与 B (或 A与 C), 通过方法 splitImage 将图片分解为多个元素, 然后比较这些元素 
    #    得到A 与 B (或 A与 C) 之间的元素增减变化规律, 再将 另一图片 按相同规律 增减同一元素求解
    #    
    #
    ###########
    def try_solve_2x2_byImgElementChange(self):
        solveByCase  = ""

        # CASE1 : C - (A-B)    
        answer,delta = self.try_solve_byImgElementChange1("C","-","A","B")
        if answer>0:
            return answer
        # CASE1 : C - (A-B)    
        answer,delta = self.try_solve_byImgElementChange1("B","-","A","C")
        if answer>0:
            return answer
        # CASE1 : C - (A-B)    
        answer,delta = self.try_solve_byImgElementChange1("C","+","B","A")
        if answer>0:
            return answer
        # CASE1 : C - (A-B)    
        answer,delta = self.try_solve_byImgElementChange1("B","+","C","A")
        if answer>0:
            return answer
        return -1
   
        
    def solve_2x2(self):
        """
        Solves the 2x2 Raven's Progressive Matrices problem by comparing the similarity
        between images A, B, C, and the potential answers.

        Returns:
            int: The answer key corresponding to the correct image.
        """
        delta_dict = {}
        


        # Calculate the absolute difference between image pairs

        """
        delta_AB = cv2.absdiff(self.images["A"], self.images["B"])
        delta_AC = cv2.absdiff(self.images["A"], self.images["C"])
        
        if not delta_AB.any() and not delta_AC.any():
            print("A, B, C are the same")
            #delta_dict = self.fill_delta_dict(self.images["A"])
            return findBestMatchAnswerImage("A")

        if not delta_AB.any():
            print("A, B are the same")
            #delta_dict = self.fill_delta_dict(self.images["C"])
            return self.findBestMatchAnswerImage("C")

        if not delta_AC.any():
            print("A, C are the same")
            #delta_dict = self.fill_delta_dict(self.images["B"])
            return self.findBestMatchAnswerImage("B") 
        """    
        
        """
          两个图形 的相等判断 允许 存在 一点误差,
           由于 delta_AB.any() 的 判断 , 只要有一个像素 不等, 都被认为不等
           例如 B-08 中的 A, C 两个图形, 有稍小的 偏差
            ( ? 可以考虑 偏移较少像素 后 再判断 相等 ??? )
        """
        diffRadioAB = countDiffRadio(self.images["A"], self.images["B"]) 
        if  diffRadioAB<SAMEIMG_THRESHOLD :
            if Agent._DEBUG:
                print("A, B 相似(delta=%f), 使用 C 找 答案 ..." % diffRadioAB)
            return self.findBestMatchAnswerImage("C")
        diffRadioAC = countDiffRadio(self.images["A"], self.images["C"]) 
        if  diffRadioAC<SAMEIMG_THRESHOLD :
            if Agent._DEBUG:
                print("A, C 相似(delta=%f), 使用 B 找 答案 ..."% diffRadioAC)
            return self.findBestMatchAnswerImage("B")
        
        # Handle cases where images are not identical
        #print("A, B,C images are not identical (全不相同)...")

        """
          A 的反转比较 : 考虑四种可能
            (1)A垂直翻转(上下)  与  B 比较
            (2)A垂直翻转(上下)  与  C 比较
            (3)A水平翻转 ( 左右 ) 与  B 比较
            (3)A水平翻转 ( 左右 ) 与  C 比较
        """
        answerByFlip = self.try_solve_2x2_byFlip()
        if  answerByFlip>0:
            return answerByFlip

        """
           根据 A 的填充后图形 求解
        """        
        answerByFilledSolve = self.try_solve_2x2_byFilled()
        if answerByFilledSolve>0:
            return answerByFilledSolve
        
        """
           根据 图像元素 增减 规律 求解
            (!!!注意 : 在每次求解装载图片时, 需要清除 self.imagesEles, 及设置一下 self.imagesEles = {}, )
        """
        ansertBySubstract = self.try_solve_2x2_byImgElementChange()
        if ansertBySubstract>0 :
            return ansertBySubstract


        # If no match found, compare with both B and C
        delta_AB_to_answer,deltaB = self.findBestMatchAnswerImage2("B")
        delta_AC_to_answer,deltaC = self.findBestMatchAnswerImage2("C")
        if deltaB<deltaC :
            print(" 结果来自 B ... ")
            return delta_AB_to_answer 
        print(" 结果来自 C ... ")
        return delta_AC_to_answer
        """
        for key, img in self.potential_answers.items():
            delta_AB_to_answer = calculate_delta(self.images["B"], img)
            delta_AC_to_answer = calculate_delta(self.images["C"], img)
            delta_dict[key] = min(delta_AB_to_answer, delta_AC_to_answer)
    
        return find_best_match(delta_dict)
        """

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
        #print("\n")
        print("--->This Problem: ", problem.name)

        if problem.problemType == "2x2":
            # if problem.name[-4:] == 'B-04':
                # print("Debug B12:", problem.name[-4:])
                answer = self.solve_2x2()
        elif problem.problemType == "3x3":
            pass

        return answer


"""
 2024-09-24: 代码说明
  (1) 两个图片是否相等的比较, 原来 delta_AC.any() 的方式, 求解 B-08 时, 发现A,C 有稍稍的误差, 这个代码里使用 countDiffRadio 比较, 阈值 SAMEIMG_THRESHOLD ( 代码 628 行附近 )
  (2) 反转图片的比较 , 原来两种方式, 改成了 四中情况, 具体参考 代码 455行附近 方法 try_solve_2x2_byFlip 
  (3) 增加了 从非实心图 变为 实心图 ( 如 B-09 ),具体参考 代码 500行附近 方法 try_solve_2x2_byFilled
  (4) 增加了按 图像元素 增减 规律 求解的 方法, 具体参考 代码 555 行附近 方法 try_solve_2x2_byImgElementChange
"""