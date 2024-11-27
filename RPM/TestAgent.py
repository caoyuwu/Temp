
#
# python D:\snsoftn10\snadk-srcx\python-projects\RPM-Project-Code\TestAgent.py
# python3 /snsoftn10/snadk-srcx/python-projects/RPM-Project-Code/TestAgent.py
#

import random
import re
import os
import json
import cv2
from datetime import datetime
from time import time
from CV2Utils import CV2Utils 
#import json
from Agent import Agent,APPPATH 
#,countImagesDiff,countImagesXOR
from Agent import ImageElement,Image1
from Agent import countImageDiffRatio,countImageDiff
from Agent2 import Agent2
from RavensFigure import RavensFigure
from RavensProblem import RavensProblem
from RavensObject import RavensObject
from ProblemSet import ProblemSet
def getNextLine( r):
    line = r.readline().rstrip()
    while  line.strip().startswith("#"):
        line = r.readline().rstrip()
    return line    
    
def loadProblem(setName ,problemName):
    #if len(problemName)==2:
    #    problemName = setName+"-"+problemName
    data_filename =  os.path.join(APPPATH,"Problems",setName,problemName,"ProblemData.txt")
    #data_filename = "Problems" + os.sep + setName + os.sep + problemName + os.sep + "ProblemData2.txt"

    with open(data_filename) as r:
        problemType=getNextLine(r)

        hasVisual=getNextLine(r)=="true"
        hasVerbal=getNextLine(r)=="true"

        newProblem=RavensProblem(problemName, problemType, setName, hasVisual, hasVerbal)
        if newProblem.hasVerbal:
            figures=[]
            currentFigure=None
            currentObject=None

            line = getNextLine(r)
            while not line=="":
                if not line.startswith("\t"):
                    newFigure=RavensFigure(line, problemName, setName)
                    figures.append(newFigure)
                    currentFigure=newFigure
                elif not line.startswith("\t\t"):
                    line=line.replace("\t","")
                    newObject=RavensObject(line)
                    currentFigure.objects[line]=newObject
                    currentObject=newObject
                elif line.startswith("\t\t"):
                    line=line.replace("\t","")
                    split=re.split(":",line)
                    currentObject.attributes[split[0]]=split[1]
                line=getNextLine(r)
            for figure in figures:
                newProblem.figures[figure.name]=figure
        else:
            newProblem.figures["A"]=RavensFigure("A", problemName, setName)
            newProblem.figures["B"]=RavensFigure("B", problemName, setName)
            newProblem.figures["C"]=RavensFigure("C", problemName, setName)
            newProblem.figures["1"]=RavensFigure("1", problemName, setName)
            newProblem.figures["2"]=RavensFigure("2", problemName, setName)
            newProblem.figures["3"]=RavensFigure("3", problemName, setName)
            newProblem.figures["4"]=RavensFigure("4", problemName, setName)
            newProblem.figures["5"]=RavensFigure("5", problemName, setName)
            newProblem.figures["6"]=RavensFigure("6", problemName, setName)
            if newProblem.problemType=="3x3":
                newProblem.figures["D"]=RavensFigure("D", problemName, setName)
                newProblem.figures["E"]=RavensFigure("E", problemName, setName)
                newProblem.figures["F"]=RavensFigure("F", problemName, setName)
                newProblem.figures["G"]=RavensFigure("G", problemName, setName)
                newProblem.figures["H"]=RavensFigure("H", problemName, setName)
                newProblem.figures["7"]=RavensFigure("7", problemName, setName)
                newProblem.figures["8"]=RavensFigure("8", problemName, setName)
                
        img9 = RavensFigure("9", problemName, setName)    
        if os.path.exists(img9.visualFilename):    
            newProblem.figures["9"] = img9  # 临时 调试使用      
        answer_filename = os.path.join(APPPATH,"Problems",setName,problemName, "ProblemAnswer.txt" )  
        with open(answer_filename) as r:
            newProblem.correctAnswer = int(getNextLine(r))        
        return newProblem
        
def loadBasicProblemByID(problemId):
    if len(problemId)!=4:
        print("Error -- ",problemId)
        return None
    setName = "Basic Problems "+problemId[0]
    #problemName = "Basic Problem B-01"
    problemName = "Basic Problem "+problemId
    return  loadProblem(setName,problemName)

def loadChallengeProblemByID(problemId):
    if len(problemId)!=4:
        print("Error -- ",problemId)
        return None
    setName = "Challenge Problems "+problemId[0]
    #problemName = "Basic Problem B-01"
    problemName = "Challenge Problem "+problemId
    return  loadProblem(setName,problemName)

def loadProblemByID(problemId):
    if problemId.startswith("Challenge "):
        problemId = problemId[10:]
        return  loadChallengeProblemByID(problemId)    
    else:
        return loadBasicProblemByID(problemId)

#
#  setName : "Basic Problems B"
#            "Basic Problems C"
#
def loadProblemSet(setName:str)->ProblemSet:
    return ProblemSet(setName)

def prepareAgent(problemId) ->Agent:
    agent = Agent()
    agent.prepareProblem(loadProblemByID(problemId))
    return agent

def prepareAgent4Challenge(problemId):
    agent = Agent()
    problem =  loadChallengeProblemByID(problemId)
    agent.prepareProblem(problem)
    return agent

def printImageElement(imgElement,name):
    if imgElement==None:
        print("空图")
        return
    height,width = imgElement.image.shape
    wholeX0,wholeY0  = width/2,height/2
    print("[元素(%s - %s)] : 元素区间(%d,%d) - (%d,%d) 相对整图中心点偏移(%d,%d),像素个数=%d/%d( 占比=%f) :" %(name,imgElement.name,imgElement.x0,imgElement.y0,imgElement.ex,imgElement.ey,(imgElement.x0+imgElement.ex)/2-wholeX0,(imgElement.y0+imgElement.ey)/2-wholeY0,imgElement.blackPixelCount,imgElement.getTotalPixel(),imgElement.getBlackPixelRatio()))
    flipedCenterPoint0X,flipedCenterPoint0Y =  imgElement.getWholeFlipedCenterPoint(0)
    print("整个上下反转后的中心点=%d,%d"%(flipedCenterPoint0X-wholeX0,flipedCenterPoint0Y-wholeY0))
    flipedCenterPoint1X,flipedCenterPoint1Y =  imgElement.getWholeFlipedCenterPoint(1)
    print("整个左右反转后的中心点=%d,%d"%(flipedCenterPoint1X-wholeX0,flipedCenterPoint1Y-wholeY0))
    CV2Utils.printImage2(imgElement.image)
 
def getImagesX(agent,imgsId:str):
    if len(imgsId)==1:
        return agent.getImage1(imgsId)
    if len(imgsId)==2:
        return agent.getImages2(imgsId)
    if len(imgsId)==3:
        return agent.getImages3(imgsId)
    raise BaseException("imgsId = ",imgsId) 
    
#
# showSplitImages("B-12","A")
#
def  showSplitImages(problemId,imgId):
    agent = prepareAgent(problemId)
    imgs = agent.getImageElements(imgId)
    i = 0
    for e in imgs:
        printImageElement(e,problemId)
        i += 1

def testHLineSegments(problemId,imgId):
    agent = prepareAgent(problemId)
    imgs = agent.getImageElements(imgId)
    img = imgs[0]
    for y in range(img.image.shape[0]):
        s = "%d: " % y
        selments = img.getHLineSegments(y)
        for seg in selments:
            s += " (%d-%d)" % seg
        print(s)  
def testHLineSegments2(problemId,imgId):
    agent = prepareAgent(problemId)
    imgs = agent.getImageElements(imgId)
    img = imgs[0]
    hlineSegs = img.getAllHLineSegments()
    y = img.y0
    for selments in hlineSegs:
        s = "%d: " % y
        for seg in selments:
            s += " (%d-%d)" % seg
        print(s)         
        y += 1   
    img.getAllHLineSegments()    

def testVLineSegments(problemId,imgId):
    agent = prepareAgent(problemId)
    imgs = agent.getImageElements(imgId)
    img = imgs[0]
    for x in range(img.image.shape[1]):
        s = "%d: " % x
        selments = img.getVLineSegments(x)
        for seg in selments:
            s += " (%d-%d)" % seg
        print(s)    
        
    print(img.getVLineSegments(90))    
    print(img.getHLineSegments(90))    
    print(img.getHLineSegments(90))    
        
def testVLineSegments2(problemId,imgId):
    agent = prepareAgent(problemId)
    imgs = agent.getImageElements(imgId)
    img = imgs[0]
    vlineSegs = img.getAllVLineSegments()
    x = img.x0
    for selments in vlineSegs:
        s = "%d: " % x
        for seg in selments:
            s += " (%d-%d)" % seg
        print(s)         
        x += 1   
    img.getAllVLineSegments()           

def testImageFilledRatio(problemId,imgId):
    agent = prepareAgent(problemId)
    #e = agent.imageElementsSubtract("C","B")
    #CV2Utils.printImage2(e.image)
    #answer = agent.try_solve_2x2_byImgElementChange()
    v,v1,v2,v3 = getImageFilledRatio(agent.images[imgId])
    print("v=",v,v1,v2,v3)
    #v,v1,v2,v3 = getImageFilledRatio(agent.images["B"])
    #print("v=",v,v1,v2,v3)
    imgA1 = agent.getImageElements("C")[0]
    v,v1,v2,v3 = imgA1.getImageFilledRatio()
    print("v=",v,v1,v2,v3)
    v,v1,v2,v3 = imgA1.getImageFilledRatio()
    print("v=",v,v1,v2,v3)
    h,w = imgA1.getSize()
    print("size = (%d,%d)"% (w,h))
    #for y in range(agent.images["A"].shape[0]):
    #    print("%d   : %d - %d" % (y,imgA1.getStartPointX(y),imgA1.getEndPointX(y)))        

def test_countImagesDiff(problemId,imgs1Id:str,imgs2Id:str):
    agent = prepareAgent(problemId)
    """
    imgs1 = []
    imgs2 = []
    for imgId in imgs1Id:
         imgs1.append(agent.getImage(imgId))
    for imgId in imgs2Id:
         imgs2.append(agent.getImage(imgId))
    diff,blackCount,size = countImagesDiff(imgs1,imgs2)     
    print("[%s] %s 与 %s 前景像素 相差 %d/%d/%d" %(problemId,imgs1Id,imgs2Id,diff,blackCount,size))
    """
    imgs1 = []
    imgs2 = []
    for imgId in imgs1Id:
         imgs1.append(agent.getImage1(imgId))
    for imgId in imgs2Id:
         imgs2.append(agent.getImage1(imgId))
    diff,blackCount,size = Image1.countImagesDiff(imgs1,imgs2) 
    print("[%s] %s 与 %s 前景像素 相差 %d/%d/%d" %(problemId,imgs1Id,imgs2Id,diff,blackCount,size))
    #diff,blackCount,size = Image1.countImagesDiff(imgs1,imgs2) 
    #print("[%s] %s 与 %s 前景像素 相差 %d/%d/%d" %(problemId,imgs1Id,imgs2Id,diff,blackCount,size))
    
def test_countImagesXOR(problemId,imgsId:str):
    agent = prepareAgent(problemId)
    imgs = []
    for imgId in imgsId:
         imgs.append(agent.getImage(imgId))
    diff,blackCount,size,img = countImagesXOR(imgs)     
    print("[%s] %s 异或后 前景像素 相差 %d/%d/%d" %(problemId,imgsId,diff,blackCount,size))   
    CV2Utils.printImage2(img) 

def testFlipImage(problemId,imgId,elementdx=0):
    agent = prepareAgent(problemId)
    img0 = agent.getImageElements(imgId)[elementdx]
    print("原图: ")
    printImageElement(img0,problemId)
    print("上下翻转: ")
    printImageElement(img0.getFlipedImage(0),problemId)
    print("左右翻转: ")
    printImageElement(img0.getFlipedImage(1),problemId)
    print("上下左右翻转: ")
    printImageElement(img0.getFlipedImage(-1),problemId)

    print("以整图为中心上下翻转: ")
    printImageElement(img0.getWholeFlipedImage(0),problemId)
    print("以整图为中心左右翻转: ")
    printImageElement(img0.getWholeFlipedImage(1),problemId)
    #printImageElement(img0.getFlipedImage(-1),problemId)

def testRotateImage(problemId:str,imgId,elementdx=0):
    agent = prepareAgent(problemId)
    img0 = agent.getImageElements(imgId)[elementdx]
    print("原图: ")
    printImageElement(img0,problemId)
    print("旋转 90 ")
    printImageElement(img0.getRotateImage(1),problemId)  
    print("旋转 180 ")  
    printImageElement(img0.getRotateImage(2),problemId)   
    print("旋转 270 ") 
    printImageElement(img0.getRotateImage(3),problemId)    
    
def testFilledImage(problemId,imgId,elementdx=0):    
    agent = prepareAgent(problemId)
    img0 = agent.getImageElements(imgId)[elementdx]
    print("原图: ")
    printImageElement(img0,problemId)
    print("填充图片:")
    printImageElement(img0.getFilledImage(),problemId)  
    img0.getFilledImage()
    
def testImageTransInfo(problemId,imsgName): 
    agent = prepareAgent(problemId)
    #AC = agent.getImageTransInfo(srcImgId,dstImgId)
    #print("AC = ",AC)    
    img2 = agent.getImages2(imsgName) # ImageTransformInfo
    transModeLst = ["EQUALS","FLIPV","FLIPH","FILLED","UNFILLED","ROTATE90","ROTATE180","ROTATE270"]
    for i in range(img2.getImgElementCount()):
        transInfo = img2.getAllImgElementTrans(i,transModeLst)  # ImageElementTrans[]
        for transVal in transInfo:  
            #if transVal.matched or transVal.matched2:
            print("[%s - %s]元素-%d: 变换=%s 相似度=%f,%f : 匹配度=%s %s %s  大小比例=%f " % (problemId,imsgName,i,transVal.transMode,transVal.similar,transVal.similar2,transVal.matched,transVal.matched2,transVal.matched3,transVal.scale))
        """
        t,v1,v2 = transInfo.getImgElementTrans(i,"FILLED") 
        print("t=%s %f %f" %(t,v1,v2)) 
        t,v1,v2 = transInfo.getImgElementTrans(i,"UNFILLED") 
        print("t=%s %f %f" %(t,v1,v2))  
        t,v1,v2 = transInfo.getImgElementTrans(i,"FILLED") 
        print("t=%s %f %f" %(t,v1,v2))  
        t,v1,v2 = transInfo.getImgElementTrans(i) 
        print("t=%s %f %f" %(t,v1,v2))  
        """

def getImageElement(agent:Agent,imgId:str,elementdx:int):
    if len(imgId)>1 and imgId[1]=="-":
        img = agent.getImageElements(imgId[0])[elementdx]
        return img.getTransImage(imgId[2:])
    return  agent.getImageElements(imgId)[elementdx]

def testImageElementSimilarScale(problemId,imgId1,imgId2,elementdx1=0,elementdx2=0):
    agent = prepareAgent(problemId)    

    img1 = getImageElement(agent,imgId1,elementdx1)
    printImageElement(img1,problemId)
    img2 = getImageElement(agent,imgId2,elementdx2)
    printImageElement(img2,problemId)
    similar,similar2,pixMatched,scale = img1.getImageElementSimilarScale(img2)
    print("[%s] 中 %s.%d 与 %s.%d 相似(similar) = %f similar2=%f pixMatched=%s, 比例 = %f" %(problemId,imgId1,elementdx1,imgId2,elementdx2,similar,similar2,pixMatched,scale))    
    #similar1,scale1 = img1.getImageElementSimilarScale(img2)
    #print("[%s] 中 %s.%d 与 %s.%d 相似 = %f, 比例 = %f" %(problemId,imgId1,elementdx1,imgId2,elementdx2,similar1,scale1))   
    #similar1,scale1 = img1.getImageElementSimilarScale(img2)
    #print("[%s] 中 %s.%d 与 %s.%d 相似 = %f, 比例 = %f" %(problemId,imgId1,elementdx1,imgId2,elementdx2,similar1,scale1))   
 
def testCalculaImages2MatchScore(problemId,imgs1Name,imgs2Name):
    agent = prepareAgent(problemId)    
    scoreDesc = []
    score = agent.calculateImages2MatchScore(imgs1Name,imgs2Name,scoreDesc)
    desc = ""
    for  s in scoreDesc:
        if len(desc)>0:
            desc += " + "
        desc += "%d(%s)"%s
    print("[%s] : %s - %s 得分 = %d = %s" % (problemId,imgs1Name,imgs2Name,score,desc)) 
    pass    

#
# imgs1Name = "ABC"  otherImdId = "G"
#
def testImageIndexOfEqualsImage(problemId,imgFrmName,otherImdIds:list):
    agent = prepareAgent(problemId)   
    imgsFrm = agent.getImages3(imgFrmName)
    for otherImdId in otherImdIds:
        j = imgsFrm.getIndexOfEqualsImage(otherImdId,[])
        print("图片 %s 在 %s 中的位置 = %d" % (otherImdId,imgFrmName,j))
    pass

def testSumImgElementsBlackPonts(problemId,imgsId:str):
    agent = prepareAgent(problemId)   
    for imgId in imgsId:
        n = ImageElement.getSumImgElementsBlackPoints(agent.getImageElements(imgId))
        print("[%s] %s 的前景像素 = %d, 元素个数 =%d " %(problemId,imgId,n,len(agent.getImageElements(imgId))))
        img = agent.getImage1(imgId)
        n = img.getSumImgElementsBlackPoints()
        print("    ...%d",n)
        n = img.getSumImgElementsBlackPoints()
        print("    ...%d",n)
        agent.getImageElements(imgId)[0].update()
        print("[%s] %s 的前景像素 = %d  " %(problemId,imgId,agent.getImageElements(imgId)[0].blackPixelCount))

def testImgBlackPointsRatio(problemId:str,img3Id:str)->None:
    agent = prepareAgent(problemId)   
    # AB BC AC
    for imgsName in [img3Id[0:2],img3Id[1:3],img3Id[0:1]+img3Id[2:3]]:
        print( imgsName )
        img2 = agent.getImages2(imgsName)
        print("[%s] %s 的前景像素比例 = %f = %d/%d" %(problemId,imgsName,img2.getBlackRatio(),ImageElement.getSumImgElementsBlackPoints(agent.getImageElements(imgsName[0:1])),ImageElement.getSumImgElementsBlackPoints(agent.getImageElements(imgsName[1:2]))))

    img3 = agent.getImages3(img3Id)
    print("[%s] %s 的前景像素比例=%s" %(problemId,img3Id,img3.getImagePixelRatio()))

def testIsImg2ElementsSwapped(problemId:str,img2Id:str)->None:
    agent = prepareAgent(problemId)  
    img2 = agent.getImages2(img2Id)
    print("[%s] %s Img2ElementsSwapped = %s " %(problemId,img2Id,img2.isImg2ElementsSwapped()))

def  testGetNotEqImgElementIdx(problemId:str,img3Id:str)->None:
    agent = prepareAgent(problemId)  
    img3 = agent.getImages3(img3Id)
    i = img3.getNotEqImgElementIdx()
    print("[%s] %s getNotEqImgElementIdx = %d" %(problemId,img3Id,i) )

def testIsIncSameElements(problemId:str,img2Id:str)->None:
    agent = prepareAgent(problemId)  
    img2 = agent.getImages2(img2Id)
    v = img2.isIncSameElements()
    print("[%s] %s isIncSameElements = %s" %(problemId,img2Id,v) )

def test_allElementsInCenter(problemId:str,imgIds:list)->None:    
    agent = prepareAgent(problemId)  
    for imgId in imgIds:
        elements = agent.getImageElements(imgId)
        x = ImageElement.allElementsInCenterX(elements)
        y = ImageElement.allElementsInCenterY(elements)
        print("[%s] %s allElementsInCenter = x=%f,y=%f" %(problemId,imgId,x,y) )

def test_allElementsInCenter3(problemId:str,img3Id:str)->None:    
    agent = prepareAgent(problemId)          
    img3 = agent.getImages3(img3Id)
    x = img3.allElementsInCenterX()
    y = img3.allElementsInCenterY()
    print("[%s] %s allElementsInCenter = x=%f,y=%f" %(problemId,img3Id,x,y) )

#
# rotaMode IMGTRANSMODE_ROTATE1,IMGTRANSMODE_ROTATE2, IMGTRANSMODE_ROTATE3
#   ROTATE90, ROTATE270
#
def test_RotateImage(problemId:str,img2Id:str,rotaMode:str)->None:
    agent = prepareAgent(problemId)          
    img = agent.getImages2(img2Id)
    #imgsFrm1.img1.getRotateImage(checkAllRota).isEquals(imgsFrm1.img2.asImgElement()) 
    img1 = img.img1.getRotateImage(rotaMode)
    printImageElement(img1,problemId)
    
    print("[%s] %s : ImgElementTransMatched =%s" %(img2Id,rotaMode,img.isImgElementTransMatched(0,rotaMode)))
    trans = img.getImgElementTrans(0,rotaMode)
    print("[%s] %s :  matched=%s,similar=%s" %(img2Id,rotaMode,trans.matched,trans.similar))

def test_isRoteteMatched(problemId:str,img2Id:str,rotate:int)->None:
    agent = prepareAgent(problemId)          
    img = agent.getImages2(img2Id)
    print("[%s] %s : isRoteteMatched(%d) =%s" %(problemId,img2Id,rotate,img.isRoteteMatched(0,rotate)))    

def test_getNotEqImgElementIdx(problemId:str,img3Id:str)->None:
    agent = prepareAgent(problemId)          
    img = agent.getImages3(img3Id)
    print("NotEqImgElementIdx : ",img.getNotEqImgElementIdx())

def test_getXORImage(problemId:str,img3Id:str)->None:
    agent = prepareAgent(problemId)          
    img = agent.getImages3(img3Id).getXORImage()
    printImageElement(img,problemId)

def test_XORImageCmp(problemId:str,img3Id1:str,img3Id2:str)->None:    
    agent = prepareAgent(problemId)          
    img1 = getImagesX(agent,img3Id1).getXORImage()
    img2 = getImagesX(agent,img3Id2).getXORImage()
    printImageElement(img1,problemId)
    #cv2.imwrite('/temp/1.jpg',img1.image) 
    printImageElement(img2,problemId)
    #cv2.imwrite('/temp/2.jpg',img2.image) 
    similar,similar2,pixMatched,scale = img1.getImageElementSimilarScale(img2)
    print("[%s] 中 %s 与 %s 相似 = %f %f pixMatched=%s, 比例 = %f" %(problemId,img1.name,img2.name,similar,similar2,pixMatched,scale))    
    r,n1,n2 = countImageDiffRatio(img1.image,img2.image)
    print("%f %d/%d" %(r,n1,n2 ))
    
def test_isFilledImage(problemId:str,imgIdLst:str,elementIdx:int=0)->None:    
    agent = prepareAgent(problemId) 
    for imgId in imgIdLst:
        img = agent.getImage1(imgId)
        filled = img.getImageElements()[elementIdx].isFilledImage()
        print("[%s] %s .%d isFilled = %s" %(problemId,img.name,elementIdx,filled))
        #img.getImageElements()[elementIdx].isFilledImage()

def test_isWholeImgElementsFliped(problemId:str,imgs2Id:str)->None:
    agent = prepareAgent(problemId) 
    img = agent.getImages2(imgs2Id)
    wholeImgElementsFlipedH = img.isWholeImgElementsFliped("FLIPH")
    wholeImgElementsFlipedV = img.isWholeImgElementsFliped("FLIPV")
    print("[%s] %s : isWholeImgElementsFliped FLIPH = %s, FLIPV %s" %(problemId,img.name,wholeImgElementsFlipedH,wholeImgElementsFlipedV))

def test_isAllElementsEquals(problemId:str,imgIdLst:str)->None:
    agent = prepareAgent(problemId) 
    for imgId in imgIdLst:
        img = agent.getImage1(imgId)
        allElementsEquals = ImageElement.isAllElementsEquals(img.getImageElements())
        print("[%s] %s : allElementsEquals=%s " %(problemId,img.name,allElementsEquals))

def test_getANDImage(problemId:str,imgs2Id:str)->None:
    agent = prepareAgent(problemId) 
    img = agent.getImages2(imgs2Id)
    CV2Utils.printImage1(img.getANDImage())

def test_getBitOPMatched(problemId:str,imgs3Id:str)->None: 
    agent = prepareAgent(problemId) 
    img = agent.getImages3(imgs3Id)
    andMatched = img.getBitOPMatched()
    print("[%s] %s : andMatched=%d " %(problemId,img.name,andMatched)) #diffRatio=0.042563,diffCount=1441
    
def test_getXORImage(problemId:str,imgs2Id:str)->None:
    agent = prepareAgent(problemId) 
    img = agent.getImages2(imgs2Id)
    CV2Utils.printImage1(img.getXORImage())    
    
def test_getORImage(problemId:str,imgs2Id:str)->None:
    agent = prepareAgent(problemId) 
    img = agent.getImages2(imgs2Id)
    orImg = img.getORImage()
    CV2Utils.printImage1(orImg)    
    cv2.imwrite('/temp/1.jpg',orImg) 

def test_allElementsInCenter1(problemId:str,imgIdLst:str)->None:
    agent = prepareAgent(problemId) 
    for imgId in imgIdLst:
        img = agent.getImage1(imgId)
        print("[%s] %s : allElementsInCenter=%s ; %s " %(problemId,img.name,img.allElementsInCenter(),img.getElementsCenterDistanceXY())) 

def test_getAllElementsInLine(problemId:str,imgIdLst:str)->None:
    agent = prepareAgent(problemId) 
    for imgId in imgIdLst:
        img = agent.getImage1(imgId)
        print("[%s] %s : allElementsInLine=%d " %(problemId,img.name,img.getAllElementsInLine())) 

def test_getImgElementsEqualsIdxMap(problemId:str,imgs2Id:str)->None:
    agent = prepareAgent(problemId) 
    img = agent.getImages2(imgs2Id)
    print("[%s] %s : ImgElementsEqualsIdxMap = %s" %(problemId,img.name,img.getImgElementsEqualsIdxMap()))
    print("---[%s] %s : ImgElementsEqualsIdxMap = %s" %(problemId,img.name,img.getImgElementsEqualsIdxMap()))

def test_isEqualsByElementIdx(problemId:str,imgsId:str,elementIdx=0):    
    agent = prepareAgent(problemId) 
    img = agent.getImages3(imgsId)
    #print("[%s] %s[%d] : imgsElements.len = %d %d %d" %(problemId,img.name,elementIdx,len(img.img1Elements),len(img.img2Elements),len(img.img3Elements)))
    print("[%s] %s[%d] : isEqualsByElementIdx = %s" %(problemId,img.name,elementIdx,img.isEqualsByElementIdx(elementIdx)))

def test_getOnlyNotEqElementIdx(problemId:str,imgsId:str):
    agent = prepareAgent(problemId) 
    img = agent.getImages3(imgsId)
    #print("[%s] %s[%d] : imgsElements.len = %d %d %d" %(problemId,img.name,elementIdx,len(img.img1Elements),len(img.img2Elements),len(img.img3Elements)))
    print("[%s] %s : onlyNotEqElementIdx = %d" %(problemId,img.name,img.getOnlyNotEqElementIdx()))

def test_getImgElementEqualsIdxMap(problemId:str,imgsId1:str,elementIdx1:int,imgsId2:str,elementIdx2:int)->None:
    agent = prepareAgent(problemId) 
    img1 = agent.getImages3(imgsId1)
    img2 = agent.getImages3(imgsId2)
    print("[%s] %s[%d]/%s[%d] : getImgElementEqualsIdxMap = %s" %(problemId,img1.name,elementIdx1,img2.name,elementIdx2,img1.getImgElementEqualsIdxMap(elementIdx1,img2,elementIdx2)))

def test_isLinesFielldImage(problemId:str,imgsIdLst:str,elementIdx=0):
    agent = prepareAgent(problemId) 
    for imgsId in imgsIdLst:
        img = agent.getImage1(imgsId)
        print("[%s] %s : isLinesFielldImage = %s" %(problemId,img.name,img.getImageElements()[elementIdx].isLinesFielldImage()))

def test_isDifferentFillMode(problemId:str,imgsId:str,elementIdx=0):
    agent = prepareAgent(problemId) 
    img = agent.getImages3(imgsId)
    #print("[%s] %s[%d] : imgsElements.len = %d %d %d" %(problemId,img.name,elementIdx,len(img.img1Elements),len(img.img2Elements),len(img.img3Elements)))
    print("[%s] %s : isDifferentFillMode = %s" %(problemId,img.name,img.isDifferentFillMode(elementIdx)))

def test_isOuterSimilarAllElements(problemId:str,imgId1,imgId2):
    agent = prepareAgent(problemId) 
    img1 = agent.getImage1(imgId1)
    img2 = agent.getImage1(imgId2)
    v = img1.getImageElements()[0].isOuterSimilarAllElements(img2.getImageElements())
    print("[%s] %s-%s : isOuterSimilarAllElements = %s" %(problemId,img1.name,img2.name,v))

def test_isMatchedLRMerged(problemId:str,imgId1,imgId2,imgId3):
    agent = prepareAgent(problemId) 
    img1 = agent.getImage1(imgId1)
    img2 = agent.getImage1(imgId2)
    img3 = agent.getImage1(imgId3)
    v = Image1.isMatchedLRMerged(img1,img2,img3)
    print("[%s] %s+%s==%s : isMatchedLRMerged = %s" %(problemId,img1.name,img2.name,img3.name,v))

def test_getIncedElements(problemId:str,imgs2Id:str):
    agent = prepareAgent(problemId) 
    img = agent.getImages2(imgs2Id)
    a =  img.getIncedElements()
    if a==None:
        print("[%s] %s.IncedElements = None" %(problemId,imgs2Id))
    else:
        print("[%s] %s.IncedElements = %s" %(problemId,imgs2Id,",".join(map(lambda e:e.name,a))))

def test_isIncedSameElements(problemId:str,imgs2Id1:str,imgs2Id2:str):
    agent = prepareAgent(problemId) 
    img1 = agent.getImages2(imgs2Id1)
    img2 = agent.getImages2(imgs2Id2)
    print("[%s] %s-%s.isIncedSameElements = %s" %(problemId,imgs2Id1,imgs2Id2,img1.isIncedSameElements(img2)))


def tmpTestImage()->None:    
    problemId = "Challenge D-09"
    agent = prepareAgent(problemId) 
    img1 = agent.getImages3("ABC")
    img2 = agent.getImages3("GH7")
    img1XOR = img1.getXORImageElement()
    img2XOR = img2.getXORImageElement()
    cv2.imwrite('/temp/1.jpg',img1XOR.image) 
    cv2.imwrite('/temp/2.jpg',img2XOR.image) 
    #e1 = img.img1Elements[0]
    #e2 = img.img1Elements[1]
    #print( "%s %s" %(e1.isImageShapeMatched(e2),e1.isBlackPixelEquals(e2)))
    #e1 = img.img1Elements[1]
    #e2 = img.img1Elements[0]
    #print( "%s %s" %(e1.isImageShapeMatched(e2),e1.isBlackPixelEquals(e2)))
    #orImg = img.getORImage()
    #imgC = agent.getImage("C")
    #xorImg = cv2.bitwise_not(cv2.bitwise_xor(orImg,imgC,mask=None),mask=None)
    #cv2.imwrite('/temp/1.jpg',orImg) 
    #cv2.imwrite('/temp/2.jpg',xorImg) 
        
#
#  D-04
#    
def testAgentSolve(problemId):  
    agent = Agent()
    problem =  loadProblemByID(problemId)
    answer = agent.Solve(problem) 
    answerInfo = ""
    if answer!=problem.correctAnswer and problem.correctAnswer>0:
        answerInfo = "(!!!期望结果 = %d)" % problem.correctAnswer
    print("[%s] 结果 = %d  %s" % ( problem.name, answer,answerInfo))

def testAgentSolveChallenge(problemId):
    agent = Agent()
    problem =  loadChallengeProblemByID(problemId)
    answer = agent.Solve(problem) 
    answerInfo = ""
    if answer!=problem.correctAnswer and problem.correctAnswer>0:
        answerInfo = "(!!!期望结果 = %d)" % problem.correctAnswer
    print("[%s] 结果 = %d  %s" % ( problem.name, answer,answerInfo))

def testSolveProblemSet(setName:str):
    print("SolveProblemSet [%s]"%setName)
    startTime = time()
    problemSet = ProblemSet(setName)
    agent = Agent()
    totalProblems = 0
    correctProblems = 0
    errProblems = []
    for problem in problemSet.problems:   # Your agent will solve one problem at a time.
            #try:
        answer = agent.Solve(problem)  # The problem will be passed to your agent as a RavensProblem object as a parameter to the Solve method
                                            # Your agent should return its answer at the conclusion of the execution of Solve.
            #    results.write("%s,%s,%d\n" % (set.name, problem.name, answer))
        answerInfo = ""
        if answer!=problem.correctAnswer and problem.correctAnswer>0:
            answerInfo = "(!!!期望结果 =%d)" % problem.correctAnswer
            errProblems.append(problem.name[-2:])
        else:
            correctProblems += 1    
        print("[%s] : 结果 = %d %s\n" % (problem.name, answer,answerInfo))
        totalProblems += 1
        #print("%s . %s : 结果 = %d %s\n" % (set.name, problem.name, answer,answerInfo))
    print("[%s] %d/%d , 耗时=%f %s" %(setName,correctProblems,totalProblems,(time()-startTime),"" if len(errProblems)==0 else "错误:"+",".join(errProblems)))
    
def main():
    Agent._DEBUG = True
    Agent._WARN = True
    #tempTest()
    #return
    #showSplitImages(problemId,"H")
    #testHLineSegments("B-04","C")
    #testHLineSegments("B-10","C")
    #testHLineSegments2("B-04","C")
    #testVLineSegments("B-04","C")
    #testVLineSegments2("B-04","C")
    #testImageFilledRatio("B-06","B")
    #showSplitImages("B-12","A") #** 5 个 圆
    #showSplitImages("B-02","A") #** 圆 + 十字 , 十字 像素 占比 0.39
    #showSplitImages("E-01","1") 
    #showSplitImages("B-04","A")
    #showSplitImages("B-10","C")
    #showSplitImages("C-09","C")
    #showSplitImages("C-09","2")
    #showSplitImages("Challenge E-02","A")
    #showSplitImages("Challenge D-09","C")
    #showSplitImages("Challenge E-11","H")
    #testFlipImage("B-03","A") # ** 
    #testFlipImage("B-07","9") 
    #testFlipImage("B-05","A") 
    #testRotateImage("B-07","9") # **
    #testRotateImage("Challenge D-04","A")
    #testFilledImage("B-11","B") #心形 图
    #testFilledImage("B-07","C") # 3/4 弧图
    #testFilledImage("B-07","9")
    #testFilledImage("Challenge D-11","A")
    #testImageElementSimilarScale("B-11","A","B") # 心形 图 [B-11] 中 A 与 B 相似 = 1.000000, 比例 = 1.000000
    #testImageElementSimilarScale("B-12","A","B") # 两个不正的圆 0.88 [B-12] 中 A 与 B 相似 = 0.880000, 比例 = 2.200539
    #testImageElementSimilarScale("B-12","B","A") # [B-12] 中 B 与 A 相似 = 0.880000, 比例 = 0.454434
    #testImageElementSimilarScale("C-01","A","G") # [C-01] 中 A 与 G 相似 = 0.980000, 比例 = 0.121946
    #testImageElementSimilarScale("C-02","A","G",1,1)  # 中 A.1 与 G.1 相似 = 1.000000, 比例 = 1.000000
    #testImageElementSimilarScale("C-11","A","B") # 两个小菱形  [C-11] 中 A 与 B 相似 = 1.000000, 比例 = 0.958333
    #testImageElementSimilarScale("B-03","A","B") #[B-03] 中 A 与 B 相似 = 0.000000, 比例 = 1.000000
    #testImageElementSimilarScale("B-06","A-FLIPV","C") 
    #estImageElementSimilarScale("Challenge B-01","A","C")  #中 A.0 与 C.0 相似(similar) = 0.980000 similar2=0.000000 
    #testImageElementSimilarScale("Challenge B-03","A","B") # 相似 = 0.000000 1.000000 False
    #testImageElementSimilarScale("Challenge D-02","B-ROTATE270","C")
    #testImageElementSimilarScale("Challenge B-07","C-ROTATE90","6")
    #testImageElementSimilarScale("Challenge B-07","C-FLIPH","6")  #  相似 = 0.000000 0.980000 True, 比例 = 1.000000
    #testImageElementSimilarScale("D-09","B","3") #  相似 = 0.000000 1.000000 False
    #testImageElementSimilarScale("D-09","A","1")  # 相似 = 1.000000
    #testImageElementSimilarScale("Challenge D-05","G","2") 
    #testImageElementSimilarScale("Challenge D-10","A","C",0,0) 
    #testImageElementSimilarScale("Challenge D-10","A","C",0,1) 
    #testImageElementSimilarScale("Challenge D-10","A","C",0,2) 



    #testImageTransInfo("B-02","AB") # 相等图形 园+十字
    #testImageTransInfo("B-03","AB") # B-03 - AB]元素-0: 变换=FLIPH 相似度=1.000000 大小比例=1.000000 
    #testImageTransInfo("B-04","AB") # [B-04 - AB]元素-0: 变换=FLIPH 相似度=0.980000 大小比例=1.000000 
    #testImageTransInfo("B-04","AC") # [B-04 - AC]元素-0: 变换=FLIPV 相似度=0.990000 大小比例=1.000000 
    #testImageTransInfo("B-05","AC") #[B-05 - AC]元素-0: 变换=FLIPV 相似度=1.000000 大小比例=1.016461 
    #testImageTransInfo("B-06","AB") # 没有 B-06 - AB]元素-0: 变换=None 相似度=0.000000 大小比例=0.000000 
    #testImageTransInfo("B-06","AC")  #[B-06 - AC]元素-0: 变换=FLIPV 相似度=1.000000 大小比例=1.000000, 变换=FLIPH 相似度=1.000000 大小比例=1.000000
    #testImageTransInfo("B-07","AB")  #[B-07 - AB]元素-0: 变换=FLIPH 相似度=0.980000 大小比例=1.000000 
    #testImageTransInfo("B-07","AC")  #[B-07 - AC]元素-0: 变换=None 相似度=0.000000 大小比例=0.000000 
    #testImageTransInfo("B-07","25") # [B-07 - 25]元素-0: 变换=UNFILLED 相似度=0.980000 大小比例=1.047740 
    #testImageTransInfo("B-07","52")  #[B-07 - 52]元素-0: 变换=FILLED 相似度=0.980000 大小比例=1.047740 
    #testImageTransInfo("B-09","AB") #[B-09 - AB]元素-0: 变换=FILLED 相似度=1.000000 大小比例=1.047363 
    #testImageTransInfo("B-09","BA")  #[B-09 - BA]元素-0: 变换=UNFILLED 相似度=1.000000 大小比例=1.047363 
    #testImageTransInfo("C-05","BD") # 三个相等
    #testImageTransInfo("Challenge B-02","AB") 
    #testImageTransInfo("Challenge B-03","AB")  #  变换=EQUALS 相似度=0.000000,1.000000 : 匹配度=False False True  大小比例=1.658481
    #testImageTransInfo("Challenge B-04","C6") 
    #testImageTransInfo("Challenge B-07","AB") #  变换=FLIPH 相似度=0.840000,0.970000 : 匹配度=False True True  大小比例=1.000000
    #testImageTransInfo("Challenge B-07","C6") #  变换=FLIPH 相似度=0.000000,0.980000 : 匹配度=False True True  大小比例=1.000000
    #testImageTransInfo("D-09","B3") # EQUALS 相似度=0.000000,1.000000 : 匹配度=False False True 
    
    #testCalculaImages2MatchScore("B-01","AB","C1")  #[B-01] : AB - C1 得分 = 3
    #testCalculaImages2MatchScore("B-01","AB","C2")  #[B-01] : AB - C2 得分 = 13
    #testCalculaImages2MatchScore("B-10","AC","B3")  #[B-10] : AC - B3 得分 = 30
    #testCalculaImages2MatchScore("B-10","AC","B2")  #[B-10] : AC - B2 得分 = 10
    
    #testImageIndexOfEqualsImage("C-02","ABC","G") # 全相同, 比例不同
    # testImageIndexOfEqualsImage("D-11","ABC","DEF") # 同 组合
      #D12
    #testGetNotEqImgElementIdx("D-06","GH3")  
    #test_allElementsInCenter("Challenge C-02",["A","B","C","8","6"])
    #test_allElementsInCenter3("Challenge C-02","A87")
    
    #test_countImagesDiff("E-03","GH","2")
    #test_countImagesDiff("E-03","AB","C")
    #test_countImagesDiff("E-02","AB","C")
    #test_countImagesDiff("E-02","GH","7")
    #test_countImagesDiff("E-04","BC","A")
    #test_countImagesXOR("D-11","AB")
    
    #testSumImgElementsBlackPonts("E-04","ABC")
    #testSumImgElementsBlackPonts("C-04","DEF")
    #testImgBlackPointsRatio("C-04","DEF")
    #testImgBlackPointsRatio("C-04","GH8")
    #testIsImg2ElementsSwapped("C-09","AC")

    #testIsIncSameElements("Challenge B-01","AC")
    #test_RotateImage("Challenge B-02","AB","ROTATE270")
    #test_RotateImage("Challenge B-10","AB",2)
    #test_RotateImage("Challenge D-02","AB","ROTATE270")
    #test_RotateImage("Challenge D-02","BC","ROTATE270")
    #test_RotateImage("Challenge D-02","GH","ROTATE270")
    #test_RotateImage("Challenge D-02","H1","ROTATE270")
    #test_isRoteteMatched("Challenge D-04","AB",-45)
    #test_isRoteteMatched("Challenge D-04","GH",-45)
    #test_getNotEqImgElementIdx("D-06","ABC")
    #test_getNotEqImgElementIdx("D-06","GH1")
    #test_getXORImage("D-09","ABC")
    #test_XORImageCmp("D-09","ABC","GH3") #0.026642 902/33856  0.062411
    #test_XORImageCmp("D-09","DEF","GH6") #0.120717 4087/33856
    #test_XORImageCmp("D-09","ABC","GH7") #0.117173 3967/33856   0.129903
    #test_getXORImage("D-09","AE3")
    #test_XORImageCmp("D-09","BFG","AE3") #0.031634 1071/33856  0.038339
    #test_XORImageCmp("Challenge B-04","AB","C4")
    #test_isFilledImage("Challenge B-04","A43")
    #test_isFilledImage("Challenge B-04","3") # True
    #test_isFilledImage("B-11","B") # False
    #test_isFilledImage("D-01","7") # 实心的 红心 True
    #test_isFilledImage("Challenge D-11","A123") # True
    #test_isFilledImage("Challenge E-07","78") # True
    #test_isWholeImgElementsFliped("C-07","AG")    
    #test_isWholeImgElementsFliped("C-07","BH")    
    #test_isWholeImgElementsFliped("C-07","C2")    
    #test_isAllElementsEquals("D-12","C")
    #test_isAllElementsEquals("D-12","ABCDEFG12345678")
    #test_getANDImage("E-10","AB")
    #test_getBitOPMatched("E-10","ABC") #diffRatio=0.042563,diffCount=1441 , andMatched=1 
    # test_getANDImage("E-10","GH")
    #test_getBitOPMatched("E-10","GH2") # andMatched=0
    #test_getBitOPMatched("E-10","GH8")  # andMatched=1
    #test_getBitOPMatched("Challenge E-01","ABC") #andMatched=2 
    #test_getORImage("Challenge E-02","AB")
    #test_getBitOPMatched("Challenge E-02","ABC") #andMatched=3
    #test_getBitOPMatched("Challenge E-02","DEF") #andMatched=3
    #test_getBitOPMatched("Challenge E-02","GH7") #andMatched=3
    
    #test_allElementsInCenter1("Challenge C-10","CF35BA")
    #test_getAllElementsInLine("Challenge C-10","CF3ADG")
    #test_getImgElementsEqualsIdxMap("Challenge D-05","G2")
    #test_getImgElementsEqualsIdxMap("Challenge D-05","G3")
    #test_getImgElementsEqualsIdxMap("Challenge D-05","G1")
    #test_getImgElementsEqualsIdxMap("Challenge D-05","G7")
    #test_isEqualsByElementIdx("Challenge D-06","ABC",0) # False
    #test_isEqualsByElementIdx("Challenge D-06","DEF",0) # True
    #test_isEqualsByElementIdx("Challenge D-06","DEF",1) # False
    #test_isEqualsByElementIdx("Challenge D-06","DE6",0) # False
    #test_isEqualsByElementIdx("Challenge D-06","D13",0) #False
    #test_isEqualsByElementIdx("Challenge D-06","GH6",0) # True
    #test_isEqualsByElementIdx("Challenge D-06","GH6",1) # True
    #test_isEqualsByElementIdx("Challenge D-06","GH6",2) # False
    #test_getOnlyNotEqElementIdx("Challenge D-06","ABC") # -3
    #test_getOnlyNotEqElementIdx("Challenge D-06","DEF") # 1
    #test_getOnlyNotEqElementIdx("Challenge D-06","DE2") # -2
    #test_getOnlyNotEqElementIdx("Challenge D-06","GH1") # -2 元素个数不等
    #test_getOnlyNotEqElementIdx("Challenge D-06","GH6") #  2
    #test_getImgElementEqualsIdxMap("Challenge D-06","DEF",1,"GH6",2) #  
    #test_getImgElementEqualsIdxMap("Challenge D-06","DEF",1,"GH5",2) #  None
    #test_isLinesFielldImage("Challenge D-07","B")
    #test_isLinesFielldImage("B-07","C")   # 0
    #test_isLinesFielldImage("Challenge B-07","C")  # 1: 水平填充
    #test_isLinesFielldImage("Challenge B-09","AC")   # 2 : 垂直填充
    #test_isLinesFielldImage("Challenge B-09","26")   # 3:
    #test_isLinesFielldImage("Challenge D-07","BD524") # 3
    #test_isLinesFielldImage("Challenge D-07","AC") # 0
    #test_isLinesFielldImage("Challenge D-09","C")
    
    #test_isLinesFielldImage("Challenge D-07","4") 
    #test_isDifferentFillMode("Challenge D-07","ABC")  #  True
    #test_isDifferentFillMode("Challenge D-07","AE1")  #  False
    #test_isDifferentFillMode("Challenge D-07","AE3")  # False
    #test_isDifferentFillMode("Challenge D-07","AHF")  # False
    
    #test_isOuterSimilarAllElements("Challenge D-10","A","C")
    #test_isMatchedLRMerged("Challenge E-04","B","C","A")
    #test_isMatchedLRMerged("Challenge E-04","E","F","D")
    #test_isMatchedLRMerged("Challenge E-04","H","5","G")

    #test_getIncedElements("Challenge E-11","AB") # B[1]
    #test_getIncedElements("Challenge E-11","BC") # C[1]
    #test_getIncedElements("Challenge E-11","GH") #  H[7]
    #test_getIncedElements("Challenge E-11","DE") # E[2]
    #test_getIncedElements("Challenge E-11","EF") # F[5]
    #test_getIncedElements("Challenge E-11","H3") # 3[4]
    #test_getIncedElements("Challenge E-11","H6") # 6[8]
    #test_getIncedElements("Challenge E-11","AD") # D[1],D[2],D[3]
    #test_getIncedElements("Challenge E-11","CF") #  F[2],F[4],F[5]
    #test_getIncedElements("Challenge E-11","DG") #  G[2],G[3],G[6]
    #test_getIncedElements("Challenge E-11","F3") #  3[3],3[4],3[8]
    #test_isIncedSameElements("Challenge E-11","AB","GH") # True
    #test_isIncedSameElements("Challenge E-11","BC","H3") # True
    #test_isIncedSameElements("Challenge E-11","BC","H6") # False
    #test_isIncedSameElements("Challenge E-11","DE","GH") # 
    #test_isIncedSameElements("Challenge E-11","EF","H3") # 

    #tmpTestImage()
    #test_newFlipedImage("B-01","1")

    #testAgentSolve("B-05")
    #testAgentSolve("B-05")
    #testAgentSolve("B-06")
    #testAgentSolve("B-10")
    #testAgentSolve("B-12")
    #testAgentSolve("B-01")
    #testAgentSolve("C-01")
    #testAgentSolve("C-02")  # 两组图形6个全相同, 只是 比例 不同
    #testAgentSolve("C-03") #[ABC-GH4]两组图形元素个数变化按同倍数递增
    #testAgentSolve("C-08")
    #testAgentSolve("C-04")
    #testAgentSolve("C-07")  #[ABC-GH2]两组图形上下翻转关系
    #testAgentSolve("C-09") #???  
    #testAgentSolve("D-02")  # [ABC-GH1]两组图形具有相同组合
    #testAgentSolve("D-03")
    #testAgentSolve("D-04")
    #testAgentSolve("D-06")
    #testAgentSolve("D-08") #组图形外形具有相同组合, 且两组元素全为填充图或非填充图
    #testAgentSolve("D-11") #[BFG-AE3]每组图形全相等
    #testAgentSolve("D-10") 
    #testAgentSolve("D-09") #  两组图形每组XOR后的图形相似
    #testAgentSolve("D-10") #  [ABC-GH1]两组图形每组XOR后的图形相似
    #testAgentSolve("D-12") 
    #testAgentSolve("E-01") #  [ABC-GH1]前两图片像素合并==第三个图片
    #testAgentSolve("E-02") #]前两图片像素合并==第三个图片
    #testAgentSolve("E-03") # 前两图片像素合并==第三个图片
    #testAgentSolve("E-04") # [ADG-CF8]前两图片像素个数相加或减==第三个图片,且宽高匹配
    #testAgentSolve("E-08")  #  [ABC-GH1]00E000:两组图形每组XOR后的图形相似
    #testAgentSolve("E-10")  # DEF/GH8   结果 = 2  (!!!期望结果 = 8) : a bitand b==c ABC-GH8]两组图形 A bitand B == C 且 G bitand H==8
    #testAgentSolve("E-11")
    #testAgentSolve("C-06") #前两图片像素个数相加或减==第三个图片,且宽高匹配

    #testAgentSolve("Challenge B-01") 
    #testAgentSolve("Challenge B-03") # 两组元素增减个数相同
    #testAgentSolve("Challenge B-04") # [AB-C4]两组元素全为非填充图(区分答案3)
    #testAgentSolve("Challenge B-07")  #   [AB-C6]元素0匹配相同变换FLIPH(仅轮廓相似)
    #testAgentSolve("Challenge B-09")  #  [AB-C4] 两图片素个数变化率相差<0.05
    #testAgentSolve("Challenge B-06")   #  相等比较时, 斜线段 较多
    #testAgentSolve("Challenge B-07")   #  相等比较时,  需要考虑 match2 轮廓相似:[AB-C6]元素0匹配相同变换FLIPH(仅轮廓相似)
    #testAgentSolve("Challenge B-09")  #   [AB-C4]外形相似,且填充模式一致
    #testAgentSolve("Challenge B-10")  #  
    #testAgentSolve("Challenge C-04")
    #testAgentSolve("Challenge C-10")
    #testAgentSolve("Challenge C-12") # [ABC-GH2]两组图形外形具有相同组合,且两组元素大小匹配
    #testAgentSolve("Challenge C-02") #[ABC-GH7]子集关系
    #testAgentSolve("Challenge D-02")  #   [ABC-GH1]两组图形为-90度旋转关系
    #testAgentSolve("Challenge E-01") #[ABC-GH6]两组图形 A bitxor B == C 且 G bitxor H==6
    
    #testAgentSolve("Challenge B-04") # ??? 通过是否填充判断???
    #testAgentSolve("Challenge E-02")  # [ABC-GH7]两组图形 A bitor B == C 且 G bitor H==7
    #***********testAgentSolve("Challenge C-10")
    #******testAgentSolve("Challenge C-12") # ???? 结果 = 7 (!!!期望结果 =8) ;  test2:4
    #testAgentSolve("Challenge D-04") #  [DEF-GH6]两组图形为45度旋转关系
    #********testAgentSolve("Challenge D-06")
    #***********testAgentSolve("Challenge D-07") #[ABC-GH4]两组图形外形具有相同组合,且两组元素各使用不同的填充模式
    #testAgentSolve("Challenge D-09") # ??? 全依靠了 附加分
    #***********testAgentSolve("Challenge D-10") # [ADG-CF5]第一组中三个图片分别等于第二组中三个图片的每个元素
    #**********testAgentSolve("Challenge D-12") # [ABC-GH6]两组元素个数(不考虑次序的情况下)匹配

    
    #**********testAgentSolve("Challenge E-04") # [ABC-GH5]第2和3图片左右合并==第1个图片
    #
    # 有问题
    #
    #**********testAgentSolve("Challenge B-02")  # [AB-C1]满足旋转315度 
    #testAgentSolve("Challenge B-08")   #  顶点 规律 test2:2

    #testAgentSolve("C-08") # 整体对称 答案 = 2,3 (!!!期望结果 =5);  test2:1
    #testAgentSolve("C-09") # 多结果  答案 = 2,3
    #testAgentSolve("C-12") # ??? 结果 = 5 (!!!期望结果 =8) ;  test2:3


    #testAgentSolve("E-12") # ?????  答案 = 1,2,8;  (!!!期望结果 =6) ; test2 ok

    #*********testAgentSolve("Challenge C-04") # + 字=> # 字
    #testAgentSolve("Challenge C-07") #多结果  答案 = 3,7
    #testAgentSolve("Challenge C-08") # 整体对称, 同 C-8 结果 = 4 (!!!期望结果 =5) ; test2:7

    #testAgentSolve("Challenge D-01") #?????未找到规则
    #*****testAgentSolve("Challenge D-05") #?????未找到规则
    #testAgentSolve("Challenge D-11") #?????未找到规则

    #testAgentSolve("Challenge E-05") # 顶点 规律 ;  4,6,8,10,12 个数的顶点
    #testAgentSolve("Challenge E-06") #?????未找到规则
    #testAgentSolve("Challenge E-07") # 顶点 规律 , 4,6,8,10,12 个数的顶点
    #testAgentSolve("Challenge E-08") #?????未找到规则
    #testAgentSolve("Challenge E-09") # ?????未找到规则
    #testAgentSolve("Challenge E-10")  # ?????未找到规则
    #testAgentSolve("Challenge E-11") #[ABC-GH3]两组图形元素个数变化递增量相同,且AB与GH增加了相同元素,BC与H3也增加了相同元素
    #************testAgentSolve("Challenge E-12") #???


    #tempTest()
    #tempTest2("Challenge E-12","A")
    #tempTest1("C-10","C")
    #tempTest1("Challenge B-01","C")
    #tempTest2("C-10","A")
    #tempTest4ProblemSet("Basic Problems E")

    Agent._DEBUG = False
    #testSolveProblemSet("Basic Problems B")  #  ok
    #testSolveProblemSet("Challenge Problems B")  # 8

    #testSolveProblemSet("Basic Problems C") #   : 08 ,12 ? 9 
    #testSolveProblemSet("Basic Problems D") #   : ok
    #testSolveProblemSet("Basic Problems E") #      12
    #testSolveProblemSet("Challenge Problems C")  #  8 ;    ??? 7
    #testSolveProblemSet("Challenge Problems D")  #  1,11; ??5. 8,10,
    #testSolveProblemSet("Challenge Problems E")  # :05,06,07,08,09,10
    return

    
   # 
    
    


####################################
#
###################################

def tempTest():
    #agent = prepareAgent("B-06")
    agent = prepareAgent("Challenge E-11")
    imgA = agent.getImage1("G")
    imgB = agent.getImage1("H")
    for e1,e2 in zip(imgA.getImageElements(),imgB.getImageElements()[0:len(imgA.getImageElements())]):
        #print("%s %s"% (e1.name,e2.name))
        similar,similar2,pixMatched,scale = e1.getImageElementSimilarScale(e2)
        print("%s %s 相似(similar) = %f similar2=%f pixMatched=%s, 比例 = %f" %(e1.name,e2.name,similar,similar2,pixMatched,scale))    
    #imagA_Rota270 = imgA.getImageElements()[0].getRotateImage("ROTATE315")
    #img = agent.getImages2(img2Id)
    #imgsFrm1.img1.getRotateImage(checkAllRota).isEquals(imgsFrm1.img2.asImgElement()) 
    #printImageElement(imagA_Rota270,"")
    #cv2.imwrite('/temp/1.jpg',imagA_Rota270.image) 
    #similar,similar2,pixMatched,scale = imagA_Rota270.getImageElementSimilarScale(imgB.getImageElements()[0])
    #print(" 相似(similar) = %f similar2=%f pixMatched=%s, 比例 = %f" %(similar,similar2,pixMatched,scale))    
    #v= imagA_Rota270.isSimilar(imgB.asImgElement())
    #print("v=",v)
    #AC = agent.getImageTransInfo(srcImgId,dstImgId)
    #print("AC = ",AC)    
    #img2 = agent.getImages2("B3") # ImageTransformInfo
    #transInfo = img2.getImgElementTrans(0,"FLIPV")  # r similar,scale FLIPH FLIPV
    #print(" %s: %s  %f %f " %(transInfo.transMode,transInfo.matched,transInfo.similar,transInfo.scale))
    #transInfo = img2.getImgElementTrans(0,"FLIPH")  # r similar,scale FLIPH FLIPV
    #print(" %s: %s  %f %f " %(transInfo.transMode,transInfo.matched,transInfo.similar,transInfo.scale))

#loadProblemByID(problemId)
def  tempTest1(problemId,imgId):
    agent = prepareAgent(problemId)
    #img2 = agent.getImages3(imgId)
    img = agent.getImage1(imgId)
    e = img.asImgElement()
    print("%s-%s : e.blackPixelCount=%d, (%d,%d)-(%d,%d) %s"%(problemId,imgId,e.blackPixelCount,e.x0,e.y0,e.ex,e.ey, e.name))
    #printImageElement(e,problemId)
    e.update()
    print("%s-%s : e.blackPixelCount=%d, (%d,%d)-(%d,%d)"%(problemId,imgId,e.blackPixelCount,e.x0,e.y0,e.ex,e.ey))


def  tempTest3(problemId):        
    agent = prepareAgent(problemId)

#
# problemSetId: 
#
def  tempTest4ProblemSet(problemSetId):        
    problemSet = ProblemSet(problemSetId)
    agent = Agent()
    for problem in problemSet.problems: 
        tempTest4Problem(problem)

def  tempTest4Problem(problem):
    agent = Agent()
    agent.prepareProblem(problem)
    for imgId in agent.imagesFrame:
        if len(imgId)!=1:
            continue
        imgs1 = agent.getImageElements(imgId) # 旧版
        imgs2 = agent.imagesFrame[imgId].getImageElements()  # 新 版
        print("%s-%s : 元素个数 =%d, %d" %(problem.name,imgId,len(imgs1),len(imgs2)))
        if len(imgs1)!=len(imgs2):
            raise BaseException("len(imgs1)!=len(imgs2) : %d!=%d" %(len(imgs1),len(imgs2)))
        i = 0
        for i in range(len(imgs1)):
            e1 = imgs1[i]
            e2 = imgs2[i]
            if e1.name!=e2.name or e1.blackPixelCount!=e2.blackPixelCount or e1.getTotalPixel()!=e2.getTotalPixel() or e1.x0!=e2.x0 or e1.y0!=e2.y0 or e1.ex!=e2.ex or e1.ey!=e2.ey :
                raise BaseException("%s-%s : error " %(problem.name,imgId))
            #print("%s-%s : %s %d/%d , %s %d/%d" %(problemId,imgId,e1.name,e1.blackPixelCount,e1.getTotalPixel(),e2.name,e2.blackPixelCount,e2.getTotalPixel()))
            #printImageElement(e,problemId)

def test_newFlipedImage(problemId,imgId,elementIdx=0):
    agent = prepareAgent(problemId)
    #img2 = agent.getImages3(imgId)
    img = agent.getImage1(imgId)
    e = img.getImageElement(elementIdx)
    filledImg1 = e._newFlipedImage("filledImg")
    printImageElement(filledImg1,problemId)
    filledImg2 = e.getFilledImage()
    printImageElement(filledImg2,problemId)
    print("[元素(%s)] : 元素区间(%d,%d) - (%d,%d) 像素个数=%d/%d :" %(filledImg1.name,filledImg1.x0,filledImg1.y0,filledImg1.ex,filledImg1.ey,filledImg1.blackPixelCount,filledImg1.getTotalPixel()))
    print("[元素(%s)] : 元素区间(%d,%d) - (%d,%d) 像素个数=%d/%d :" %(filledImg2.name,filledImg2.x0,filledImg2.y0,filledImg2.ex,filledImg2.ey,filledImg2.blackPixelCount,filledImg2.getTotalPixel()))
    if filledImg1.blackPixelCount!=filledImg2.blackPixelCount \
       or filledImg1.x0!=filledImg2.x0 \
       or filledImg1.y0!=filledImg2.y0 \
       or filledImg1.ex!=filledImg2.ex \
       or filledImg1.ey!=filledImg2.ey :
        raise BaseException(problemId+"-"+imgId+"["+elementIdx+"]")

def forImgElement1(e):
    filledImg1 = e._newFlipedImage("filledImg")
    #printImageElement(filledImg1,problemId)
    filledImg2 = e.getFilledImage()
    #print("[元素(%s)] : 元素区间(%d,%d) - (%d,%d) 像素个数=%d/%d :" %(filledImg1.name,filledImg1.x0,filledImg1.y0,filledImg1.ex,filledImg1.ey,filledImg1.blackPixelCount,filledImg1.getTotalPixel()))
    #print("[元素(%s)] : 元素区间(%d,%d) - (%d,%d) 像素个数=%d/%d :" %(filledImg2.name,filledImg2.x0,filledImg2.y0,filledImg2.ex,filledImg2.ey,filledImg2.blackPixelCount,filledImg2.getTotalPixel()))
    if filledImg1.blackPixelCount!=filledImg2.blackPixelCount \
       or filledImg1.x0!=filledImg2.x0 \
       or filledImg1.y0!=filledImg2.y0 \
       or filledImg1.ex!=filledImg2.ex \
       or filledImg1.ey!=filledImg2.ey :
        print("!!!!!!!!!!!!!!!!!!!------------",e.name)
         #raise BaseException(e.name)
#
# problemSet = "B" or "Challenge B"
#
def forEachImageElement(problemSet:str,call):
    for i in range(1,13):
        problemId = problemSet+"-"+("%02d" % i) 
        print(problemId)   
        agent = prepareAgent(problemId)
        for imgId in agent.images.keys():
            #print(problemId+" - "+imgId)
            img = agent.getImage1(imgId)
            for e in img.getImageElements(): 
                if e.getTotalPixel()<10*10: break
                call(e)

def forProblemSets(call1,call2):   
    #for problemSet in ["B","C","D","E","Challenge B","Challenge C","Challenge D","Challenge E"]: 
    for problemSet in ["Challenge C","Challenge E"]: 
    #for problemSet in ["E","Challenge C","Challenge D","Challenge E"]:   
    #for problemSet in ["B"]:
        call1(problemSet,call2)           
            
if __name__ == "__main__":
    main()    
    #main2()    
    #tempTest()
    #test_newFlipedImage("E-04")
    #forEachImageElement("D",forImgElement1)    
    #forProblemSets(forEachImageElement,forImgElement1)
#    CaseAllTransMode = [IMGTRANSMODE_EQ,IMGTRANSMODE_FLIPV,IMGTRANSMODE_FLIPH,IMGTRANSMODE_FILLED,IMGTRANSMODE_UNFILLED]


