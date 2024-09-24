
#
# python D:\snsoftn10\snadk-srcx\python-projects\RPM-Project-Code\TestAgent.py
# python3 /snsoftn10/snadk-srcx/python-projects/RPM-Project-Code/TestAgent.py
#

import random
import re
import os
import json
from CV2Utils import CV2Utils 
#import json
from Agent import Agent,APPPATH,load_problem_images
from RavensFigure import RavensFigure
from RavensProblem import RavensProblem
from RavensObject import RavensObject
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
            
def main():
    Agent.DB_LEVEL = "DEBUG"
    Agent._DEBUG = True
    agent = Agent()
    #setName = "Basic Problems B"
    #problemName = "Basic Problem B-01"
    #problemName = "Basic Problem B-04"
    #problem =  loadProblem(setName,problemName)
    #problemId = "B-10"
    #problemId = "B-08"
    #problemId = "B-06"
    #problemId = "B-10"
    problemId = "B-11"
    problem =  loadBasicProblemByID(problemId)
    #print ("agent = %s" % (agent) )
    agent.images, agent.potential_answers = load_problem_images(problem)
    #e = agent.imageElementsSubtract("C","B")
    #CV2Utils.printImage2(e.image)
    #answer = agent.try_solve_2x2_byImgElementChange()
    
    answer = agent.Solve(problem) 
    answerInfo = ""
    if answer!=problem.correctAnswer and problem.correctAnswer>0:
        answerInfo = "(!!!期望结果 =%d)" % problem.correctAnswer
    print("[%s] 结果 =%d %s" % ( problem.name, answer,answerInfo))
    

if __name__ == "__main__":
    main()    
    


