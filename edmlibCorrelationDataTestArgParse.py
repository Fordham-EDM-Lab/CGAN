import time

from numpy import string_
#from edmlib import gradeData
from scipy.stats.stats import pearsonr
import networkx as nx
import filecmp
from edmlib.edmlib import *
#from edmlib.gradeData import gradeData
from edmlib.classCorrelationData import classCorrelationData

from shutil import copyfile # Since we use this only once, and it is inside a conditional, would it be better to place this within that conditional?

import argparse
import sys
import os
import shutil
from edmlib.edmlib import edmApplication, outDir
import re
#import matplotlib.pyplot as plt

start_time = time.time()
#df = classCorrelationData('Course-Correlation-Matrix-v1.csv') #('/u/erdos/edmProject/final-datamart-6-7-19.csv')

# I added rows into the excel for testing purposes - they are the rows with cool as course1
# majorFiltered = df.df.copy()
# majorFiltered['course1'] = majorFiltered['course1'].apply(lambda course: print(course))

def main(args):
    addedName = ""
    export = False
    df = classCorrelationData(os.path.join(args.datadir, args.csvfile))
    
    # checks which function is specified and then carries out the function
    # if we can export the result, we set export to True (scroll down to view what happens when export is True/False
    if(args.funct == "getClassesUsed"):
        print(df.getClassesUsed())
    elif(args.funct == "getNumberOfClassesUsed"):
        print(df.getNumberOfClassesUsed())
    elif(args.funct == "printUniqueValuesInColumn"):
        df.printUniqueValuesInColumn(args.oneCol) #("course1")
    elif(args.funct == "printClassesUsed"):
        df.printClassesUsed()
    elif(args.funct == "getEntryCount"):
        print(df.getEntryCount())
    elif(args.funct == "printEntryCount"):
        df.printEntryCount()
    elif(args.funct == "printFirstXRows"):
        df.printFirstXRows(args.rows)
    elif(args.funct == "printMajors"):
        df.printMajors()

    elif(args.funct == "filterColumnToValues"):
        df.removeNanInColumn(args.oneCol) #("course1")

        print(df.filterColumnToValues(args.oneCol, args.valList)) #("corr", [0, 1]))
        # addedName += "oneCol" + args.oneCol + "valList"
        # for i in range(args.valList.len):
        #     addedName += args.valList[i] + "-"
        
        export = True
        #df.exportCSV(fileName = "currFilterColumnToValues.csv")
    
    elif(args.funct == "exportCSV"):
        print(df.exportCSV())

    elif(args.funct == "filterToMultipleMajorsOrClasses"):
        df.removeNanInColumn("course1")
        print(df.filterToMultipleMajorsOrClasses(args.majorList, args.classList, args.bool1)) #([], ["ComputerandInfoScience4001"], False))
        export = True
        #df.exportCSV(fileName = "currFilterToMultipleMajorsOrClasses.csv")

    elif(args.funct == "getEntryCount"):
        print(df.getEntryCount()) # to check the remaining entry count after the filterToMultipleMajorsOrClasses function

    elif(args.funct == "substituteSubStrInColumn"):
        print(df.substituteSubStrInColumn(args.oneCol, args.subString, args.substitute)) #('course1', 'cool', 'drool'))
        export = True
    elif(args.funct == "printUniqueValuesInColumn"):
        df.printUniqueValuesInColumn(args.oneCol) #("course1") # to print the unique values after the substituteSubStrInColumn function

    elif(args.funct == "chordGraphByMajor"):
        # made sure all the required columns exist
        df.dropMissingValuesInColumn("corr")
        df.dropMissingValuesInColumn("P-value")
        df.dropMissingValuesInColumn("course1")
        df.dropMissingValuesInColumn("course2")
        print(df.chordGraphByMajor(args.corr, args.pVal, args.fileName, args.outputSize, args.imageSize, args.bool1, args.bool2)) #(0.4, 1)
        
        export = True
        # NOTE: This graph csv is already exported within the function :)
    # Trying to test this - START
    # graph = df.getNxGraph(None)

    # g = nx.to_dict_of_dicts(graph)
    # print(g)
    # print(plt.plot(df.getNxGraph())) # how to check graph?

    elif(args.funct == "getCliques"):
        print(df.getCliques(args.minNodes, args.corr))
    elif(args.funct == "outputCliqueDistribution"):
        df.outputCliqueDistribution()

    elif(args.funct == "makeMissingValuesNanInColumn"):
        df.printUniqueValuesInColumn(args.oneCol) #("course1")
        
        df.makeMissingValuesNanInColumn(args.oneCol) #("course1") # Note: to test this, I made one of the rows have a space as the input, we can delete that testing purpose row later
        export = True
        
        df.printUniqueValuesInColumn(args.oneCol) #("course1")
    
    elif(args.funct == "removeNanInColumn"):
        df.printUniqueValuesInColumn(args.oneCol)
        
        df.removeNanInColumn(args.oneCol) #("course1")
        export = True
        
        df.printUniqueValuesInColumn(args.oneCol) #("course1") # to test if the row containing nan has been removed

    elif(args.funct == "dropMissingValuesInColumn"):
        df.printUniqueValuesInColumn(args.oneCol) #("course1") # check the values before the 'dropMissingValuesInColumn' function
        
        df.dropMissingValuesInColumn(args.oneCol) #("course1")
        export = True
        
        df.printUniqueValuesInColumn(args.oneCol) #("course1") # check values after the 'dropMissingValuesInColumn' function

    elif(args.funct == "convertColumnToString"):
        df.convertColumnToString(args.oneCol) #("course1")
    else:
        print("Error: Please Input Valid Correlation Data Function Name")
        return

    # if export is True, we check if the CSV correct file is already made
    # if the CSV correct file exists:
    #   we create another CSV file containing the output from this run
    #   we then compare the current output CSV to the correct output CSV
    # else:
    #   we make the output from this run equal to the CSV correct file
    if(export):
        if(args.funct == "chordGraphByMajor"):
            correctFile = "correctChordGraphByMajor" + args.fileName.capitalize() + ".csv" # addedName + ".csv"
        else:
            correctFile = "correct" + args.funct.capitalize() + ".csv" # addedName + ".csv"
        # simplify this (for example, I used './correctFiles' a few times, so I gotta make that a variable outside of the else)
        # 
        # also add an ask, where if there is a correct file already, we have the option to either replace it or make a copy of 
        # our current output + make an 'are you sure you want to REPLACE? (this cannot be undone)' after the replace option

        correctDir = "./correctFiles"

        if(os.path.isdir(correctDir) and os.path.isfile(correctDir + "/" + correctFile)):
            if(args.funct == "chordGraphByMajor"):
                exportFile = "currChordGraphByMajor" + args.fileName.capitalize() + ".csv" # addedName + ".csv"
            else:
                exportFile = "curr" + args.funct.capitalize() + ".csv" # addedName + ".csv"
            save_path = './outputFiles'
            
            isdir = os.path.isdir(save_path)
            if(not isdir):
                os.mkdir(save_path)
            
            completeName = save_path + "/" + exportFile

            if(args.funct == "chordGraphByMajor"):
                # copy the file made in the 'exports' folder and put that into outputFiles (I recommend using shututil)
                copyfile(args.fileName + '.csv', completeName)
            else:
                df.exportCSV(fileName = completeName)
            # check this file against file w fileName "check" + args.funct.capitalize() + ".csv"
            testBool = filecmp.cmp(completeName, correctDir + "/" + correctFile, False)
            
            if testBool:
                print("output matches correct file")
            else:
                print("Error: this output does not match the correct file output")
            return
        

        print("No file to check against ... making this output the correct CSV file ...")

        isdir = os.path.isdir(correctDir)
        if(not isdir):
            os.mkdir(correctDir)

        #completeName = os.path.join(save_path, correctFile)
        completeName = correctDir + "/" + correctFile

        if(args.funct == "chordGraphByMajor"):
            # copy the file in exports folder and save as completeName
            copyfile(args.fileName + '.csv', completeName)
        else:
            df.exportCSV(fileName = completeName)
        print("Saved to " + completeName)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("funct", help="function to test")
    #args = parser.parse_args()
    parser.add_argument("--csvfile", default="Course-Correlation-Matrix-v1.csv", help="csv file with grade data")
    
    # Note: I used default values even for the ones that do not have defaults in the original,
    #       may want to change it so whether it is optional is dependent on the functions in which they are used in
    # parser.add_argument("--oneCol", type = str, default = "course1", help = "the specified column for the function")
    
    # if(args.funct == "printUniqueValuesInColumn" or args.funct == "filterColumnToValues" or args.funct == "substituteSubStrInColumn"):
    parser.add_argument("--oneCol", type = str, default = "course1", help = "the specified column for the function")

    # elif(args.funct == "filterColumnToValues"):
    parser.add_argument("--valList", type = str, default = [], help = "a list of values")
    
    # elif(args.funct == "printFirstXRows"):
        # used 0 as default rows, but need to check!
    parser.add_argument("--rows", type = str, default = "0", help = "the number of rows to print")
    
    # elif(args.funct == "exportCSV" or args.funct == "chordGraphByMajor" or args.funct == "outputCliqueDistribution"):
        # tried default = None, check if it works!
    parser.add_argument("--fileName", type = str, default = "majorGraph", help = "the name to set the exported CSV file to")
    
    # elif(args.funct == "filterToMultipleMajorsOrClasses"):
    parser.add_argument("--majorList", type = list, default = [], help = "a list of majors")
    
    # elif(args.funct == "filterToMultipleMajorsOrClasses"):
    parser.add_argument("--classList", type = list, default = [], help = "a list of classes")
    
    # elif(args.funct == "filterToMultipleMajorsOrClasses" or args.funct == "chordGraphByMajor" or args.funct == "outputCliqueDistribution"):
    parser.add_argument("--bool1", type = bool, default = True, help = "Boolean 1 to use for functions")
    
    # elif(args.funct == "chordGraphByMajor" or args.funct == "outputCliqueDistribution"):
    parser.add_argument("--bool2", type = bool, default = True, help = "Boolean 2 to use for functions")

    # elif(args.funct == "outputCliqueDistribution"):
    parser.add_argument("--logScale", type = bool, default = False, help = "Whether or not to output graph in Log 10 scale on the y-axis")

    parser.add_argument("--exportPNG", type = bool, default = True, help = "Whether or not to export a PNG version of the graph")

    # elif(args.funct == "chordGraphByMajor" or args.funct == "getNxGraph" or args.funct == "getCliques" or args.funct == "outputCliqueDistribution"):
    parser.add_argument("--corr", type = float, default = 0.4, help = "Correlation value that will be compared against")
    
    # elif(args.funct == "chordGraphByMajor"):
    parser.add_argument("--pVal", type = float, default = 1, help = "P value that will be compared against")
    
    # elif(args.funct == "chordGraphByMajor"):
    parser.add_argument("--outputSize", type = int, default = 200, help = "Size (units unknown) of html graph to output")

    # elif(args.funct == "chordGraphByMajor"):
    parser.add_argument("--imageSize", type = int, default = 300, help = "Size (units unknown) of image of the graph to output")

    # elif(args.funct == "substituteSubStrInColumn"):
    parser.add_argument("--subString", type = str, default = "", help = "string to replace in the replace function")

    parser.add_argument("--substitute", type = str, default = "", help = "string that will replace toRepStr")
    
    # elif(args.funct == "outputCliqueDistribution"):
    parser.add_argument("--title", type = str, default = "Class Correlation Cliques", help = "Title displayed on the histogram")
    
    
    parser.add_argument("-D", "--datadir", default = "./", help = "data directory")
    '''
    elif(args.funct == ""):
        parser.add_argument("-e", "--edges", type = int, default = 10, help = "minimum edge value")
    '''
    args = parser.parse_args()
    main(args)



# ALL QUESTIONS ANSWERED

# df.filterToMultipleMajorsOrClasses(majorsToFilterTo, coreClasses + preMedClasses + otherClasses)
# df.filterByGpaDeviationMoreThan(0.2)
# df.defineWorkingColumns('OTCM_FinalGradeN', 'SID', 'REG_term',
#         'REG_CourseCrn', 'REG_Programcode', 'REG_Numbercode', 'GRA_MajorAtGraduation', 'REG_REG_credHr')
# print("--- %s seconds ---" % (time.time() - start_time))
# start_time = time.time()
# df.exportCorrelationsWithAvailableClasses()
# print("--- %s seconds ---" % (time.time() - start_time))
