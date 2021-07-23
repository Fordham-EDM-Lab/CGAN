import time

from numpy import string_
from edmlib import gradeData
from scipy.stats.stats import pearsonr
import networkx as nx
from edmlib.edmlib import *
from edmlib.gradeData import gradeData
from edmlib.classCorrelationData import classCorrelationData

import argparse
import os
from edmlib.edmlib import edmApplication, outDir
import re
#import matplotlib.pyplot as plt

start_time = time.time()
#df = classCorrelationData('Course-Correlation-Matrix-v1.csv') #('/u/erdos/edmProject/final-datamart-6-7-19.csv')

majorsToFilterTo = ['Computer and Info Science', 
                    'Psychology']
coreClasses = [ 'Philosophy1000',
                'Theology1000',
                'English1102',
                'English1101',
                'History1000',
                'Theology3200',
                'VisualArts1101',
                'Physics1201',
                'Chemistry1101']
preMedClasses = ['BiologicalSciences1403',
                 'BiologicalSciences1413',
                 'BiologicalSciences1404',
                 'BiologicalScience1414',
                 'Chemistry1321',
                 'Chemistry1331',
                 'Chemistry1322',
                 'Chemistry1332']
otherClasses = ['Physics1501',
                    'Physics1511',
                    'Economics2140',
                    'Mathematics1100',
                    'Theatre1100',
                    'Music1100']

# I added rows into the excel for testing purposes - they are the rows with cool as course1
# majorFiltered = df.df.copy()
# majorFiltered['course1'] = majorFiltered['course1'].apply(lambda course: print(course))

def main(args):
    df = classCorrelationData(os.path.join(args.datadir, args.csvfile))
    print(df.getClassesUsed())
    # print(df.getNumberOfClassesUsed())
    # print(df.printUniqueValuesInColumn(args.oneCol)) #("course1")
    # print(df.printClassesUsed())
    # print(df.getEntryCount())
    # df.printEntryCount()
    # print(df.printFirstXRows())
    # print(df.printMajors())

    # print(df.filterColumnToValues(args.oneCol, args.list) #("corr", [0, 1]))
    # print(df.exportCSV())

    # print(df.filterToMultipleMajorsOrClasses(args.majorList, args.classList, args.bool1) #([], ["ComputerandInfoScience4001"], False))
    # print(df.getEntryCount()) # to check the remaining entry count after the filterToMultipleMajorsOrClasses function

    # print(df.substituteSubStrInColumn(args.oneCol, args.toRepStr, args.newStr) #('course1', 'cool', 'drool'))
    # df.printUniqueValuesInColumn(args.oneCol) #("course1") # to print the unique values after the substituteSubStrInColumn function

    # made sure all the required columns exist
    # df.dropMissingValuesInColumn("corr")
    # df.dropMissingValuesInColumn("P-value")
    # df.dropMissingValuesInColumn("course1")
    # df.dropMissingValuesInColumn("course2")
    # print(df.chordGraphByMajor(corr, pVal)) #(0.4, 1))

    # Trying to test this - START
    # graph = df.getNxGraph(None)

    # g = nx.to_dict_of_dicts(graph)
    # print(g)
    # print(plt.plot(df.getNxGraph())) # how to check graph?

    # print(df.getCliques())
    # df.outputCliqueDistribution()

    # df.printUniqueValuesInColumn(args.oneCol) #("corr")
    # df.makeMissingValuesNanInColumn(args.oneCol) #("corr") # Note: to test this, I made one of the rows have a space as the input, we can delete that testing purpose row later
    # df.printUniqueValuesInColumn(args.oneCol) #("corr")
    # df.removeNanInColumn(args.oneCol) #("corr")
    # df.printUniqueValuesInColumn(args.oneCol) #("corr") # to test if the row containing nan has been removed

    # df.printUniqueValuesInColumn(args.oneCol) #("corr") # check the values before the 'dropMissingValuesInColumn' function
    # df.dropMissingValuesInColumn(args.oneCol) #("corr")
    # df.printUniqueValuesInColumn(args.oneCol) #("corr") # check values after the 'dropMissingValuesInColumn' function

    # df.convertColumnToString(args.oneCol) #("corr")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvfile", default="Course-Correlation-Matrix-v1.csv", help="csv file with grade data")
    parser.add_argument("--oneCol", type=str, default="course1", help="the specified column for the function")
    parser.add_argument("--majorList", type=list, default=[], help="a list of majors")
    parser.add_argument("--classList", type=list, default=[], help="a list of classes")
    parser.add_argument("--bool1", type=bool, default=True, help="boolean to use for functions")
    parser.add_argument("--corr", type=float, default=0.4, help="boolean to use for functions")
    parser.add_argument("--pVal", type=float, default=1, help="boolean to use for functions")
    parser.add_argument("--toRepStr", type=str, default="", help="string to replace in the replace function")
    parser.add_argument("--newStr", type=str, default="", help="string that will replace toRepStr")
    parser.add_argument("-D", "--datadir", default="./", help="data directory")
    parser.add_argument("-e", "--edges", type=int, default=10, help="minimum edge value")
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
