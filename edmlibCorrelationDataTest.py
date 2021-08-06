import time
from edmlib import gradeData
from scipy.stats.stats import pearsonr
import networkx as nx
from edmlib.edmlib import *
from edmlib.classCorrelationData import classCorrelationData

from edmlib.edmlib import edmApplication, outDir
import re
#import matplotlib.pyplot as plt

start_time = time.time()
df = classCorrelationData('Course-Correlation-Matrix-v1.csv') #('/u/erdos/edmProject/final-datamart-6-7-19.csv')

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


# print(df.getClassesUsed())
# print(df.getNumberOfClassesUsed())
# print(df.printUniqueValuesInColumn("course1"))
# print(df.printClassesUsed())
# print(df.getEntryCount())
# df.printEntryCount()
# df.printFirstXRows(2)
# print(df.printMajors())

# print(df.filterColumnToValues("corr", [0, 1]))
# print(df.exportCSV())

df.removeNanInColumn("course1")
print(df.filterToMultipleMajorsOrClasses([], ["ComputerandInfoScience4001"], False))
# print(df.getEntryCount()) # to check the remaining entry count after the filterToMultipleMajorsOrClasses function

# print(df.substituteSubStrInColumn('course1', 'cool', 'drool'))
# df.printUniqueValuesInColumn("course1") # to print the unique values after the substituteSubStrInColumn function

# df.dropMissingValuesInColumn("corr")
# df.dropMissingValuesInColumn("P-value")
# df.dropMissingValuesInColumn("course1")
# df.dropMissingValuesInColumn("course2")
# print(df.chordGraphByMajor(0.4, 1))

# Trying to test this - START
# graph = df.getNxGraph(None)

# g = nx.to_dict_of_dicts(graph)
# print(g)
# print(plt.plot(df.getNxGraph())) # how to check graph?

# print(df.getCliques())
# df.outputCliqueDistribution()

# df.printUniqueValuesInColumn("corr")
# df.makeMissingValuesNanInColumn("corr") # Note: to test this, I made one of the rows have a space as the input, we can delete that testing purpose row later
# df.printUniqueValuesInColumn("corr")
# df.removeNanInColumn("corr")
# df.printUniqueValuesInColumn("corr") # to test if the row containing nan has been removed

# df.printUniqueValuesInColumn("corr") # check the values before the 'dropMissingValuesInColumn' function
# df.dropMissingValuesInColumn("corr")
# df.printUniqueValuesInColumn("corr") # check values after the 'dropMissingValuesInColumn' function

# df.convertColumnToString("corr")

# why use >, not >= avgCorr, why not do <= avgPVal
# what does the index thing do - 

# df.filterToMultipleMajorsOrClasses(majorsToFilterTo, coreClasses + preMedClasses + otherClasses)
# df.filterByGpaDeviationMoreThan(0.2)
# df.defineWorkingColumns('OTCM_FinalGradeN', 'SID', 'REG_term',
#         'REG_CourseCrn', 'REG_Programcode', 'REG_Numbercode', 'GRA_MajorAtGraduation', 'REG_REG_credHr')
# print("--- %s seconds ---" % (time.time() - start_time))
# start_time = time.time()
# df.exportCorrelationsWithAvailableClasses()
# print("--- %s seconds ---" % (time.time() - start_time))
