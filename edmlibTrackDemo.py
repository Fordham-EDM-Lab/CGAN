import sys
import time
from edmlib import gradeData, classCorrelationData
from scipy.stats.stats import pearsonr

edmHome="/u/erdos/edmProject/"
outDir="./"
sys.path.append(edmHome)
start_time = time.time()
df = gradeData(edmHome + "final-datamart-6-7-19.csv")
# majorsToFilterTo = ['Computer and Info Science', 
#                     'Psychology']
# coreClasses = [ 'Philosophy1000',
#                 'Theology1000',
#                 'English1102',
#                 'English1101',
#                 'History1000',
#                 'Theology3200',
#                 'VisualArts1101',
#                 'Physics1201',
#                 'Chemistry1101']
# preMedClasses = ['BiologicalSciences1403',
#                  'BiologicalSciences1413',
#                  'BiologicalSciences1404',
#                  'BiologicalScience1414',
#                  'Chemistry1321',
#                  'Chemistry1331',
#                  'Chemistry1322',
#                  'Chemistry1332']
# otherClasses = ['Physics1501',
#                     'Physics1511',
#                     'Economics2140',
#                     'Mathematics1100',
#                     'Theatre1100',
#                     'Music1100']
df.defineWorkingColumns('OTCM_FinalGradeN', 'SID', 'REG_term', 
		'REG_CourseCrn', 'REG_Programcode', 'REG_Numbercode', 'GRA_MajorAtGraduation', 'REG_REG_credHr')
# df.outputGpaDistribution(True, 'gpaHistogram.html', 'Fordham GPA Distribution', 36)
df.substituteSubStrInColumn('REG_Programcode', 'Computer and Info Science', 'CISC')
df.substituteSubStrInColumn('REG_term', 'S', '0')
df.substituteSubStrInColumn('REG_term', 'M', '1')
df.substituteSubStrInColumn('REG_term', 'F', '2')
csClasses = ['CISC1600', 'CISC2000', 'CISC2200', 'CISC4080', 'CISC4090', 'CISC3500', 'CISC3593', 'CISC3595', 'CISC4597', 'CISC4615', 'CISC4631']
df.sankeyGraphByCourseTracks([['CISC1600'],
																['CISC2000'],
																['CISC2200', 'CISC4080', 'CISC4090'],
																['CISC2200', 'CISC4080', 'CISC4090'],
																['CISC2200', 'CISC4080', 'CISC4090']], 'CS Track Class Distribution', minEdgeValue=10)
