import time
from edmlib import gradeData
from scipy.stats.stats import pearsonr

start_time = time.time()
df = gradeData()
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
df.reduceToMultipleMajorsOrClasses(majorsToFilterTo, coreClasses + preMedClasses + otherClasses)
df.filterByGpaDeviationMoreThan(0.2)
print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
df.exportCorrelationsWithAvailableClasses()
print("--- %s seconds ---" % (time.time() - start_time))