import os
import sys
import argparse
import time
from edmlib import gradeData, classCorrelationData
from scipy.stats.stats import pearsonr


workingCols = ['OTCM_FinalGradeN', 'SID', 'REG_term', 'REG_CourseCrn',
           'REG_Programcode', 'REG_Numbercode', 'GRA_MajorAtGraduation',
           'REG_REG_credHr']


def main(args):
  sys.path.append(args.datadir)
  start_time = time.time()
  df = gradeData(os.path.join(args.datadir, args.csvfile))
  df.defineWorkingColumns(*args.cols)
  df.substituteSubStrInColumn('REG_Programcode', 'Computer and Info Science', 'CISC')
  df.substituteSubStrInColumn('REG_term', 'S', '0')
  df.substituteSubStrInColumn('REG_term', 'M', '1')
  df.substituteSubStrInColumn('REG_term', 'F', '2')
  csClasses = ['CISC1600', 'CISC2000', 'CISC2200', 'CISC4080', 'CISC4090', 'CISC3500', 'CISC3593', 'CISC3595', 'CISC4597', 'CISC4615', 'CISC4631']
  df.sankeyGraphByCourseTracks([['CISC1600'],
								['CISC2000'],
								['CISC2200', 'CISC4080', 'CISC4090'],
								['CISC2200', 'CISC4080', 'CISC4090'],
								['CISC2200', 'CISC4080', 'CISC4090']], 'CS Track Class Distribution', minEdgeValue=args.edges)

# Calls main after setting up program arguments
# Tested with: python edmlibTrackingDemo.py -D /u/erdos/edmProject cs-sample-6-17-19.csv
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("csvfile", help="csv file with grade data")
  parser.add_argument("-c", "--cols", type=list,default=workingCols,help="working columns in datamart") 
  parser.add_argument("-D", "--datadir", default="./", help="data directory")
  parser.add_argument("-e", "--edges", type=int, default=10, help="minimum edge value")
  args = parser.parse_args()
  main(args)

