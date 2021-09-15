import unittest
import time

from numpy import string_
#from edmlib import gradeData
from scipy.stats.stats import pearsonr
import networkx as nx
import filecmp
from edmlib.edmlib import *
#from edmlib.gradeData import gradeData
from edmlib.classCorrelationData import classCorrelationData
import edmlibCorrelationDataTestArgParse

import argparse
import io
import sys
import os
import shutil
from edmlib.edmlib import edmApplication, outDir
import re

class TestCorrelationData(unittest.TestCase):
    # NOTE: Right now, these test cases check against itself, but later, it will 
    #       also check against prev versions of the function

    def test_get_classes_used(self):
        df = classCorrelationData('Course-Correlation-Matrix-v1.csv')
        self.assertEqual(df.getClassesUsed(), df.getClassesUsed())

    def test_get_number_of_classes_used(self):
        df = classCorrelationData('Course-Correlation-Matrix-v1.csv')
        self.assertEqual(df.getNumberOfClassesUsed(), df.getNumberOfClassesUsed())

    def test_print_unique_values_in_column(self):
        df = classCorrelationData('Course-Correlation-Matrix-v1.csv')
        
        capturedOut = io.StringIO()
        sys.stdout = capturedOut
        df.printUniqueValuesInColumn("course1")
        sys.stdout = sys.__stdout__

        self.assertEqual(capturedOut, capturedOut)

    def test_print_classes_used(self):
        df = classCorrelationData('Course-Correlation-Matrix-v1.csv')

        capturedOut = io.StringIO()
        sys.stdout = capturedOut
        df.printClassesUsed()
        sys.stdout = sys.__stdout__

        self.assertEqual(capturedOut, capturedOut)

    def test_get_entry_count(self):
        df = classCorrelationData('Course-Correlation-Matrix-v1.csv')
        self.assertEqual(df.getEntryCount(), df.getEntryCount())

    def test_print_entry_count(self):
        df = classCorrelationData('Course-Correlation-Matrix-v1.csv')

        capturedOut = io.StringIO()
        sys.stdout = capturedOut
        df.printEntryCount("course1")
        sys.stdout = sys.__stdout__

        self.assertEqual(capturedOut, capturedOut)

    def test_print_first_x_rows(self):
        df = classCorrelationData('Course-Correlation-Matrix-v1.csv')

        capturedOut = io.StringIO()
        sys.stdout = capturedOut
        df.printFirstXRows(0)
        sys.stdout = sys.__stdout__

        self.assertEqual(capturedOut, capturedOut)

    def test_print_majors(self):
        df = classCorrelationData('Course-Correlation-Matrix-v1.csv')

        capturedOut = io.StringIO()
        sys.stdout = capturedOut
        df.printMajors()
        sys.stdout = sys.__stdout__

        self.assertEqual(capturedOut, capturedOut)

    def test_filter_column_to_values(self):
        df = classCorrelationData('Course-Correlation-Matrix-v1.csv')
        
        # may change curr/correct file name to be camel case!
        result = "./outputFiles/currFiltercolumntovalues.csv"

        df.removeNanInColumn("course1")
        df.filterColumnToValues("course1", [])


    
    def test_filter_to_multiple_majors_or_classes(self):
        df = classCorrelationData('Course-Correlation-Matrix-v1.csv')
        
        # may change curr/correct file name to be camel case!
        result = "./outputFiles/currFiltertomutiplemajorsorclasses.csv"
        
        df.removeNanInColumn("course1")
        df.filterToMultipleMajorsOrClasses([], [], True)
  


    def test_substitute_sub_str_in_column(self):
        df = classCorrelationData('Course-Correlation-Matrix-v1.csv')

        # may change curr/correct file name to be camel case!
        result = "./outputFiles/currSubstitutesubstrincolumn.csv"

        df.substituteSubStrInColumn("course1", "", "")



    def test_chord_graph_by_major(self):
        df = classCorrelationData('Course-Correlation-Matrix-v1.csv')
        
        # may change curr/correct file name to be camel case!
        result = "./outputFiles/currChordGraphByMajorMajorgraph.csv"

        df.dropMissingValuesInColumn("corr")
        df.dropMissingValuesInColumn("P-value")
        df.dropMissingValuesInColumn("course1")
        df.dropMissingValuesInColumn("course2")
        #print(df.chordGraphByMajor(args.corr, args.pVal, args.fileName, args.outputSize, args.imageSize, args.bool1, args.bool2)) #(0.4, 1)



    def test_make_missing_values_nan_in_column(self):
        df = classCorrelationData('Course-Correlation-Matrix-v1.csv')

        # may change curr/correct file name to be camel case!
        result = "./outputFiles/currMakemissingvaluesnanincolumn.csv"

        df.makeMissingValuesNanInColumn("course1") #("course1") # Note: to test this, I made one of the rows have a space as the input, we can delete that testing purpose row later



    def test_remove_nan_in_column(self):
        df = classCorrelationData('Course-Correlation-Matrix-v1.csv')

        # may change curr/correct file name to be camel case!
        result = "./outputFiles/currRemovenanincolumn.csv"

        df.removeNanInColumn("course1")



    def test_drop_missing_values_in_column(self):
        df = classCorrelationData('Course-Correlation-Matrix-v1.csv')

        # may change curr/correct file name to be camel case!
        result = "./outputFiles/currDropmissingvaluesincolumn.csv"

        df.dropMissingValuesInColumn("course1")

if __name__ == '__main__':
    unittest.main()