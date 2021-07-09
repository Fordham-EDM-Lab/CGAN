import unittest
import filecmp
from edmlib import *

entries = 1000
inputFile = 'sample/sample-datamart-' + str(entries) + '.csv'

data = gradeData(inputFile)
data.defineWorkingColumns("OTCM_FinalGradeN", "SID", "REG_term", "REG_CourseCrn", "REG_Programcode", "REG_Numbercode", "GRA_MajorAtGraduation", "STU_ulevel", "REG_REG_credHr")
data2 = classCorrelationData(inputFile)


class TestCorrelationData(unittest.TestCase):
    def test_entries_num(self):
        """
        Test that it can return the number of entries
        """
        result = data2.getEntryCount()
        self.assertEqual(result, entries)


class TestGradeData(unittest.TestCase):
    def test_correlation_matrix(self):
        """
        Test that the correlation matrix generated is correct
        """
        exportFile = 'TempMatrix_' + str(entries) + '.csv'
        data.exportCorrelationsWithMinNSharedStudents(filename=exportFile)
        #correctFile is the file path of the supposedly correct output
        correctFile = '../CorrelationMatrix/Output Correlation Matrices/CorrelationOutput_' + str(entries) + '.csv'
        #correctFile = '../CorrelationMatrix/Output Correlation Matrices/CorrelationOutput_300.csv'
        testBool = filecmp.cmp(exportFile, correctFile, False)
        self.assertTrue(testBool)



if __name__ == '__main__':
    unittest.main()
