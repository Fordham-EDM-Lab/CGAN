
import time
import numpy as np
import pandas as pd
import csv
import sys
import math
from scipy.stats.stats import pearsonr
from scipy.stats import norm as sciNorm
import re, os
import networkx as nx
import itertools
import holoviews as hv
from holoviews import opts, dim
from bokeh.plotting import show, save, output_file
from bokeh.io import export_png
from bokeh.models import Title
from edmlib.edmlib import edmApplication, outDir

class gradeDataHelper:
          ######## Variables #########
  df = None 
  sourceFile = ""

  FINAL_GRADE_COLUMN = 'finalGrade'
  STUDENT_ID_COLUMN = 'studentID'
  FACULTY_ID_COLUMN = 'facultyID'
  CLASS_NUMBER_COLUMN = 'classNumber'
  CLASS_DEPT_COLUMN = 'classDept'
  TERM_COLUMN = 'term'
  CLASS_ID_COLUMN = 'classID'
  STUDENT_MAJOR_COLUMN = 'studentMajor'
  CLASS_CREDITS_COLUMN = 'classCredits'
  STUDENT_YEAR_COLUMN = 'studentYear'

  CLASS_ID_AND_TERM_COLUMN = 'courseIdAndTerm'
  CLASS_CODE_COLUMN = 'classCode'
  GPA_STDDEV_COLUMN = 'gpaStdDeviation'
  GPA_MEAN_COLUMN = 'gpaMean'
  NORMALIZATION_COLUMN = 'norm'
  GPA_NORMALIZATION_COLUMN = 'normByGpa'
  STUDENT_CLASS_NORMALIZATION_COLUMN = 'normByStudentByClass'

  ######## Methods #########
  def __init__(self,sourceFileOrDataFrame,copyDataFrame=True):
    """Class constructor, creates an instance of the class given a .CSV file or pandas dataframe.

    Used with gradeData('fileName.csv') or gradeData(dataFrameVariable).

    Args:
        sourceFileOrDataFrame (:obj:`object`): name of the .CSV file (extension included) in the same path or pandas dataframe variable. Dataframes are copied so as to not affect the original variable.

    """
    if type(sourceFileOrDataFrame).__name__ == 'str':
      self.sourceFile = sourceFileOrDataFrame
      self.df = pd.read_csv(self.sourceFile, dtype=str)

    elif type(sourceFileOrDataFrame).__name__ == 'DataFrame':
      # JH: Consider adding a bool = True to make a copy, set False if no copy.
      #      if not edmApplication:
      if copyDataFrame:
        self.df = sourceFileOrDataFrame.copy()
      else:
        self.df = sourceFileOrDataFrame

  def getColumn(self, column):
    """Returns a given column.

    Args:
        column (:obj:`str`): name of the column to return.

    Returns:
        :obj:`pandas.series`: column contained in pandas dataframe.

    """
    return self.df[column]

  def printColumn(self, column):
    """Prints the given column.

    Args:
        column (:obj:`str`): Column to print to console.
        
    """
    print(self.df[column])

  def defineWorkingColumns(self, finalGrade, studentID, term, classID = 'classID', classDept = 'classDept', classNumber = 'classNumber', studentMajor = 'studentMajor', studentYear = 'studentYear', classCredits = 'classCredits', facultyID = 'facultyID', classCode = 'classCode'):
    """Defines the column constants to target in the pandas dataframe for data manipulation. Required for proper functioning 
    of the library.

    Note:
        Either both :obj:`classDept` and :obj:`classNumber` variables need to be defined in the dataset's columns, or :obj:`classCode` needs to be defined for the library to function. The opposite variable(s) are then optional. :obj:`classDept` needs to be defined for major related functions to work.

    Args:
        finalGrade (:obj:`str`): Name of the column with grades given to a student in the respective class, grades are expected to be on a 1.0 - 4.0+ scale.
        studentID (:obj:`str`): Name of the column with student IDs, which does not need to follow any format.
        term (:obj:`str`): Name of the column with the term the class was taken, does not need to follow any format.
        classID (:obj:`str`): Number or string specific to a given class section, does not need to follow any format.
        classDept (:obj:`str`, optional): Name of the column stating the department of the class, e.g. 'Psych'. Defaults to 'classDept'. 
        classNumber (:obj:`str`, optional): Name of the column stating a number associated with the class, e.g. '1000' in 'Psych1000' or 'Intro to Psych'. Defaults to 'classNumber'.
        studentMajor (:obj:`str`, optional): Name of the column stating the major of the student. Optional, but required for functions involving student majors.
        studentYear (:obj:`str`, optional): Name of the column stating the year a student is in.
        classCredits (:obj:`str`, optional): Name of the column stating the number of credits a class is worth. Optional, but can be used to make student GPA calculations more accurate.
        facultyID (:obj:`str`, optional): Name of the column with faculty IDs, which does not need to follow any format. This is the faculty that taught the class. Optional, but required for instructor effectiveness functions.
        classCode (:obj:`str`, optional): Name of the column defining a class specific name, e.g. 'Psych1000'. Defaults to 'classCode'.

    """
    self.FINAL_GRADE_COLUMN = finalGrade
    self.STUDENT_ID_COLUMN = studentID
    self.FACULTY_ID_COLUMN = facultyID
    self.CLASS_ID_COLUMN = classID
    self.CLASS_DEPT_COLUMN = classDept
    self.CLASS_NUMBER_COLUMN = classNumber
    self.TERM_COLUMN = term
    self.CLASS_CODE_COLUMN = classCode
    self.STUDENT_MAJOR_COLUMN = studentMajor
    self.CLASS_CREDITS_COLUMN = classCredits
    self.STUDENT_YEAR_COLUMN = studentYear

  def makeMissingValuesNanInColumn(self, column, test_column=False):
    if test_column and not self.__requiredColumnPresent(column):
      return
    self.df[column].replace(' ', np.nan, inplace=True)

  def removeNanInColumn(self, column, test_column=False):
    if test_column and not self.__requiredColumnPresent(column):
      return
    self.df.dropna(subset=[column], inplace=True)
    self.df.reset_index(inplace = True, drop=True)

  def dropMissingValuesInColumn(self, column, test_column=False):
    """Removes rows in the dataset which have missing data in the given column.

      Args:
        column (:obj:`str`): Column to check for missing values in.

    """
    if test_column and not self.__requiredColumnPresent(column):
      return
    self.makeMissingValuesNanInColumn(column)
    self.removeNanInColumn(column)
    

  def convertColumnToNumeric(self, column, test_column=False):
    if test_column and not self.__requiredColumnPresent(column):
      return
    self.df[column] = pd.to_numeric(self.df[column])

  def dropNullAndConvertToNumeric(self, column,test_column=False):
    self.dropMissingValuesInColumn(column,test_column)
    self.convertColumnToNumeric(column,test_column)

  def convertColumnToString(self, column,test_column=False):
    if test_column and not self.__requiredColumnPresent(column):
      return
    self.df.astype({column:str}, copy=False)

  def termMapping(self, mapping):
    self.df['termOrder'] = self.df[self.TERM_COLUMN].map(mapping)


  def getUniqueIdentifiersForSectionsAcrossTerms(self):
    """Used internally. If a column 'classCode' is unavailable, a new column is made by combining 'classDept' and 
    'classNumber' columns. Also makes a new 'classIdAndTerm' column by combining the 'classID' and 'term' columns,
    to differentiate specific class sections in specific terms.

    """
    if self.CLASS_CODE_COLUMN not in self.df.columns:
      if not self.__requiredColumnPresent(self.CLASS_DEPT_COLUMN):
        print("Note: Optionally, the 'classDept' column does not need to be defined if the class specific (e.g. 'Psych1000' or 'IntroToPsych') column 'classCode' is defined. This can be done with the 'defineWorkingColumns' function. This does, however, break department or major specific course functions.")
        return
      if not self.__requiredColumnPresent(self.CLASS_NUMBER_COLUMN):
        print("Note: Optionally, the 'classNumber' column does not need to be defined if the class specific (e.g. 'Psych1000' or 'IntroToPsych') column 'classCode' is defined. This can be done with the 'defineWorkingColumns' function.")
        return
      self.df[self.CLASS_CODE_COLUMN] = self.df[self.CLASS_DEPT_COLUMN].apply(str) + self.df[self.CLASS_NUMBER_COLUMN].apply(str)
      self.df[self.CLASS_CODE_COLUMN] = self.df[self.CLASS_CODE_COLUMN].str.replace(" ","")
    if self.CLASS_ID_AND_TERM_COLUMN not in self.df.columns:
      if not self.__requiredColumnPresent(self.CLASS_ID_COLUMN):
        return
      if not self.__requiredColumnPresent(self.TERM_COLUMN):
        return
      self.df[self.CLASS_ID_AND_TERM_COLUMN] = self.df[self.CLASS_ID_COLUMN].apply(str) + self.df[self.TERM_COLUMN]
      self.df[self.CLASS_ID_AND_TERM_COLUMN] = self.df[self.CLASS_ID_AND_TERM_COLUMN].str.replace(" ","")

  def getDictOfStudentMajors(self):
    """Returns a dictionary of students and their latest respective declared majors. Student ID, Student Major, and Term columns are required.

      Args: 
        N/A

      Returns:
        :obj:`dict`(:obj:`str` : :obj:`str`): Dictionary of students and their latest respective declared majors.

    """
    check = [self.__requiredColumnPresent(self.STUDENT_ID_COLUMN), self.__requiredColumnPresent(self.STUDENT_MAJOR_COLUMN), self.__requiredColumnPresent(self.TERM_COLUMN)]
    if not all(check):
      return
    lastEntries = self.df.sort_values(self.TERM_COLUMN)
    lastEntries.drop_duplicates(self.STUDENT_ID_COLUMN, keep='last', inplace=True)
    return pd.Series(lastEntries[self.STUDENT_MAJOR_COLUMN].values,index=df[self.STUDENT_ID_COLUMN]).to_dict()

  #This function is used when directed is False
  def corrAlg(self, a, b, nSharedStudents, directed, classDetails, sequenceDetails, semsBetweenClassesLimit): 
    """
    Args: a and b, dataframe selections for a particular class "A" and "B"
    See getCorrelationsWithMinNSharedStudents above for remainder

    Returns: list including correlation, p value, and length of normalized grades of students who took both A and B 

    """
    #norms is a DataFrame that includes data from a of students in b
    norms = a.loc[a[self.STUDENT_ID_COLUMN].isin(b[self.STUDENT_ID_COLUMN].values)]
    #Remove missing values
    norms = norms.dropna(subset=[self.NORMALIZATION_COLUMN])
    if (semsBetweenClassesLimit >= 0):
      #Combine dataframes into new dataframe combinedClasses to calculate the gap between classes
      combinedClasses = norms.set_index("SID").join(b.set_index("SID"), lsuffix = "_a", rsuffix = "_b")
      combinedClasses["semDifference"] = abs(combinedClasses["semNumber_a"] - combinedClasses["semNumber_b"])
      norms["semDifference"] = combinedClasses["semDifference"].values
      #Filter data to only include students with a small enough gap between classes
      norms = norms.loc[norms.semDifference <= semsBetweenClassesLimit]
    if len(norms) >= nSharedStudents:
      #Set the index of norms to Student ID and sort
      norms.set_index(self.STUDENT_ID_COLUMN, inplace=True)
      norms.sort_index(inplace=True)
      ##Set norms2 to be a dataframe of the data from class b of students who took class a before class b
      norms2 = b.loc[b[self.STUDENT_ID_COLUMN].isin(norms.index)]
      #Set the index of norms2 to Student ID and sort
      norms2.set_index(self.STUDENT_ID_COLUMN, inplace=True)
      norms2.sort_index(inplace=True)
      #Use pearsonr to get correlation and p-value
      corr, Pvalue = pearsonr(norms[self.NORMALIZATION_COLUMN], norms2[self.NORMALIZATION_COLUMN])
      return [corr, Pvalue, len(norms.index)]
    #Return math.nan for each value if there aren't enough students
    else:
      return [math.nan, math.nan, math.nan]

  def corrAlgDirected(self, a, b, nSharedStudents, directed, classDetails, sequenceDetails, semsBetweenClassesLimit): 
    """
    Args: a and b, dataframe selections for a particular class "A" and "B"
    See getCorrelationsWithMinNSharedStudents above for remainder

    Returns: res, a list which contains at least: 
    [corr, Pvalue, len(aNorms),                             //correlation, pvalue, and number of values for normalized grades in class A
    corr1, Pvalue1, len(abANorms),                          //above for students who took class a prior to b
    corr2, Pvalue2, len(baANorms),                          //above for students who took class b prior to a
    corr3, Pvalue3, len(concANorms),                        //above for student who took classes a and b concurrently
    AGrd, BGrd, AstdDevGrd, BstdDevGrd, ANrm, BNrm,         //grades in A and B along with their stdevs and normalizations
    AstdDevNrm, BstdDevNrm, ABASDGrd, ABBSDGrd, BAASDGrd, BABSDGrd, ABASDNrm, ABBSDNrm, BAASDNrm, BABSDNrm]
    //final row is various additional stdev and normalization data
    res contains additional data if classDetails or sequenceDetails
    """
    #norms is a DataFrame that includes data from a of students in b
    norms = a.loc[a[self.STUDENT_ID_COLUMN].isin(b[self.STUDENT_ID_COLUMN].values)]
    #Remove all data with missing grades
    norms = norms.dropna(subset=[self.NORMALIZATION_COLUMN])
    if (semsBetweenClassesLimit >= 0 and len(norms) >= nSharedStudents):
      #Combine dataframes into new dataframe combinedClasses to calculate the gap between classes
      combinedClasses = norms.set_index("SID").join(b.set_index("SID"), lsuffix = "_a", rsuffix = "_b")
      combinedClasses["semDifference"] = abs(combinedClasses["semNumber_a"] - combinedClasses["semNumber_b"])
      norms["semDifference"] = combinedClasses["semDifference"].values
      #Filter data to only include students with a small enough gap between classes
      norms = norms.loc[norms.semDifference <= semsBetweenClassesLimit]
    #Return no data if there aren't enough students
    if len(norms) < nSharedStudents and not sequenceDetails:
      return ([math.nan] * 36)
    elif len(norms) < nSharedStudents:
      return ([math.nan] * 52)
    #Set index to student ID and sort
    norms.set_index(self.STUDENT_ID_COLUMN, inplace=True)
    norms.sort_index(inplace=True)
    #aNorms is a series of normalized grades in class a of students who took class a before class b
    aNorms = norms[self.NORMALIZATION_COLUMN]
    #Set norms2 to be a dataframe of the data from class b of students who took class a before class b
    norms2 = b.loc[b[self.STUDENT_ID_COLUMN].isin(norms.index)]
    #Set index to student ID and sort
    norms2.set_index(self.STUDENT_ID_COLUMN, inplace=True)
    norms2.sort_index(inplace=True)
    #Define more, less, concurrentA, and concurrentB to create columns in the DataFrame
    #less: A taken before B
    #more: B taken before A
    #concurrent A and concurrent B: A and B taken at the same time
    if np.issubdtype(norms[self.TERM_COLUMN].dtype, np.number) and np.issubdtype(norms2[self.TERM_COLUMN].dtype, np.number) and numLibInstalled:
      n = norms[self.TERM_COLUMN].values
      m = norms2[self.TERM_COLUMN].values
      less = numexpr.evaluate('(n < m)')
      more = numexpr.evaluate('(n > m)')
      same = numexpr.evaluate('(n == m)')
      concurrentA = norms.loc[same]
      concurrentB = norms2.loc[same]
    #The same calculation but without numexpr (presumably is slower)
    else:
      less = norms["semNumber"].values < norms2["semNumber"].values
      more = norms["semNumber"].values > norms2["semNumber"].values
      concurrentA = norms.loc[(~less) & (~more)]
      concurrentB = norms2.loc[(~less) & (~more)]
    #Here all of the columns are created as lists
    #In aToBA, the aToB means that we're looking at students who took course A before course B, the A at the end means we're looking at grades in class A
    #The same naming scheme applies to all of the columns below
    aToBA = norms.loc[less]
    bToAA = norms.loc[more]
    aToBB = norms2.loc[less]
    bToAB = norms2.loc[more]
    bNorms = norms2[self.NORMALIZATION_COLUMN].dropna()
    #abB is used here to mean aToBB
    #So essentially this is normalized grades in b from students who took class a before class b
    #The same naming scheme applies to all of the columns below
    #conc can be used instead of ab or ba if the classes are taken concurrently
    abBNorms = aToBB[self.NORMALIZATION_COLUMN].dropna()
    baBNorms = bToAB[self.NORMALIZATION_COLUMN].dropna()
    concBNorms = concurrentB[self.NORMALIZATION_COLUMN].dropna()
    abANorms = aToBA[self.NORMALIZATION_COLUMN].dropna()
    baANorms = bToAA[self.NORMALIZATION_COLUMN].dropna()
    concANorms = concurrentA[self.NORMALIZATION_COLUMN].dropna()
    AstdDevGrd = norms[self.FINAL_GRADE_COLUMN].std()
    BstdDevGrd = norms2[self.FINAL_GRADE_COLUMN].std()
    AstdDevNrm = norms[self.NORMALIZATION_COLUMN].std()
    BstdDevNrm = norms2[self.NORMALIZATION_COLUMN].std()
    ABASDGrd = aToBA[self.FINAL_GRADE_COLUMN].std()
    ABBSDGrd = aToBB[self.FINAL_GRADE_COLUMN].std()
    BAASDGrd = bToAA[self.FINAL_GRADE_COLUMN].std()
    BABSDGrd = bToAB[self.FINAL_GRADE_COLUMN].std()
    ABASDNrm = aToBA[self.NORMALIZATION_COLUMN].std()
    ABBSDNrm = aToBB[self.NORMALIZATION_COLUMN].std()
    BAASDNrm = bToAA[self.NORMALIZATION_COLUMN].std()
    BABSDNrm = bToAB[self.NORMALIZATION_COLUMN].std()
    AGrd = norms[self.FINAL_GRADE_COLUMN].mean()
    BGrd = norms2[self.FINAL_GRADE_COLUMN].mean()
    ANrm = norms[self.NORMALIZATION_COLUMN].mean()
    BNrm = norms2[self.NORMALIZATION_COLUMN].mean()
    if classDetails:
      abAMean = aToBA[self.FINAL_GRADE_COLUMN].mean()
      abANormMean = aToBA[self.NORMALIZATION_COLUMN].mean()
      baAMean = bToAA[self.FINAL_GRADE_COLUMN].mean()
      baANormMean = bToAA[self.NORMALIZATION_COLUMN].mean()
      concAMean = concurrentA[self.FINAL_GRADE_COLUMN].mean()
      concANormMean = concurrentA[self.NORMALIZATION_COLUMN].mean()
      abBMean = aToBB[self.FINAL_GRADE_COLUMN].mean()
      abBNormMean = aToBB[self.NORMALIZATION_COLUMN].mean()
      baBMean = bToAB[self.FINAL_GRADE_COLUMN].mean()
      baBNormMean = bToAB[self.NORMALIZATION_COLUMN].mean()
      concBMean = concurrentB[self.FINAL_GRADE_COLUMN].mean()
      concBNormMean = concurrentB[self.NORMALIZATION_COLUMN].mean()
      #Calculate the mean GPA for each course order
      #Note that the students will have the same gpa regardless of being in class a or b, so only 2 calculations are needed
      abAGPAMean = aToBA["studentGPA"].mean()
      baBGPAMean = bToAB["studentGPA"].mean()
    
    if sequenceDetails:
      #returns list of list of bools of whether or not each student is a freshman, sophomore, junior, and senior
      def yearTruths(x):
        try:
          return [numexpr.evaluate('(x == 1)'), numexpr.evaluate('(x == 2)'), numexpr.evaluate('(x == 3)'), numexpr.evaluate('(x == 4)')]
        except NameError:
          freshmen = []
          for i in x:
            freshmen.append((i == "First-Time Freshman") or (i == "Continuing Freshman"))
          return [freshmen, x == "Sophomores", x == "Juniors", x == "Seniors"]
      
      c = aToBA[self.STUDENT_YEAR_COLUMN].values
      d = aToBB[self.STUDENT_YEAR_COLUMN].values
      e = bToAA[self.STUDENT_YEAR_COLUMN].values
      f = bToAB[self.STUDENT_YEAR_COLUMN].values
      
      aToBAFreshT, aToBASophT, aToBAJunT, aToBASenT = yearTruths(c)
      aToBBFreshT, aToBBSophT, aToBBJunT, aToBBSenT = yearTruths(d)
      bToAAFreshT, bToAASophT, bToAAJunT, bToAASenT = yearTruths(e)
      bToABFreshT, bToABSophT, bToABJunT, bToABSenT = yearTruths(f)
      crs1FreshMin = min(sum(aToBAFreshT), sum(bToAAFreshT))
      crs2FreshMin = min(sum(aToBBFreshT), sum(bToABFreshT))
      crs1SophMin = min(sum(aToBASophT), sum(bToAASophT))
      crs2SophMin = min(sum(aToBBSophT), sum(bToABSophT))
      crs1JunMin = min(sum(aToBAJunT), sum(bToAAJunT))
      crs2JunMin = min(sum(aToBBJunT), sum(bToABJunT))
      crs1SenMin = min(sum(aToBASenT), sum(bToAASenT))
      crs2SenMin = min(sum(aToBBSenT), sum(bToABSenT))
      aToBAFresh = aToBA.loc[aToBAFreshT]
      aToBASoph = aToBA.loc[aToBASophT]
      aToBAJun = aToBA.loc[aToBAJunT]
      aToBASen = aToBA.loc[aToBASenT]
      aToBBFresh = aToBB.loc[aToBBFreshT]
      aToBBSoph = aToBB.loc[aToBBSophT]
      aToBBJun = aToBB.loc[aToBBJunT]
      aToBBSen = aToBB.loc[aToBBSenT]
      bToAAFresh = bToAA.loc[bToAAFreshT]
      bToAASoph = bToAA.loc[bToAASophT]
      bToAAJun = bToAA.loc[bToAAJunT]
      bToAASen = bToAA.loc[bToAASenT]
      bToABFresh = bToAB.loc[bToABFreshT]
      bToABSoph = bToAB.loc[bToABSophT]
      bToABJun = bToAB.loc[bToABJunT]
      bToABSen = bToAB.loc[bToABSenT]
      nrmAlias, grdAlias = self.NORMALIZATION_COLUMN, self.FINAL_GRADE_COLUMN
      avNormDifCrs2Fresh = (aToBBFresh[nrmAlias].mean() - bToABFresh[nrmAlias].mean()) if crs2FreshMin > 0 else np.nan
      avNormDifCrs1Fresh = (aToBAFresh[nrmAlias].mean() - bToAAFresh[nrmAlias].mean()) if crs1FreshMin > 0 else np.nan
      avNormDifCrs2Soph = (aToBBSoph[nrmAlias].mean() - bToABSoph[nrmAlias].mean()) if crs2SophMin > 0 else np.nan
      avNormDifCrs1Soph = (aToBASoph[nrmAlias].mean() - bToAASoph[nrmAlias].mean()) if crs1SophMin > 0 else np.nan
      avNormDifCrs2Jun = (aToBBJun[nrmAlias].mean() - bToABJun[nrmAlias].mean()) if crs2JunMin > 0 else np.nan
      avNormDifCrs1Jun = (aToBAJun[nrmAlias].mean() - bToAAJun[nrmAlias].mean()) if crs1JunMin > 0 else np.nan
      avNormDifCrs2Sen = (aToBBSen[nrmAlias].mean() - bToABSen[nrmAlias].mean()) if crs2SenMin > 0 else np.nan
      avNormDifCrs1Sen = (aToBASen[nrmAlias].mean() - bToAASen[nrmAlias].mean()) if crs1SenMin > 0 else np.nan
      avGradeDifCrs2Fresh = (aToBBFresh[grdAlias].mean() - bToABFresh[grdAlias].mean()) if crs2FreshMin > 0 else np.nan
      avGradeDifCrs1Fresh = (aToBAFresh[grdAlias].mean() - bToAAFresh[grdAlias].mean()) if crs1FreshMin > 0 else np.nan
      avGradeDifCrs2Soph = (aToBBSoph[grdAlias].mean() - bToABSoph[grdAlias].mean()) if crs2SophMin > 0 else np.nan
      avGradeDifCrs1Soph = (aToBASoph[grdAlias].mean() - bToAASoph[grdAlias].mean()) if crs1SophMin > 0 else np.nan
      avGradeDifCrs2Jun = (aToBBJun[grdAlias].mean() - bToABJun[grdAlias].mean()) if crs2JunMin > 0 else np.nan
      avGradeDifCrs1Jun = (aToBAJun[grdAlias].mean() - bToAAJun[grdAlias].mean()) if crs1JunMin > 0 else np.nan
      avGradeDifCrs2Sen = (aToBBSen[grdAlias].mean() - bToABSen[grdAlias].mean()) if crs2SenMin > 0 else np.nan
      avGradeDifCrs1Sen = (aToBASen[grdAlias].mean() - bToAASen[grdAlias].mean()) if crs1SenMin > 0 else np.nan

      #Calculate correlation and p-value of normalized grades
      corr, Pvalue = pearsonr(bNorms, aNorms)
      corr1, Pvalue1 = math.nan, math.nan
      corr2, Pvalue2 = math.nan, math.nan
      corr3, Pvalue3 = math.nan, math.nan
      #Calculate correlation and p-value of normalized grades when A is taken before, after, and concurrent to B
      if len(abANorms) >= 2:
        corr1, Pvalue1 = pearsonr(abBNorms,abANorms)
      if len(baANorms) >= 2:
        corr2, Pvalue2 = pearsonr(baBNorms,baANorms)
      if len(concANorms) >= 2:
        corr3, Pvalue3 = pearsonr(concBNorms,concANorms)
      
      #res is a list of all the data calculated above
      res = [corr, Pvalue, len(aNorms), corr1, Pvalue1, len(abANorms), corr2, Pvalue2, 
        len(baANorms), corr3, Pvalue3, len(concANorms), AGrd, BGrd, AstdDevGrd, BstdDevGrd, ANrm, BNrm, 
        AstdDevNrm, BstdDevNrm, ABASDGrd, ABBSDGrd, BAASDGrd, BABSDGrd, ABASDNrm, ABBSDNrm, BAASDNrm, BABSDNrm]

      #add more data if using classDetails or sequenceDetails
      if classDetails:
        res += [abAMean, abANormMean, abBMean, abBNormMean, baAMean, baANormMean, 
        baBMean, baBNormMean, concAMean, concANormMean, concBMean, concBNormMean, abAGPAMean, baBGPAMean]
      if sequenceDetails:
        res += [avNormDifCrs1Fresh, avNormDifCrs2Fresh, avNormDifCrs1Soph, avNormDifCrs2Soph, 
                avNormDifCrs1Jun, avNormDifCrs2Jun, avNormDifCrs1Sen, avNormDifCrs2Sen,
                avGradeDifCrs1Fresh, avGradeDifCrs2Fresh, avGradeDifCrs1Soph, avGradeDifCrs2Soph, 
                avGradeDifCrs1Jun, avGradeDifCrs2Jun, avGradeDifCrs1Sen, avGradeDifCrs2Sen,
                crs1FreshMin, crs2FreshMin, crs1SophMin, crs2SophMin, crs1JunMin, crs2JunMin, 
                crs1SenMin, crs2SenMin]
      #return the list of data
      return res

  def getListOfClassCodes(self):
    """Returns a list of unique class codes currently in the dataset from the 'classCode' column, which is the conjoined 'classDept' and 'classNumber' columns by default (e.g. 'Psych1000' from 'Psych' and '1000').

    Returns:
        :obj:`list`: List of unique class codes in the dataset.

    """
    if self.CLASS_CODE_COLUMN not in self.df.columns:
      self.getUniqueIdentifiersForSectionsAcrossTerms()
    if self.CLASS_CODE_COLUMN not in self.df.columns:
      return
    return self.getUniqueValuesInColumn(self.CLASS_CODE_COLUMN)

  def getNormalizationColumn(self):
    """Used internally. Creates a normalization column 'norm' that is a "normalization" of grades recieved in specific classes. 
    This is equivelant to the grade given to a student minus the mean grade in that class, all divided by the standard 
    deviation of grades in that class.
    
    """
    print('Getting normalization column...')
    if self.CLASS_CODE_COLUMN not in self.df.columns:
      self.getUniqueIdentifiersForSectionsAcrossTerms()
    if self.GPA_STDDEV_COLUMN not in self.df.columns:
      self.getGPADeviations()
    if self.GPA_MEAN_COLUMN not in self.df.columns:
      self.getGPAMeans()
    if not all(item in self.df.columns for item in [self.CLASS_CODE_COLUMN, self.GPA_STDDEV_COLUMN, self.GPA_MEAN_COLUMN]):
      return
    begin = self.getEntryCount()
    self.filterByGpaDeviationMoreThan(0.001)
    self.df[self.NORMALIZATION_COLUMN] = (self.df[self.FINAL_GRADE_COLUMN].values - self.df[self.GPA_MEAN_COLUMN].values) / self.df[self.GPA_STDDEV_COLUMN].values

  def exportNormalizationColumn(self, fileName = 'normalized.csv'):
    """Function to export a csv file that is a view of the original DataFrame after being normalized
    
    Args: fileName: name of output file

    Return: export csv file with normalized final grade
    """
    self.getNormalizationColumn()
    # select column
    view = self.df[[self.STUDENT_ID_COLUMN, self.CLASS_ID_AND_TERM_COLUMN, self.FINAL_GRADE_COLUMN, \
                    self.GPA_MEAN_COLUMN, self.NORMALIZATION_COLUMN]]
    # export to csv
    if not fileName.endswith('.csv'):
      fileName = "".join((fileName, '.csv'))
    view.to_csv(fileName, index=False)

  def getDictOfStudentGPAs(self, getStdDev = False):
    self.dropNullAndConvertToNumeric(self.FINAL_GRADE_COLUMN)
    if self.CLASS_CREDITS_COLUMN in self.df.columns:
      self.dropNullAndConvertToNumeric(self.CLASS_CREDITS_COLUMN)
      temp = self.df.loc[:, [self.STUDENT_ID_COLUMN, self.CLASS_CREDITS_COLUMN, self.FINAL_GRADE_COLUMN]]
      temp['classPoints'] = temp[self.CLASS_CREDITS_COLUMN] * temp[self.FINAL_GRADE_COLUMN]
      temp2 = temp.groupby(self.STUDENT_ID_COLUMN, as_index=False)
      sums = temp2.sum()
      sums['gpa'] = sums['classPoints'] / sums[self.CLASS_CREDITS_COLUMN]
      # print(temp2)
      # print(sums)
      gpas = dict(zip(sums[self.STUDENT_ID_COLUMN], sums['gpa']))
      if getStdDev:
        # temp2.apply(lambda x: print(x))
        sums['stddev'] = temp2.apply(lambda x: np.average((x[self.FINAL_GRADE_COLUMN]- \
                        gpas[x[self.STUDENT_ID_COLUMN].iloc[0]])**2, weights=x[self.CLASS_CREDITS_COLUMN]))
        stdDevs = dict(zip(sums[self.STUDENT_ID_COLUMN], sums['stddev']))
        # np.average(subEntries['normBenefit'].astype(float), weights=subEntries[weighting].astype(float))
      # print(sums)
    else:
      temp = self.df.loc[:, [self.STUDENT_ID_COLUMN, self.FINAL_GRADE_COLUMN]]
      temp2 = temp.groupby(self.STUDENT_ID_COLUMN, as_index=False)
      means = temp2.mean()
      gpas = dict(zip(means[self.STUDENT_ID_COLUMN], means[self.FINAL_GRADE_COLUMN]))
      if getStdDev:
        gpas['stds'] = temp2.apply(lambda x: np.average((x[self.FINAL_GRADE_COLUMN]-gpas[x[self.STUDENT_ID_COLUMN].iloc[0]])**2))
        stdDevs = dict(zip(temp2[self.STUDENT_ID_COLUMN], gpas['stds']))
    
    if getStdDev:
      return (gpas, stdDevs)
    return gpas

  def getNormalizationByGPA(self):
    if not self.__requiredColumnPresent(self.STUDENT_ID_COLUMN):
      return
    if not self.__requiredColumnPresent(self.FINAL_GRADE_COLUMN):
      return
    
    gpas, stds = self.getDictOfStudentGPAs(getStdDev=True)
    temp = self.df.loc[:,[self.STUDENT_ID_COLUMN, self.FINAL_GRADE_COLUMN]]
    def rowOp(row):
      try:
        res = (row[self.FINAL_GRADE_COLUMN] - gpas[row[self.STUDENT_ID_COLUMN]]) / stds[row[self.STUDENT_ID_COLUMN]]
        return res
      except ZeroDivisionError:
        return math.nan
    self.df[self.GPA_NORMALIZATION_COLUMN] = temp.apply(rowOp, axis = 1)

  def getNormalizationByStudentByClass(self):
    if not self.__requiredColumnPresent(self.STUDENT_ID_COLUMN):
      return
    if not self.__requiredColumnPresent(self.FINAL_GRADE_COLUMN):
      return
    if self.NORMALIZATION_COLUMN not in self.df.columns:
      self.getNormalizationColumn()
      if not self.__requiredColumnPresent(self.NORMALIZATION_COLUMN):
        return
    self.dropNullAndConvertToNumeric(self.NORMALIZATION_COLUMN)
    temp = self.df.loc[:,[self.STUDENT_ID_COLUMN, self.NORMALIZATION_COLUMN]]
    temp2 = temp.groupby(self.STUDENT_ID_COLUMN, as_index=False)
    means = temp2.mean()
    meanDict = dict(zip(means[self.STUDENT_ID_COLUMN], means[self.NORMALIZATION_COLUMN]))
    print(means)
    # print(meanDict)
    # print(temp2)
    stds = temp2.apply(lambda x: np.average((x[self.NORMALIZATION_COLUMN]-meanDict[x[self.STUDENT_ID_COLUMN].iloc[0]])**2))
    stds.columns = [self.NORMALIZATION_COLUMN, 'stds']
    means['stds'] = stds['stds']
    print(means)
    stdDict = dict(zip(means[self.STUDENT_ID_COLUMN], means['stds']))
    def rowOp(row):
      try:
        res = (row[self.NORMALIZATION_COLUMN] - meanDict[row[self.STUDENT_ID_COLUMN]]) / stdDict[row[self.STUDENT_ID_COLUMN]]
        return res
      except ZeroDivisionError:
        return math.nan
    self.df[self.STUDENT_CLASS_NORMALIZATION_COLUMN] = temp.apply(rowOp, axis = 1)

  def exportCompoundNormalization(self, fileName = 'compoundNormalized.csv'):
    """Function to export a csv file that is a view of the original DataFrame after being normalized
    
    Args: fileName: name of output file

    Return: export csv file with normalized final grade
    """
    self.getNormalizationByStudentByClass()
    # select column
    view = self.df[[self.STUDENT_ID_COLUMN, self.CLASS_ID_AND_TERM_COLUMN, self.FINAL_GRADE_COLUMN, \
                    self.GPA_MEAN_COLUMN, self.NORMALIZATION_COLUMN]]
    # export to csv
    if not fileName.endswith('.csv'):
      fileName = "".join((fileName, '.csv'))
    view.to_csv(fileName, index=False)

  def getGradesByYear(self):
    if self.NORMALIZATION_COLUMN not in self.df.columns:
      self.getNormalizationColumn()
    if self.NORMALIZATION_COLUMN not in self.df.columns:
      print('Error: Could not find normalization column.')
      return
    if not self.__requiredColumnPresent(self.STUDENT_YEAR_COLUMN):
      return
    years = sorted(self.df[self.STUDENT_YEAR_COLUMN].unique().tolist())
    classes = self.df[self.CLASS_CODE_COLUMN].unique().tolist()
    # grouped = self.df.groupby(self.STUDENT_YEAR_COLUMN)
    # temp = self.df.loc[:, [self.CLASS_ID_AND_TERM_COLUMN, self.FINAL_GRADE_COLUMN]]
    # meanGpas = temp.groupby(self.CLASS_ID_AND_TERM_COLUMN).mean()
    # meanGpas.rename(columns={self.FINAL_GRADE_COLUMN:self.GPA_MEAN_COLUMN}, inplace=True)
    # self.df = pd.merge(self.df,meanGpas, on=self.CLASS_ID_AND_TERM_COLUMN)
    dictList = []
    for year in years:
      yearGroup = self.df.loc[self.df[self.STUDENT_YEAR_COLUMN] == year]
      yearDict = {}
      print(year)
      counter = 0
      count = len(classes)
      for course in classes:
        counter += 1
        print(str(counter) + '/' + str(count) + ' classes')
        classYearGroup = yearGroup[self.CLASS_CODE_COLUMN] == course
        if any(classYearGroup):
          classGroup = yearGroup.loc[classYearGroup]
          yearDict[course] = classGroup[self.NORMALIZATION_COLUMN].mean()
        else:
          yearDict[course] = np.nan
      dictList.append(yearDict)

    def getMean(year, course):
      return dictList[years.index(year)][course]
    for item in years:
      self.df[item+'Norm'] = self.df.apply(lambda row: getMean(item, row[self.CLASS_CODE_COLUMN]), axis=1)


  def filterColumnToValues(self, col, values = []):
    """Filters dataset to only include rows that contain any of the given values in the given column.

    Args:
        col (:obj:`str`): Name of the column to filter.
        values (:obj:`list`): Values to filter to.
        
    """
    if not self.__requiredColumnPresent(col):
        return
    self.printEntryCount()
    if all([isinstance(x,str) for x in values]):
      lowered = [x.lower() for x in values]
      possibilities = "|".join(lowered)
      loweredCol = self.df[col].str.lower()
      self.df = self.df.loc[loweredCol.str.contains(possibilities)]
    else:
      self.df = self.df.loc[np.in1d(self.df[col],values)]
    self.df.reset_index(inplace=True, drop=True)
    self.printEntryCount()

  def filterToMultipleMajorsOrClasses(self, majors = [], classes = []):
    """Reduces the dataset to only include entries of certain classes and/or classes in certain majors. This function is 
    inclusive; if a class in 'classes' is not of a major defined in 'majors', the class will still be included, and 
    vice-versa.

    Note:
        The 'classDept' column as set by :obj:`defineWorkingColumns` must be defined in your dataset to filter by major.

    Args:
        majors (:obj:`list`, optional): List of majors to include. Filters by the 'classDept' column.
        classes (:obj:`list`, optional): List of classes to include. Filters by the 'classCode' column, or the conjoined version of 'classDept' and 'classNumber' columns.

    """
    if self.CLASS_CODE_COLUMN not in self.df.columns:
      self.getUniqueIdentifiersForSectionsAcrossTerms()
    if self.CLASS_CODE_COLUMN not in self.df.columns:
      return
    self.printEntryCount()
    self.df = self.df.loc[(np.in1d(self.df[self.CLASS_CODE_COLUMN], classes)) | (np.in1d(self.df[self.CLASS_DEPT_COLUMN], majors))]
    self.df.reset_index(inplace=True, drop=True)
    self.printEntryCount()
  
  def filterStudentsByMajors(self, majors):
    """Filters the dataset down to students who were ever recorded as majoring in one of the given majors.

    Args:
      majors (:obj:`list`, optional): List of student majors to include when finding matching students. Filters by the 'studentMajor' column.

    """
    students = self.getUniqueValuesInColumn(self.STUDENT_ID_COLUMN)
    matchingRecords = self.df.loc[self.df[self.STUDENT_MAJOR_COLUMN].isin(majors)]
    validStudents = matchingRecords[self.STUDENT_ID_COLUMN].unique()
    self.df = self.df.loc[self.df[self.STUDENT_ID_COLUMN].isin(validStudents)]
    self.df.reset_index(inplace=True, drop=True)

  def getGPADeviations(self):
    """Used internally. Makes a new column called 'gpaStdDeviation', the standard deviation of grades of the respective 
    class for each entry.
    
    """
    print('Getting grade deviations by class...')
    if not self.__requiredColumnPresent(self.FINAL_GRADE_COLUMN):
        return
    self.dropNullAndConvertToNumeric(self.FINAL_GRADE_COLUMN)
    if self.CLASS_ID_AND_TERM_COLUMN not in self.df.columns:
      self.getUniqueIdentifiersForSectionsAcrossTerms()
    if not self.__requiredColumnPresent(self.CLASS_ID_AND_TERM_COLUMN):
        return
    temp = self.df.loc[:, [self.CLASS_ID_AND_TERM_COLUMN, self.FINAL_GRADE_COLUMN]]
    standardDev = temp.groupby(self.CLASS_ID_AND_TERM_COLUMN).std()
    standardDev.rename(columns={self.FINAL_GRADE_COLUMN:self.GPA_STDDEV_COLUMN}, inplace=True)
    self.df = pd.merge(self.df,standardDev, on=self.CLASS_ID_AND_TERM_COLUMN)

  def getGPAMeans(self):
    """Used internally. Makes a new column called 'gpaMean', the mean of grades recieved in the respective class of each entry.
    
    """
    if not self.__requiredColumnPresent(self.FINAL_GRADE_COLUMN):
        return
    self.dropMissingValuesInColumn(self.FINAL_GRADE_COLUMN)
    self.convertColumnToNumeric(self.FINAL_GRADE_COLUMN)
    if self.CLASS_ID_AND_TERM_COLUMN not in self.df.columns:
      self.getUniqueIdentifiersForSectionsAcrossTerms()
    if not self.__requiredColumnPresent(self.CLASS_ID_AND_TERM_COLUMN):
        return
    temp = self.df.loc[:, [self.CLASS_ID_AND_TERM_COLUMN, self.FINAL_GRADE_COLUMN]]
    meanGpas = temp.groupby(self.CLASS_ID_AND_TERM_COLUMN).mean()
    meanGpas.rename(columns={self.FINAL_GRADE_COLUMN:self.GPA_MEAN_COLUMN}, inplace=True)
    self.df = pd.merge(self.df,meanGpas, on=self.CLASS_ID_AND_TERM_COLUMN)

  def filterByGpaDeviationMoreThan(self, minimum, outputDropped = False, droppedCSVName = 'droppedData.csv'):
    """Filters data to only include classes which have a standard deviation more than or equal to a given minimum (0.0 to 4.0 scale).

    Args:
        minimum (:obj:`float`): Minimum standard deviation of grades a class must have.
        outputDropped (:obj:`bool`, optional): Whether to output the dropped data to a file. Default is False.
        droppedCSVName (:obj:`str`, optional): Name of file to output dropped data to. Default is 'droppedData.csv`.

    """
    if self.GPA_STDDEV_COLUMN not in self.df.columns:
      self.getGPADeviations()
    if not self.__requiredColumnPresent(self.GPA_STDDEV_COLUMN):
        return
    self.convertColumnToNumeric(self.GPA_STDDEV_COLUMN)
    if outputDropped:
      self.df[self.df[self.GPA_STDDEV_COLUMN] < minimum].to_csv(droppedCSVName, index=False)
    start = self.getEntryCount() 
    self.df = self.df[self.df[self.GPA_STDDEV_COLUMN] >= minimum]
    if start > self.getEntryCount():
      print('filtering by grade standard deviation of '+ str(minimum) +', ' + str(start - self.getEntryCount()) + ' entries dropped')

  def filterToSpecificValueInColumn(self, column, value):
    if not self.__requiredColumnPresent(column):
        return
    self.df = self.df[self.df[column] == value]
  
  def removeDuplicatesInColumn(self, column):
    if not self.__requiredColumnPresent(column):
        return
    self.df.drop_duplicates(subset=column, inplace=True)

  def countDuplicatesInColumn(self, column, term):
    if not self.__requiredColumnPresent(column):
        return
    return self.df[column].value_counts()[term]

  def countUniqueValuesInColumn(self, column):
    if not self.__requiredColumnPresent(column):
        return
    return self.df[column].nunique()

  def getUniqueValuesInColumn(self, column):
    if not self.__requiredColumnPresent(column):
        return
    return self.df[column].unique()

  def printUniqueValuesInColumn(self, column):
    """Prints to console the unique variables in a given column.

    Args:
        column (:obj:`str`): Column to get unique variables from.
    
    """
    if not self.__requiredColumnPresent(column):
        return
    print(self.df[column].unique())

  def getEntryCount(self):
    return len(self.df.index)

  def printEntryCount(self):
    """Prints to console the number of entries (rows) in the current dataset.
    
    """
    print(str(len(self.df.index)) + ' entries')

  def printFirstXRows(self, rows):
    """Prints to console the first X number of rows from the dataset.
    
    Args:
        rows (:obj:`int`): Number of rows to print from the dataset.

    """
    print(self.df.iloc[:rows])
  
  def removeRowsWithNullInColumn(self, column):
    if not self.__requiredColumnPresent(column):
        return
    self.df = self.df[self.df[column] != ' ']

  def removeRowsWithZeroInColumn(self, column):
    if not self.__requiredColumnPresent(column):
        return
    self.df = self.df[self.df[column] != 0]

  def getPandasDataFrame(self):
    """Returns the pandas dataframe of the dataset.

    Returns:
        :obj:`pandas.dataframe`: Dataframe of the current dataset.

    """
    return self.df

  def __requiredColumnPresent(self, column):
    if column not in self.df.columns:
    # JH: Shouldn't the caller decide what to print? 
      if edmApplication:
        print("Error: required column '" + column + "' not present in dataset. Fix by right clicking / setting columns.")
      else:
        print("Error: required column '" + column + "' not present in dataset. Fix or rename with the 'defineWorkingColumns' function.")
      return False
    return True

  def makeSemesterNumberColumn(self, firstYear = 2010):
    """Used internally. Makes a new column called 'semNumber', starting at 0 for Spring 2010 and adding 1 for each normal semester. 
    Summer semesters are considered in between normal semesters, and have an increment of 0.5 semesters.
    
    Args: 
        firstYear (:obj:`int`, optional), the earliest year in the dataset. Defaults to 2010
    
    """
    semNumberList = []
    #Cycle through each student
    for term in self.df.REG_banTerm:
      #Separate the string term at the last 4 characters to get a season and a year
      #This code will have to be updated in 7,979 years to account for 5 digit years
      season = term[:-5]
      year = term[-4:]
      #Convert the year to int
      year = int(year)
      seasonNum = 0
      if (season == "Summer"):
        seasonNum = 0.5
      elif (season == "Fall"):
        seasonNum = 1
      #Calculate the semesterNum
      #Designed so that there is 1 semester between two normal semesters and 0.5 semesters between a summer semester and normal semester
      semesterNum = (year - firstYear) * 2 + seasonNum
      #Append to the list
      semNumberList.append(semesterNum)
    #Make this list the column semNumber in the dataFrame
    self.df["semNumber"] = semNumberList
    
  def makeStudentGPAColumn(self):
    gpaDict = self.getDictOfStudentGPAs()
    gpaList = []
    for student in self.df["SID"]:
      gpaList.append(gpaDict[student])
    self.df["studentGPA"] = gpaList

  def createClassCodeColumn(self):
    self.df[self.CLASS_CODE_COLUMN] = self.df[self.CLASS_DEPT_COLUMN] + self.df[self.CLASS_NUMBER_COLUMN]

