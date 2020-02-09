# Library created by Michael Riad Zaky mriadzaky@fordham.edu (2020), with Daniel Leeds, Gary Weiss, Mavis Zhang at Fordham 
# University. Library free for use provided you cite https://github.com/MichaelRZ/EDMLib in any resulting publications.  
# Library free for redistribution provided you retain the author attributions above.

import time
import numpy as np
import pandas as pd
import csv
import sys 
import math
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.mode.chained_assignment = None 

class classCorrelationData:
  df = None
  sourceFile = ""

  def __init__(self,sourceFileOrDataFrame):
    if type(sourceFileOrDataFrame).__name__ == 'str':
      self.sourceFile = sourceFileOrDataFrame
      self.df = pd.read_csv(self.sourceFile)

    elif type(sourceFileOrDataFrame).__name__ == 'DataFrame':
      self.df = sourceFileOrDataFrame.copy()
  
  def getClassesUsed(self):
    return self.df['course1'].unique()

  def getNumberOfClassesUsed(self):
    return self.df['course1'].nunique()

  def printUniqueValuesInColumn(self, column):
    print(self.df[column].unique())

  def printClassesUsed(self):
    self.printUniqueValuesInColumn('course1')

  def getEntryCount(self):
    return len(self.df.index)

  def printEntryCount(self):
    print(str(len(self.df.index)) + ' entries')

  def printFirstXRows(self, rows):
    print(self.df.iloc[:rows])
  
  def graph(self):
    #print(self.df.corr())
    print(self.getClassesUsed())
    # matrix = self.df.pivot(index='course1', columns='course2', values='P-value')
    # items = list(matrix)
    # items.extend(list(matrix.index))
    # items = list(set(items))
    # corr = matrix.values.tolist()
    # corr.insert(0, [np.nan] * len(corr))
    # corr = pd.DataFrame(corr)
    # corr[len(corr) - 1] = [np.nan] * len(corr)
    # for i in range(len(corr)):
    #     corr.iat[i, i] = 1.  # Set diagonal to 1.00
    #     corr.iloc[i, i:] = corr.iloc[i:, i].values  # Flip matrix.
    # corr.index = items
    # corr.columns = items
    matrix = pd.DataFrame()
    for firstCourse in self.getClassesUsed():
      for secondCourse in self.getClassesUsed():
        row = self.df.loc[(self.df['course1'] == firstCourse) 
                        & (self.df['course2'] == secondCourse)]
        if not row.empty:
          matrix.loc[firstCourse, secondCourse] = row.iloc[0]['P-value']
        else:
          matrix.loc[firstCourse, secondCourse] = 0
    matrix.index = self.getClassesUsed()
    matrix.columns = self.getClassesUsed()
    print(matrix)
    matrix.to_csv('edmExampleMatrix.csv')
    # Generate a mask for the upper triangle
    mask = np.zeros_like(matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    ax.figure.tight_layout()
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.matshow(matrix)
    plt.show()
    # corr.to_csv('edmExampleMatrix.csv')
    # plt.matshow(corr)
    # plt.show()

class gradeData:
  
  ######## Variables #########
  df = None
  sourceFile = ""
  FINAL_GRADE_COLUMN = 'OTCM_FinalGradeN'

  CRN_AND_TERM_COLUMN = 'courseNumberAndTerm'
  CLASS_CODE_COLUMN = 'classCode'
  ClASS_MAJOR_COLUMN = 'REG_Programcode'
  GPA_STDDEV_COLUMN = 'gpaStdDeviation'
  GPA_MEAN_COLUMN = 'gpaMean'
  STUDENT_ID_COLUMN = 'SID'

  dtypes = {#'SID':str, 
          #'REG_Programcode':str, 
          'REG_Numbercode':str,#
          #'REG_crsSchool':str,
          #'REG_CourseCrn':str,
          #'REG_REG_credHr':str,#
          #'REG_banTerm':str,
          #'REG_term':str,
          #'REG_classSize':str,#
          #'CRS_crsCampus':str,
          #'CRS_schdtyp':str,
          #'FID':str,
          #'CRS_coursetitle':str,
          'CRS_contact_hrs':str,#
          #'CRS_XLSTGRP':str,
          #'CRS_PrimarySect':str,#
          #'CRS_enrolltally':str,#
          #'STU_ulevel':str,
          #'STU_DegreeSeek':str,
          #'STU_credstua':str,
          #'STU_G_transfer':str,
          #'GRA_Grad_Term':str,#
          #'GRA_MajorAtGraduation':str,
          #'GRA_Major2atGraduation':str,
          #'GRA_MinorAtGraduation':str,
          #'GRA_Minor2AtGraduation':str,
          #'GRA_ConcAtGraduation':str,
          #'GRA_Conc2atGraduation':str,
          #'GRA_degreeatGraduation':str,
          #'OTCM_FinalGradeC':str,
          #'OTCM_Crs_Graded':str,
          #'OTCM_FinalGradeN':str
          }

  ######## Methods #########
  def __init__(self,sourceFileOrDataFrame = "final-datamart-6-7-19.csv"):
    if type(sourceFileOrDataFrame).__name__ == 'str':
      self.sourceFile = sourceFileOrDataFrame
      self.df = pd.read_csv(self.sourceFile, dtype=self.dtypes)

    elif type(sourceFileOrDataFrame).__name__ == 'DataFrame':
      self.df = sourceFileOrDataFrame.copy()

  def getColumn(self, column):
    return self.df[column]

  def printColumn(self, column):
    print(self.df[column])

  def defineWorkingColumns(self, finalGradeInClassColumn, classDefiningColumn, courseNumberAndTerm, studentIdColumn, classMajorColumn):
    self.FINAL_GRADE_COLUMN = finalGradeInClassColumn
    self.CRN_AND_TERM_COLUMN = courseNumberAndTerm
    self.CLASS_CODE_COLUMN = classDefiningColumn
    self.STUDENT_ID_COLUMN = studentIdColumn
    self.ClASS_MAJOR_COLUMN = classMajorColumn

  def makeMissingValuesNanInColumn(self, column):
    self.df[column].replace(' ', np.nan, inplace=True)

  def removeNanInColumn(self, column):
    self.df.dropna(subset=[column], inplace=True)

  def dropMissingValuesInColumn(self, column):
    self.makeMissingValuesNanInColumn(column)
    self.removeNanInColumn(column)

  def convertColumnToNumeric(self, column):
    self.df[column] = pd.to_numeric(self.df[column])

  def convertColumnToString(self, column):
    self.df.astype({column:str}, copy=False)

  def getUniqueIdentifiersForSectionsAcrossTerms(self):
    self.df[self.CLASS_CODE_COLUMN] = self.df['REG_Programcode'].astype(str) + self.df['REG_Numbercode']
    self.df[self.CLASS_CODE_COLUMN] = self.df[self.CLASS_CODE_COLUMN].str.replace(" ","")
    self.df[self.CRN_AND_TERM_COLUMN] = self.df['REG_CourseCrn'].astype(str) + self.df['REG_term']
    self.df[self.CRN_AND_TERM_COLUMN] = self.df[self.CRN_AND_TERM_COLUMN].str.replace(" ","")

  def getCorrelationsWithMinNSharedStudents(self, filename = 'CorrelationOutput_EDMLIB.csv', nSharedStudents = 20):
    print("Getting correlations...")
    start_time = time.time()
    if "norm" not in self.df.columns:
      self.getNormalizationColumn()
    
    def corrAlg(a, b): 
      def getNormsList(df1, df2):
        norms = df1.loc[df1[self.STUDENT_ID_COLUMN].isin(df2[self.STUDENT_ID_COLUMN])]
        norms = norms['norm']
        norms.dropna(inplace = True)
        norms = norms.reset_index()
        norms = norms.drop(['index'], axis=1)
        return norms
      x0 = getNormsList(b, a)
      if len(x0) >= nSharedStudents:
        y0 = getNormsList(a, b)
        corr, Pvalue = pearsonr(x0['norm'], y0['norm'])
        return [corr, Pvalue, len(x0)]
      else:
        return [math.nan, math.nan, math.nan]

    print("Getting classes...")
    classes = self.getListOfClassCodes()
    d={}
    print("Organizing classes...")
    for n in classes:
      d["df{0}".format(n)] = self.df.loc[self.df[self.CLASS_CODE_COLUMN] == n]
      d["df{0}".format(n)] = d["df{0}".format(n)].drop_duplicates(subset="SID", keep=False)
      d["df{0}".format(n)].sort_values(by=[self.STUDENT_ID_COLUMN], inplace=True)
    f = []
    classCount = 0
    totalClasses = len(classes)
    print("Sorting classes...")
    classes.sort()
    for n in classes:
        classCount = classCount + 1
        print("class " + str(classCount) + "/" + str(totalClasses))
        for m in classes:
            result = corrAlg(d["df{0}".format(n)],d["df{0}".format(m)])
            r, p, c = result[0], result[1], result[2]
            f.append((n, m, r, p, c))
    normoutput = pd.DataFrame(f, columns=('course1', 'course2', 'corr', 'P-value', '#students'))
    normoutput = normoutput.dropna()
    print(str((totalClasses ** 2) - len(normoutput.index)) + ' class correlations dropped out of ' 
    + str(totalClasses ** 2) + ' from ' + str(nSharedStudents) + ' shared student threshold.')
    print(str(len(normoutput.index)) + ' correlations calculated. ' + str(time.time() - start_time) + ' seconds.')
    return normoutput

  def exportCorrelationsWithMinNSharedStudents(self, filename = 'CorrelationOutput_EDMLIB.csv', nStudents = 20):
    self.getCorrelationsWithMinNSharedStudents(nSharedStudents=nStudents).to_csv(filename)

  def exportCorrelationsWithAvailableClasses(self, filename = 'CorrelationOutput_EDMLIB.csv'):
    self.getCorrelationsWithMinNSharedStudents().to_csv(filename)

  def getCorrelationsOfAvailableClasses(self):
    return self.getCorrelationsWithMinNSharedStudents()

  def getListOfClassCodes(self):
    if self.CLASS_CODE_COLUMN not in self.df.columns:
      self.getUniqueIdentifiersForSectionsAcrossTerms()
    return self.getUniqueValuesInColumn(self.CLASS_CODE_COLUMN)

  def getNormalizationColumn(self):
    print('getting normalization column...')
    if self.CLASS_CODE_COLUMN not in self.df.columns:
      self.getUniqueIdentifiersForSectionsAcrossTerms()
    if self.GPA_STDDEV_COLUMN not in self.df.columns:
      self.getGPADeviations()
    if self.GPA_MEAN_COLUMN not in self.df.columns:
      self.getGPAMeans()
    self.df['norm'] = (self.df[self.FINAL_GRADE_COLUMN].values - self.df[self.GPA_MEAN_COLUMN].values) / self.df[self.GPA_STDDEV_COLUMN].values
  
  def reduceToMultipleMajorsOrClasses(self, majors, classes):
    if self.CLASS_CODE_COLUMN not in self.df.columns:
      self.getUniqueIdentifiersForSectionsAcrossTerms()
    self.df = self.df.loc[(self.df[self.CLASS_CODE_COLUMN].isin(classes)) | self.df.REG_Programcode.isin(majors)]
  
  def getGPADeviations(self):
    print('getting grade deviations by class...')
    self.dropMissingValuesInColumn(self.FINAL_GRADE_COLUMN)
    self.convertColumnToNumeric(self.FINAL_GRADE_COLUMN)
    if self.CRN_AND_TERM_COLUMN not in self.df.columns:
      self.getUniqueIdentifiersForSectionsAcrossTerms()
    temp = self.df.loc[:, [self.CRN_AND_TERM_COLUMN, self.FINAL_GRADE_COLUMN]]
    standardDev = temp.groupby(self.CRN_AND_TERM_COLUMN).std()
    standardDev.rename(columns={self.FINAL_GRADE_COLUMN:self.GPA_STDDEV_COLUMN}, inplace=True)
    self.df = pd.merge(self.df,standardDev, on=self.CRN_AND_TERM_COLUMN)

  def getGPAMeans(self):
    self.dropMissingValuesInColumn(self.FINAL_GRADE_COLUMN)
    self.convertColumnToNumeric(self.FINAL_GRADE_COLUMN)
    if self.CRN_AND_TERM_COLUMN not in self.df.columns:
      self.getUniqueIdentifiersForSectionsAcrossTerms()
    temp = self.df.loc[:, [self.CRN_AND_TERM_COLUMN, self.FINAL_GRADE_COLUMN]]
    meanGpas = temp.groupby(self.CRN_AND_TERM_COLUMN).mean()
    meanGpas.rename(columns={self.FINAL_GRADE_COLUMN:self.GPA_MEAN_COLUMN}, inplace=True)
    self.df = pd.merge(self.df,meanGpas, on=self.CRN_AND_TERM_COLUMN)

  def filterByGpaDeviationMoreThan(self, minimum, outputDropped = False, droppedCSVName = 'droppedData.csv'):
    if self.GPA_STDDEV_COLUMN not in self.df.columns:
      self.getGPADeviations()
    if outputDropped:
      self.df[self.df[self.GPA_STDDEV_COLUMN] < minimum].to_csv(droppedCSVName)
    print('filtering by grade standard deviation...')
    print(str(len(self.df[self.df[self.GPA_STDDEV_COLUMN] < minimum].index)) + ' entries dropped')
    self.df = self.df[self.df[self.GPA_STDDEV_COLUMN] >= minimum]

  def reduceToSpecificValueInColumn(self, column, value):
    self.df = self.df[self.df[column] == value]
  
  def removeDuplicatesInColumn(self, column):
    self.df.drop_duplicates(subset=column, inplace=True)

  def countDuplicatesInColumn(self, column, term):
    return self.df[column].value_counts()[term]

  def countUniqueValuesInColumn(self, column):
    return self.df[column].nunique()

  def getUniqueValuesInColumn(self, column):
    return self.df[column].unique()

  def printUniqueValuesInColumn(self, column):
    print(self.df[column].unique())

  def getEntryCount(self):
    return len(self.df.index)

  def printEntryCount(self):
    print(str(len(self.df.index)) + ' entries')

  def printFirstXRows(self, rows):
    print(self.df.iloc[:rows])
  
  def removeRowsWithNullInColumn(self, column):
    self.df = self.df[self.df[column] != ' ']

  def getPandasDataFrame(self):
    return self.df

  def exportCSV(self, fileName = 'csvExport.csv'):
    self.df.to_csv(fileName)

