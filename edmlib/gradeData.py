"""
Library created by Michael Riad Zaky mriadzaky@fordham.edu (2020), with Daniel Leeds, Gary Weiss, Mavis Zhang at Fordham
University. Library free for use provided you cite https://github.com/MichaelRZ/EDMLib in any resulting publications.
Library free for redistribution provided you retain the author attributions above.

The following packages are required for installation before use: 
"""
from ctypes import sizeof
import time
import numpy as np
import pandas as pd
import csv
import sys
import math
from scipy.stats.stats import pearsonr
from scipy.stats import norm as sciNorm
import statistics
from statistics import stdev
import re, os
import networkx as nx
import itertools
import holoviews as hv
from holoviews import opts, dim
from bokeh.plotting import show, save, output_file
from bokeh.io import export_png
from bokeh.models import Title
from edmlib.gradeDataHelper import gradeDataHelper
from edmlib.edmlib import edmApplication, outDir

class gradeData(gradeDataHelper):
  """Class for manipulating grade datasets.

    Attributes:
        df (:obj:`pandas.dataframe`): dataframe containing all grade data.
        sourceFile (:obj:`str`): Name of source .CSV file with grade data (optional).

  """
  ######## Methods #########
  def __init__(self,sourceFileOrDataFrame,copyDataFrame=True):
    """Class constructor, creates an instance of the class given a .CSV file or pandas dataframe.

    Used with gradeData('fileName.csv') or gradeData(dataFrameVariable).

    Args:
      sourceFileOrDataFrame (:obj:`object`): name of the .CSV file (extension included) in the same path or pandas dataframe variable. Dataframes are copied so as to not affect the original variable.

    """
    super().__init__(sourceFileOrDataFrame,copyDataFrame)

  def outputEnrollmentDistribution(self, makeHistogram = False, fileName = 'enrollmentHistogram', graphTitle='Enrollment Distribution', exportPNG=False):
    """Make a graph overview overview of student distribution per semester. 
        Optionally, outputs a histogram as well.

    Args:
        makeHistogram (:obj:`bool`, optional): Whether or not to make a histogram graph. Default false.
        fileName (:obj:`str`): Name of histogram files to output. Default 'enrollmentHistogram'.
        graphTitle (:obj:`str`): Title to display on graph. Default 'Enrollment Distribution'.

    """
    # Check to see if Student ID and Semester ID columns are present
    if not self._gradeDataHelper__requiredColumnPresent(self.STUDENT_ID_COLUMN):
      return
    if not self._gradeDataHelper__requiredColumnPresent(self.TERM_COLUMN):
      return
    # Locate needed columns
    temp = self.df.loc[:, [self.TERM_COLUMN, self.STUDENT_ID_COLUMN]]
    # Locate, Group, and Count students per course
    temp2 = temp.groupby(self.STUDENT_ID_COLUMN)[self.TERM_COLUMN].unique()
    frequencyTable = pd.DataFrame.from_records(temp2.values.tolist()).stack().value_counts()
    print(frequencyTable)
    vals = frequencyTable.tolist()
    if makeHistogram:
      lowest = round(float('%.1f'%(min(vals))), 1)
      highest = round(max(vals), 1)
      frequencies, edges = np.histogram(vals, int((highest - lowest) / 500), (lowest, highest)) # Scale Division in this line
      histo = hv.Histogram((edges, frequencies))
      histo.opts(opts.Histogram(xlabel='Students per Semester', ylabel='Frequency', title=graphTitle))
      subtitle= 'n = ' + str(temp[self.TERM_COLUMN].nunique()) + ' | mean = ' + str(round(sum(vals) / len(vals), 2))\
                + ' | std dev = ' + str(round(stdev(vals),2))
      hv.output(size=125)
      graph = hv.render(histo)
      graph.add_layout(Title(text=subtitle, text_font_style="italic", text_font_size="10pt"), 'above')
      output_file(outDir +fileName + '.html', mode='inline')
      save(graph)
      show(graph)
      if exportPNG:
        histo.opts(toolbar=None)
        graph = hv.render(histo)
        graph.add_layout(Title(text=subtitle, text_font_style="italic", text_font_size="10pt"), 'above')
        export_png(graph, filename=outDir +fileName + '.png')

  def outputCourseDistribution(self, makeHistogram = False, fileName = 'courseHistogram', graphTitle='Course Distribution', exportPNG=False):
    """Make a graph overview of course distribution per department. 
        Optionally, outputs a histogram as well.

    Args:
        makeHistogram (:obj:`bool`, optional): Whether or not to make a histogram graph. Default false.
        fileName (:obj:`str`): Name of histogram files to output. Default 'courseHistogram'.
        graphTitle (:obj:`str`): Title to display on graph. Default 'Course Distribution'.

    """
    # Check to see if Dept ID and Class ID columns are present
    if not self._gradeDataHelper__requiredColumnPresent(self.CLASS_DEPT_COLUMN):
      return
    if not self._gradeDataHelper__requiredColumnPresent(self.CLASS_ID_COLUMN):
      return
    # Locate needed columns
    temp = self.df.loc[:, [self.CLASS_DEPT_COLUMN, self.CLASS_ID_COLUMN]]
    # Locate, Group, and Count students per course
    temp2 = temp.groupby(self.CLASS_ID_COLUMN)[self.CLASS_DEPT_COLUMN].unique()
    frequencyTable = pd.DataFrame.from_records(temp2.values.tolist()).stack().value_counts()
    print(frequencyTable)
    vals = frequencyTable.tolist()
    if makeHistogram:
      lowest = round(float('%.1f'%(min(vals))), 1)
      highest = round(max(vals), 1)
      frequencies, edges = np.histogram(vals, int((highest - lowest) / 10), (lowest, highest)) # Scale Division in this line
      histo = hv.Histogram((edges, frequencies))
      histo.opts(opts.Histogram(xlabel='Courses per Department', ylabel='Frequency', title=graphTitle))
      subtitle= 'n = ' + str(temp[self.CLASS_DEPT_COLUMN].nunique()) + ' | mean = ' + str(round(sum(vals) / len(vals), 2))\
                + ' | std dev = ' + str(round(stdev(vals),2))
      hv.output(size=125)
      graph = hv.render(histo)
      graph.add_layout(Title(text=subtitle, text_font_style="italic", text_font_size="10pt"), 'above')
      output_file(outDir +fileName + '.html', mode='inline')
      save(graph)
      show(graph)
      if exportPNG:
        histo.opts(toolbar=None)
        graph = hv.render(histo)
        graph.add_layout(Title(text=subtitle, text_font_style="italic", text_font_size="10pt"), 'above')
        export_png(graph, filename=outDir +fileName + '.png')

  def outputClassSizeDistribution(self, makeHistogram = False, fileName = 'classSizeHistogram', graphTitle='Class Size Distribution', barWidth=20, exportPNG=False):
    """Make a graph overview of class size distribution for each course. 
        Optionally, outputs a histogram as well.

    Args:
        makeHistogram (:obj:`bool`, optional): Whether or not to make a histogram graph. Default false.
        fileName (:obj:`str`): Name of histogram files to output. Default 'classSizeHistogram'.
        graphTitle (:obj:`str`): Title to display on graph. Default 'Class Size Distribution'.
        barWidth (:obj:`int`): interval, width of each bar in histogram. Default 20.

    """
    # Check to see if Student ID and Class ID columns are present
    if not self._gradeDataHelper__requiredColumnPresent(self.STUDENT_ID_COLUMN):
      return
    if not self._gradeDataHelper__requiredColumnPresent(self.CLASS_ID_COLUMN):
      return
    # Locate needed columns
    temp = self.df.loc[:, [self.CLASS_ID_COLUMN, self.STUDENT_ID_COLUMN]]
    # Locate, Group, and Count students per course
    temp2 = temp.groupby(self.STUDENT_ID_COLUMN)[self.CLASS_ID_COLUMN].unique()
    frequencyTable = pd.DataFrame.from_records(temp2.values.tolist()).stack().value_counts()
    print(frequencyTable)
    vals = frequencyTable.tolist()
    if makeHistogram:
      lowest = min(vals)
      highest = max(vals)
      frequencies, edges = np.histogram(vals, int((highest - lowest) / barWidth), (lowest, highest)) # Scale Division in this line
      histo = hv.Histogram((edges, frequencies))
      histo.opts(opts.Histogram(xlabel='Students per Course', ylabel='Frequency', title=graphTitle)) # For future feature: add logy=true to draw log scale
      subtitle= 'n = ' + str(temp[self.CLASS_ID_COLUMN].nunique()) + ' | mean = ' + str(round(sum(vals) / len(vals), 2))\
                + ' | std dev = ' + str(round(stdev(vals),2))
      hv.output(size=125)
      graph = hv.render(histo)
      graph.add_layout(Title(text=subtitle, text_font_style="italic", text_font_size="10pt"), 'above')
      output_file(outDir +fileName + '.html', mode='inline')
      save(graph)
      show(graph)
      if exportPNG:
        histo.opts(toolbar=None)
        graph = hv.render(histo)
        graph.add_layout(Title(text=subtitle, text_font_style="italic", text_font_size="10pt"), 'above')
        export_png(graph, filename=outDir +fileName + '.png')
  
  def outputGpaDistribution(self, makeHistogram = False, fileName = 'gpaHistogram', graphTitle='GPA Distribution', minClasses = 36, exportPNG=False):
    """Prints to console an overview of student GPAs in increments of 0.1 between 1.0 and 4.0. Optionally, outputs a histogram as well.

    Args:
        makeHistogram (:obj:`bool`, optional): Whether or not to make a histogram graph. Default false.
        fileName (:obj:`str`): Name of histogram files to output. Default 'gpaHistogram'.
        graphTitle (:obj:`str`): Title to display on graph. Default 'GPA Distribution'.
        minClasses (:obj:`int`): Number of classes a student needs to have on record to count GPA. Default 36.

    """
    # Check to see if Student ID and Final Grade columns are present
    if not self._gradeDataHelper__requiredColumnPresent(self.FINAL_GRADE_COLUMN):
      return
    if not self._gradeDataHelper__requiredColumnPresent(self.STUDENT_ID_COLUMN):
      return
    self.df[self.FINAL_GRADE_COLUMN] = pd.to_numeric(self.df[self.FINAL_GRADE_COLUMN], errors='coerce')
    # Access each student student final grade and credit numbers
    if self.CLASS_CREDITS_COLUMN in self.df.columns: # check if class credits is present
      temp = self.df.loc[:, [self.STUDENT_ID_COLUMN, self.CLASS_CREDITS_COLUMN, self.FINAL_GRADE_COLUMN]]
    else:
      temp = self.df.loc[:, [self.STUDENT_ID_COLUMN, self.FINAL_GRADE_COLUMN]]
    classCount = temp[self.STUDENT_ID_COLUMN].value_counts()
    temp = temp[temp[self.STUDENT_ID_COLUMN].isin(classCount[classCount >= minClasses].index)] # only count students that take more than minClasses
    print('Number of Students: ' + str(temp[self.STUDENT_ID_COLUMN].nunique()))
    if temp[self.STUDENT_ID_COLUMN].nunique() == 0:
      print('Error: no students meet the criteria given')
      return
    if self.CLASS_CREDITS_COLUMN in self.df.columns: # check if class credits is present
      temp[self.CLASS_CREDITS_COLUMN] = pd.to_numeric(temp[self.CLASS_CREDITS_COLUMN], errors='coerce') # convert all credits to numeric type
      temp['classPoints'] = temp[self.CLASS_CREDITS_COLUMN] * temp[self.FINAL_GRADE_COLUMN] # calculate Grade * Credit numbers
      sums = temp.groupby(self.STUDENT_ID_COLUMN).sum()
      sums['gpa'] = sums['classPoints'] / sums[self.CLASS_CREDITS_COLUMN] # calculate weighted GPA
      gpas = sums['gpa'].tolist()
    else: # don't count class credits if they are not present
      gradeAverages = temp.groupby(self.STUDENT_ID_COLUMN).mean()
      gpas = gradeAverages[self.FINAL_GRADE_COLUMN].tolist()
    mean = sum(gpas) / len(gpas)
    grade = 4.0
    print(">= "+ str(grade) + ": " + str(len([x for x in gpas if x >= grade]))) # print how many grades are larger than a certain value
    while grade != 1.0:
      lowerGrade = round(grade - 0.1, 1) # rounding results
      print(str(lowerGrade) + " - " + str(grade) + ": " + str(len([x for x in gpas if x >= lowerGrade and x < grade]))) # print how many grades are between 2 certain values
      grade = round(grade - 0.1, 1)
    print("< "+ str(grade) + ": " + str(len([x for x in gpas if x < grade]))) # print how many grades are smaller than a certain value
    print('mean: ' + str(mean))
    if makeHistogram:
      lowest = round(float('%.1f'%(min(gpas))), 1)
      highest = round(max(gpas), 1)
      frequencies, edges = np.histogram(gpas, int((highest - lowest) / 0.1), (lowest, highest))
      #print('Values: %s, Edges: %s' % (frequencies.shape[0], edges.shape[0]))
      histo = hv.Histogram((edges, frequencies))
      histo.opts(opts.Histogram(xlabel='Student GPA', ylabel='Frequency', title=graphTitle))
      subtitle= 'mean: ' + str(round(sum(gpas) / len(gpas), 2))+ ', n = ' + str(temp[self.STUDENT_ID_COLUMN].nunique())
      hv.output(size=125)
      graph = hv.render(histo)
      graph.add_layout(Title(text=subtitle, text_font_style="italic", text_font_size="10pt"), 'above')
      output_file(outDir +fileName + '.html', mode='inline')
      save(graph)
      show(graph)
      # JH: Why not add a bool for exportPng=True, set False when in edmApplication?
      #      if not edmApplication:
      if exportPNG:
        histo.opts(toolbar=None)
        graph = hv.render(histo)
        graph.add_layout(Title(text=subtitle, text_font_style="italic", text_font_size="10pt"), 'above')
        export_png(graph, filename=outDir +fileName + '.png')

  def exportCoursesByStudents(self, sectionLevel = False, fileName = 'courses.csv'):
    """Export csv file containing the number of students in each course
    Args:
        sectionLevel (:obj:`bool`): whether to use unique section ID (or CRN) to identify unique classes (different sections
                                    in different semesters are counted seperately)
        fileName (:obj:`str`): name of export file
    """
    if sectionLevel:
      temp = self.df.loc[:,[self.CLASS_ID_COLUMN, self.CLASS_DEPT_COLUMN, self.CLASS_NUMBER_COLUMN]]
    else:
      temp = self.df.loc[:,[self.CLASS_DEPT_COLUMN, self.CLASS_NUMBER_COLUMN]]
    vals = dict(temp.value_counts()) # count class ID (CRN)
    vals = {k: v for k, v in sorted(vals.items(), key=lambda item: item[1], reverse=True)} # sort frequency table in descending order
    studentsPerCourse = pd.DataFrame(vals.items(), columns=['Course', '# of students'])
    # export to csv file
    if not fileName.endswith('.csv'):
      fileName = "".join((fileName, '.csv'))
    studentsPerCourse.to_csv(fileName, index=False)

  def exportDepartmentByStudents(self, fileName = 'department.csv'):
    """Export csv file containing the number of students in each department
    Args:
        fileName (:obj:`str`): name of export file
    """
    temp = self.df.loc[:,[self.CLASS_DEPT_COLUMN, self.STUDENT_ID_COLUMN]]
    temp2 = temp.groupby(self.STUDENT_ID_COLUMN)[self.CLASS_DEPT_COLUMN].unique()
    vals = dict(pd.DataFrame.from_records(temp2.values.tolist()).stack().value_counts()) # count unique students for each dept
    vals = {k: v for k, v in sorted(vals.items(), key=lambda item: item[1], reverse=True)} # sort frequency table in descending order
    studentsPerDept = pd.DataFrame(vals.items(), columns=['Department', '# of students']) # construct dataframe
    # export to csv file
    if not fileName.endswith('.csv'):
      fileName = "".join((fileName, '.csv'))
    studentsPerDept.to_csv(fileName, index=False)

  def sankeyGraphByCourseTracksOneGroup(self, courseGroup, requiredCourses = None, graphTitle='Track Distribution', outputName = 'sankeyGraph', minEdgeValue = None):
    """Exports a sankey graph according to a given course track. Input is organized as an array of classes included in the track, and optionally a subgroup of classes required for a student to be counted in the graph can be designated as well.

    Args:
        courseGroup (:obj:`list`(:obj:`str`)): List of courses to make the sankey graph with. Minimum two courses.
        requiredCourses (:obj:`list`(:obj:`str`)): List of courses required for a student to count towards the graph. All courses in 'courseGroup' by default.
        graphTitle (:obj:`str`): Title that goes on the sankey graph. Defaults to 'Track Distribution'.
        outputName (:obj:`str`): Name of sankey files (.csv, .html) to output. Defaults to 'sankeyGraph'.
        minEdgeValue (:obj:`int`, optional): Minimum value for an edge to be included on the sankey graph. Defaults to `None`, or no minimum value needed.

    """
    print('Creating Sankey graph...')
    if len(courseGroup) < 2:
      print('Error: Minimum of two courses required in given course track.')
      return
    if requiredCourses:
      requiredCourses = [x for x in requiredCourses if x in courseGroup]
      if len(requiredCourses) == 0:
        requiredCourses = None
    if not requiredCourses:
      requiredCourses = courseGroup
    if self.CLASS_CODE_COLUMN not in self.df.columns:
      self.getUniqueIdentifiersForSectionsAcrossTerms()
    if self.CLASS_CODE_COLUMN not in self.df.columns:
      return
    #The following line is a function to get number suffixes. I don't know how it works, but it does.
    ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(math.floor(n/10)%10!=1)*(n%10<4)*n%10::4])
    firstGroup = self.df.loc[self.df[self.CLASS_CODE_COLUMN].isin(courseGroup)]
    relevantStudents = firstGroup[self.STUDENT_ID_COLUMN].unique()
    edges = {}
    def addEdge(first, second, count):
      firstNode = getattr(first, self.CLASS_CODE_COLUMN) + ' ' + ordinal(count)
      secondNode = getattr(second, self.CLASS_CODE_COLUMN) + ' ' + ordinal(count + 1)
      pair = (firstNode, secondNode)
      # print(pair)
      if pair in edges:
        edges[pair] += 1
      else:
        edges[pair] = 1
    outOf = len(relevantStudents)
    stNum = 0
    for student in relevantStudents:
      stNum += 1
      print('student ' + str(stNum) + '/' + str(outOf))
      count = 1
      studentClasses = self.df.loc[self.df[self.STUDENT_ID_COLUMN]==student]
      correctClasses = studentClasses.loc[studentClasses[self.CLASS_CODE_COLUMN].isin(courseGroup)]
      sortedClasses = correctClasses.sort_values(self.TERM_COLUMN)
      uniqueClasses = set(sortedClasses[self.CLASS_CODE_COLUMN].unique())
      if all(course in uniqueClasses for course in requiredCourses) and len(sortedClasses.index) > 1:
        first = None
        second = None
        lastTerm = None
        for row in sortedClasses.itertuples(index=False):
          if not lastTerm:
            lastTerm = getattr(row, self.TERM_COLUMN)
          if getattr(row, self.TERM_COLUMN) > lastTerm:
            lastTerm = getattr(row, self.TERM_COLUMN)
            count += 1
          nextTerm = None
          for row2 in sortedClasses.itertuples(index=False):
            if getattr(row2, self.TERM_COLUMN) > getattr(row, self.TERM_COLUMN):
              if not nextTerm:
                nextTerm = getattr(row2, self.TERM_COLUMN)
                addEdge(row, row2, count)
              elif getattr(row2, self.TERM_COLUMN) == nextTerm:
                addEdge(row, row2, count)
              else:
                break
      
    edgeList = []
    skippedEdges = {}
    if minEdgeValue:
      for key, value in edges.items():
        if value < minEdgeValue:
          if key[1] in skippedEdges:
            skippedEdges[key[1]] += value
          else:
            skippedEdges[key[1]] = value
    for key, value in edges.items():
      if minEdgeValue:
        if key[0] in skippedEdges:
          value -= skippedEdges[key[0]]
        if value < minEdgeValue:
          continue
      temp = [key[0], key[1], value]
      edgeList.append(temp)
    sankey = hv.Sankey(edgeList, ['From', 'To'])
    sankey.opts(width=600, height=400, node_padding=40, edge_color_index='From', color_index='index', title=graphTitle)
    graph = hv.render(sankey)
    output_file(outDir +outputName + '.html', mode='inline')
    save(graph)
    show(graph)

  def sankeyGraphByCourseTracks(self, courseGroups, graphTitle='Track Distribution', outputName = 'sankeyGraph', consecutive = True, minEdgeValue = None, termThreshold = None):
    """Exports a sankey graph according to a given course track. Input is organized in a jagged array, with the first array the first set of classes a student can take, the second set the second possible class a student can take, etc..

    Args:
        courseGroups (:obj:`list`(:obj:`list`)): List of course groups (also lists) to make the sankey graph with. Minimum two course groups.
        graphTitle (:obj:`str`): Title that goes on the sankey graph. Defaults to 'Track Distribution'.
        outputName (:obj:`str`): Name of sankey files (.csv, .html) to output. Defaults to 'sankeyGraph'.
        consecutive (:obj:`bool`): Whether or not students must complete the entire track consecutively, or start at a group other than what is designated. This mostly affects students who needed to retake a class. Defaults to :obj:`True` (students must complete track from beginning / as designated for data to be recorded).
        minEdgeValue (:obj:`int`, optional): Minimum value for an edge to be included on the sankey graph. Defaults to `None`, or no minimum value needed.
        termThreshold (:obj:`float`, optional): If defined, attempts to use the 'termOrder' column where terms are given a numbered order and a given maximum threshold for what counts as a "consecutive" term.

    """
    print('Creating Sankey graph...')
    if self.CLASS_CODE_COLUMN not in self.df.columns:
      self.getUniqueIdentifiersForSectionsAcrossTerms()
    if self.CLASS_CODE_COLUMN not in self.df.columns:
      return
    classSet = list(set(itertools.chain.from_iterable(courseGroups)))
    #The following line is a function to get number suffixes. I don't know how it works, but it does.
    ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(math.floor(n/10)%10!=1)*(n%10<4)*n%10::4])
    firstGroup = self.df.loc[np.in1d(self.df[self.CLASS_CODE_COLUMN], courseGroups[0])]
    relevantStudents = firstGroup[self.STUDENT_ID_COLUMN].unique()
    edges = {}
    def addEdge(first, second, count):
      if consecutive:
        if (first[self.CLASS_CODE_COLUMN] not in courseGroups[count - 1]) or (second[self.CLASS_CODE_COLUMN] not in courseGroups[count]):
          return
      firstNode = first[self.CLASS_CODE_COLUMN] + ' ' + ordinal(count)
      secondNode = second[self.CLASS_CODE_COLUMN] + ' ' + ordinal(count + 1)
      pair = (firstNode, secondNode)
      print(pair)
      if pair in edges:
        edges[pair] += 1
      else:
        edges[pair] = 1
    outOf = len(relevantStudents)
    stNum = 0
    for student in relevantStudents:
      stNum += 1
      print('student ' + str(stNum) + '/' + str(outOf))
      count = 0
      studentClasses = self.df.loc[self.df[self.STUDENT_ID_COLUMN]==student]
      correctClasses = studentClasses.loc[np.in1d(studentClasses[self.CLASS_CODE_COLUMN], classSet)]
      sortedClasses = correctClasses.sort_values(self.TERM_COLUMN)
      if 'termOrder' in self.df.columns:
        print('sorting')
        sortedClasses['termOrder'] = sortedClasses['termOrder'].apply(float)
        sortedClasses = sortedClasses.sort_values('termOrder')
      sortedClasses.reset_index(inplace = True)
      firstClass = False
      currentTerm = None
      i = 0
      j = 0
      rounded = lambda x: round(float(x),2)
      # print(sortedClasses[self.CLASS_CODE_COLUMN])
      # print(sortedClasses['termOrder'])
      while i < sortedClasses.index[-1]:
        if sortedClasses.iloc[i][self.CLASS_CODE_COLUMN] in courseGroups[count]:
          firstClass = True
          currentTerm = sortedClasses.iloc[i][self.TERM_COLUMN]
          if termThreshold:
            termNum = rounded(sortedClasses.iloc[i]['termOrder'])
          count += 1
          break
        i += 1
      if firstClass:
        if not termThreshold:
          while i < sortedClasses.index[-1]:
            if sortedClasses.iloc[i][self.TERM_COLUMN] != currentTerm:
              currentTerm = sortedClasses.iloc[i][self.TERM_COLUMN]
              count += 1
            if count >= len(courseGroups):
              break
            if sortedClasses.iloc[i][self.CLASS_CODE_COLUMN] not in courseGroups[count-1]:
              i += 1
              continue
            j = i + 1
            while j <= sortedClasses.index[-1]:
              if sortedClasses.iloc[j][self.TERM_COLUMN] == currentTerm:
                j += 1
                continue
              nextTerm = sortedClasses.iloc[j][self.TERM_COLUMN]
              break
            while j <= sortedClasses.index[-1] and sortedClasses.iloc[j][self.TERM_COLUMN] == nextTerm:
              if (sortedClasses.iloc[j][self.CLASS_CODE_COLUMN] in courseGroups[count]):
                addEdge(sortedClasses.iloc[i], sortedClasses.iloc[j], count)
              j += 1
            i += 1
        else:
          # print(sortedClasses)
          while i < sortedClasses.index[-1]:
            if rounded(sortedClasses.iloc[i]['termOrder']) != rounded(termNum):
              termNum = rounded(sortedClasses.iloc[i]['termOrder'])
              count += 1
            if count >= len(courseGroups):
              break
            if sortedClasses.iloc[i][self.CLASS_CODE_COLUMN] not in courseGroups[count-1]:
              i += 1
              continue
            j = i + 1
            while j <= sortedClasses.index[-1]:
              if rounded(sortedClasses.iloc[j]['termOrder']) == rounded(termNum):
                j += 1
                continue
              break
            while j <= sortedClasses.index[-1] and rounded(rounded(sortedClasses.iloc[j]['termOrder']) - termNum) <= rounded(termThreshold):
              if (sortedClasses.iloc[j][self.CLASS_CODE_COLUMN] in courseGroups[count]):
                addEdge(sortedClasses.iloc[i], sortedClasses.iloc[j], count)
              j += 1
            i += 1
        # for row in sortedClasses.index[1:]:
        #   current = sortedClasses.iloc[row]
        #   if lastClass[self.TERM_COLUMN] == current[self.TERM_COLUMN]:
        #     continue
        #   if current[self.CLASS_CODE_COLUMN] in courseGroups[count]:
        #     first = sortedClasses.iloc[row]
        #     second = sortedClasses.iloc[row+1]
        #     if first[self.TERM_COLUMN] != second[self.TERM_COLUMN]:
        #       lastClass = first   
        #       sameTerm = True
        #       count += 1
        #     if first[self.TERM_COLUMN] != second[self.TERM_COLUMN] and second[self.CLASS_CODE_COLUMN] in courseGroups[count]:
        #       addEdge(first, second, count)
        #       count += 1
        #       if count >= len(courseGroups) - 1:
        #         break
        #     elif sameTerm and second[self.CLASS_CODE_COLUMN] in courseGroups[count]:
        #       addEdge(lastClass, second, count+1)
        # lastTerm = None
        # for row in sortedClasses.index[:-1]:
        #   first = sortedClasses.iloc[row]

    edgeList = []
    skippedEdges = {}
    # print(edges)
    # print(minEdgeValue)
    if minEdgeValue:
      for key, value in edges.items():
        if value < minEdgeValue:
          if key[1] in skippedEdges:
            skippedEdges[key[1]] += value
          else:
            skippedEdges[key[1]] = value
    for key, value in edges.items():
      if minEdgeValue:
        if key[0] in skippedEdges:
          value -= skippedEdges[key[0]]
        if value < minEdgeValue:
          continue
      temp = [key[0], key[1], value]
      edgeList.append(temp)
    sankey = hv.Sankey(edgeList, ['From', 'To'])
    sankey.opts(width=600, height=400, node_padding=40, edge_color_index='From', color_index='index', title=graphTitle)
    graph = hv.render(sankey)
    output_file(outDir +outputName + '.html', mode='inline')
    save(graph)
    show(graph)

  def substituteSubStrInColumn(self, column, subString, substitute):
    """Replace a substring in a given column.

      Args:
        column (:obj:`str`): Column to replace substring in.
        subString (:obj:`str`): Substring to replace.
        substitute (:obj:`str`): Replacement of the substring.

    """
    self.convertColumnToString(column)
    self.df[column] = self.df[column].str.replace(subString, substitute)
    # print(self.df[column])

  def instructorRanks(self, firstClass, secondClass, fileName = 'instructorRanking', minStudents = 1):
    """Create a table of instructors and their calculated benefit to students based on a class they taught and future performance in a given class taken later. Exports a CSV file and returns a pandas dataframe.

      Args:
        firstClass (:obj:`str`): Class to look at instructors / their students from.
        secondClass (:obj:`str`): Class to look at future performance of students who had relevant professors from the first class.
        fileName (:obj:`str`, optional): Name of CSV file to save. Set to 'instructorRanking' by default.
        minStudents (:obj:`int`, optional): Minimum number of students to get data from for an instructor to be included in the calculation. Set to 1 by default.

      Returns:
        :obj:`pandas.dataframe`: Pandas dataframe with columns indicating the instructor, the normalized benefit to students, the grade point benefit to students, and the number of students used to calculate for that instructor.

    """
    if not self._gradeDataHelper__requiredColumnPresent(self.FACULTY_ID_COLUMN):
      return
    if self.NORMALIZATION_COLUMN not in self.df.columns:
      self.getNormalizationColumn()
    if not self._gradeDataHelper__requiredColumnPresent(self.NORMALIZATION_COLUMN):
      return
    firstClassEntries = self.df.loc[self.df[self.CLASS_CODE_COLUMN] == firstClass]
    secondClassEntries = self.df.loc[self.df[self.CLASS_CODE_COLUMN] == secondClass]

    instructors = firstClassEntries[self.FACULTY_ID_COLUMN].unique()
    instructorRank = {}
    instructorNorm = {}
    instructorStudCount = {}

    for instructor in instructors:
      instructorStudCount[instructor] = 0
      tookInstructor = firstClassEntries.loc[firstClassEntries[self.FACULTY_ID_COLUMN] == instructor]
      studentsWithInstructor = tookInstructor[self.STUDENT_ID_COLUMN].unique()
      secondClassWithPastInstructor = secondClassEntries[self.STUDENT_ID_COLUMN].isin(studentsWithInstructor)
      if any(secondClassWithPastInstructor):
        instructorStudCount[instructor] = sum(secondClassWithPastInstructor)
        entriesWithPastInstructor = secondClassEntries.loc[secondClassWithPastInstructor]
        entriesWithoutPastInstructor = secondClassEntries.loc[~secondClassWithPastInstructor]
        AverageGradeWithInstructor = entriesWithPastInstructor[self.FINAL_GRADE_COLUMN].mean()
        AverageGradeWithoutInstructor = entriesWithoutPastInstructor[self.FINAL_GRADE_COLUMN].mean()
        stdDev = secondClassEntries[self.FINAL_GRADE_COLUMN].std()
        instructorRank[instructor] = (AverageGradeWithInstructor - AverageGradeWithoutInstructor) / stdDev
        instructorNorm[instructor] = entriesWithPastInstructor[self.NORMALIZATION_COLUMN].mean() - entriesWithoutPastInstructor[self.NORMALIZATION_COLUMN].mean()
    
    sortedRanks = sorted(instructorNorm.items(), key=lambda x: x[1], reverse=True)
    rankDf = pd.DataFrame(sortedRanks, columns=['Instructor(' + firstClass + ')', 'NormBenefit(' + secondClass + ')'])
    rankDf['GradeBenefit('+secondClass+')'] = rankDf['Instructor('+firstClass+')'].apply(lambda inst: instructorRank[inst])
    rankDf['#students'] = rankDf['Instructor('+firstClass+')'].apply(lambda inst: instructorStudCount[inst])    
    rankDf = rankDf.loc[rankDf['#students'] >= minStudents]
    if not fileName.endswith('.csv'):
      fileName = "".join((fileName, '.csv'))
    rankDf.to_csv(fileName, index=False)
    return rankDf

  def gradePredict(self, priorGrades, futureClasses, method='nearest', excludedStudents = None, normalized = False):
    """Predicts grades given a student's past grades and classes to predict for. Still being developed.

      Args:
        priorGrades (:obj:`dict`(:obj:`str` : :obj:`float`)): Dictionary of past courses and the respective grade recieved.
        futureClasses (:obj:`list`(:obj:`str`)): List of courses to predict grades for.
        method (:obj:`str`, optional): Method to use to predict grades. Current methods include 'nearest' which gives the grade recieved by the most similar student on record and 'nearestThree' which gives the grade closest to the mean of the grades recieved by the nearest three students on record. Set to 'nearest' by default.
        excludedStudents (:obj:`list`, optional): List of students to exclude when making calculation. Used for accuracy testing purposes. Set to :obj:`None` by default.
        normalized (:obj:`bool`, optional): Whether or not normalized grades are given as input. Used for accuracy testing purposes and should generally be set to :obj:`False`. Set to :obj:`False` by default.

      Returns:
        :obj:`dict`(:obj:`str` : :obj:`float`): Dictionary of grade predictions, where the key is the class and the value is the grade predicted.

    """
    if not self._gradeDataHelper__requiredColumnPresent(self.STUDENT_ID_COLUMN):
      return
    if self.CLASS_CODE_COLUMN not in self.df.columns:
      self.getUniqueIdentifiersForSectionsAcrossTerms()
      if not self._gradeDataHelper__requiredColumnPresent(self.CLASS_CODE_COLUMN):
        return
    if normalized:
      if self.NORMALIZATION_COLUMN not in self.df.columns:
        self.getNormalizationColumn()
        if not self._gradeDataHelper__requiredColumnPresent(self.NORMALIZATION_COLUMN):
          return
    else:
            # JH: check FINAL_GRADE_COLUMN pesent
      if not self.__requiredColumnPresent(self.FINAL_GRADE_COLUMN):
        return
      self.dropMissingValuesInColumn(self.FINAL_GRADE_COLUMN)
      self.convertColumnToNumeric(self.FINAL_GRADE_COLUMN)
    print('Mode set to ' + method)
    relevantClasses = list(priorGrades.keys()) + futureClasses
    relevantEntries = self.df.loc[self.df[self.CLASS_CODE_COLUMN].isin(relevantClasses)]
    if excludedStudents:
      relevantEntries = relevantEntries.loc[~(relevantEntries[self.STUDENT_ID_COLUMN].isin(excludedStudents))]
    studentDict = relevantEntries.groupby(self.STUDENT_ID_COLUMN)
    prediction = {}
    if normalized:
      priorGradeComparison = lambda g: 10 - abs(priorGrades[g[self.CLASS_CODE_COLUMN]] - g[self.NORMALIZATION_COLUMN])
      def priorGradeDistance(grades, prior):
        A = grades[self.NORMALIZATION_COLUMN].values
        B = np.array([prior[x] for x in grades[self.CLASS_CODE_COLUMN].values])
        return np.linalg.norm(A - B)
    else:
      priorGradeComparison = lambda g: 1 - abs(priorGrades[g[self.CLASS_CODE_COLUMN]] - g[self.FINAL_GRADE_COLUMN])
      def priorGradeDistance(grades, prior):
        A = grades[self.FINAL_GRADE_COLUMN].values
        B = np.array([prior[x] for x in grades[self.CLASS_CODE_COLUMN].values])
        return np.linalg.norm(A - B)
    validGrades = self.df[self.FINAL_GRADE_COLUMN].unique()
    validGrades.sort()
    def gradePachinko(grades):
      meanGrade = sum(grades) / len(grades)
      idx = validGrades.searchsorted(meanGrade)
      idx = np.clip(idx, 1, len(validGrades)-1)
      left = validGrades[idx-1]
      right = validGrades[idx]
      idx -= meanGrade - left < right - meanGrade
      return validGrades[idx]

    for futureClass in futureClasses:
      print('Calculating for ' + futureClass + '...')
      futureClassEntries = relevantEntries.loc[relevantEntries[self.CLASS_CODE_COLUMN] == futureClass]
      relevantStudents = futureClassEntries[self.STUDENT_ID_COLUMN].unique()
      internalScore = {}
      numCommonClasses = {}
      # counter = 0
      # outOf = len(relevantStudents)
      for student in relevantStudents:
        # counter += 1
        # print(str(counter) + '/' + str(outOf) + ' students')
        commonClass = studentDict.get_group(student)[self.CLASS_CODE_COLUMN].isin(priorGrades.keys())
        numCommonClasses[student] = sum(commonClass)
        if numCommonClasses[student] > 0:
          relevantClasses = studentDict.get_group(student).loc[commonClass]
          applied = relevantClasses.apply(priorGradeComparison, axis=1)
          internalScore[student] = np.sum(applied.values)
          # internalScore[student] = priorGradeDistance(relevantClasses, priorGrades)
      if method == 'nearest' or len(internalScore) < 2:
        mostRelevant = max(internalScore, key=internalScore.get)
        # mostRelevant = min(internalScore, key=internalScore.get)
        # print(studentDict.get_group(mostRelevant)[[self.CLASS_CODE_COLUMN, self.FINAL_GRADE_COLUMN]])
        prediction[futureClass] = futureClassEntries.loc[futureClassEntries[self.STUDENT_ID_COLUMN] == mostRelevant].iloc[0][self.FINAL_GRADE_COLUMN]
      elif method == 'nearestThree':
        mostRelevant = sorted(internalScore, key=internalScore.get, reverse=True)[:3]
        # mostRelevant = sorted(internalScore, key=internalScore.get, reverse=False)[:3]
        relevantScores = [futureClassEntries.loc[futureClassEntries[self.STUDENT_ID_COLUMN] == stud].iloc[0][self.FINAL_GRADE_COLUMN] for stud in mostRelevant]
        prediction[futureClass] = gradePachinko(relevantScores)


    print(str(prediction)[1:-1])
    return prediction

  def coursePairGraph(self, courseOne, courseTwo, fileName = 'coursePairGraph'):
    print('Creating course grade graph between ' + courseOne + ' and ' + courseTwo + '...')
    if not self._gradeDataHelper__requiredColumnPresent(self.STUDENT_ID_COLUMN):
      return
    if not self._gradeDataHelper__requiredColumnPresent(self.FINAL_GRADE_COLUMN):
      return
    if self.CLASS_CODE_COLUMN not in self.df.columns:
      self.getUniqueIdentifiersForSectionsAcrossTerms()
      if not self._gradeDataHelper__requiredColumnPresent(self.CLASS_CODE_COLUMN):
        return
    self.dropNullAndConvertToNumeric(self.FINAL_GRADE_COLUMN)
    classOneEntries = self.df.loc[self.df[self.CLASS_CODE_COLUMN] == courseOne]
    classTwoEntries = self.df.loc[self.df[self.CLASS_CODE_COLUMN] == courseTwo]
    studentsInClass = np.intersect1d(classOneEntries[self.STUDENT_ID_COLUMN].values,classTwoEntries[self.STUDENT_ID_COLUMN].values)
    classOneEntries.drop_duplicates(self.STUDENT_ID_COLUMN, keep='last', inplace=True)
    classTwoEntries.drop_duplicates(self.STUDENT_ID_COLUMN, keep='last', inplace=True)
    relevantEntriesOne = classOneEntries.loc[classOneEntries[self.STUDENT_ID_COLUMN].isin(studentsInClass)]
    relevantEntriesTwo = classTwoEntries.loc[classTwoEntries[self.STUDENT_ID_COLUMN].isin(studentsInClass)]
    relevantEntriesOne.sort_values(self.STUDENT_ID_COLUMN, inplace = True)
    relevantEntriesTwo.sort_values(self.STUDENT_ID_COLUMN, inplace = True)
    relevantEntriesOne.reset_index(inplace = True, drop=True)
    relevantEntriesTwo.reset_index(inplace = True, drop=True)
    relevantEntriesOne.rename(columns={self.FINAL_GRADE_COLUMN: courseOne}, inplace=True)
    relevantEntriesOne[courseTwo] = relevantEntriesTwo[self.FINAL_GRADE_COLUMN]
    combined = relevantEntriesOne.groupby([courseOne, courseTwo]).size().reset_index(name="freq")
    # print(combined)
    combined.to_csv(outDir+fileName+'.csv', index=False)
    # print(combined.values)
    n = combined['freq'].sum()
    combined['freq'] = combined['freq'] * (500 / n)
    points = hv.Points(combined.values, vdims=['frequency'])
    points.opts(opts.Points(size='frequency', xlabel=courseOne, ylabel=courseTwo, title=courseOne + ' Vs. ' + courseTwo + ' Grades'))
    hv.output(size=200)
    graph = hv.render(points)
    graph.add_layout(Title(text='n = ' + str(n), text_font_style="italic", text_font_size="10pt"), 'above')
    output_file(outDir + fileName + '.html', mode='inline')
    print('Exported ' + courseOne + ' and ' + courseTwo + ' grade graph to ' + fileName + '.html')
    save(graph)
    show(graph)
    hv.output(size=125)

  def instructorRanksAllClasses(self, fileName = 'completeInstructorRanks', minStudents = 20, directionality = 0.8, outputSubjectAverages = False, subjectFileName = 'instructorAverages', otherRank = None):
    """Create a table of instructors and their calculated benefit to students based on all classes they taught and future performance in all classes taken later. Exports a CSV file and returns a pandas dataframe.

      Args:
        fileName (:obj:`str`, optional): Name of CSV file to save. Set to 'completeInstructorRanks' by default.
        minStudents (:obj:`int`, optional): Minimum number of students to get data from for an instructor's entry to be included in the calculation. Set to 1 by default.
        directionality (:obj:`float`, optional): Minimum directionality (percentage of students who took one class before another). Range 0.0 to 1.0. Set to 0.8 by default.
        outputSubjectAverages (:obj:`bool`, optional): Output a file with averages of all the data in this file, by instructor, by subject. Set to :obj:`False` by default.
        subjectFileName (:obj:`str`, optional): File to output instructor/subject averages to. Set to 'instructorAverages' by default.

      Returns:
        :obj:`pandas.dataframe`: Pandas dataframe with columns indicating the instructor, the class taken, the future class, the normalized benefit to students, the grade point benefit to students, the number of students used to calculate for that instructor / class combination, as well as the number of students on the opposite side of that calculation (students in future class who did not take that instructor before).

    """
    # function goes through every possible pair of classes
    # minStudents is set to 20 to make sure at least 20 students from the initial class with a professor went on to take the second class. This ensures that results are not skewed by too few students
    # directionality is set to .8 to make sure that at least 80 percent of students that took both classes took them in the same order
   
    if not self._gradeDataHelper__requiredColumnPresent(self.FACULTY_ID_COLUMN): # checks to see if faculty ids are present 
      return
    if self.NORMALIZATION_COLUMN not in self.df.columns:
      self.getNormalizationColumn()
    if not self._gradeDataHelper__requiredColumnPresent(self.NORMALIZATION_COLUMN): # checks which classes and class sections are being looked at 
      return
          # JH: never tests column present self.FINAL_GRADE_COLUMN
    self.dropNullAndConvertToNumeric(self.FINAL_GRADE_COLUMN) # makes sure final grade is numeric
    self.dropNullAndConvertToNumeric(self.NORMALIZATION_COLUMN) # makes sure normalization is numeric
    print('here')
    
    if directionality > 1.0 or directionality < 0.5: 
      print('Error: directionality out of bounds (must be between 0.5 to 1, not '+ str(directionality) +').')
    if otherRank is not None:
      if not self._gradeDataHelper__requiredColumnPresent(otherRank):
        return
      self.dropNullAndConvertToNumeric(otherRank)
    print('here')
    rowList = []
    
    # processPair function - takes two classes and finds all relevant faculty scores. Scope is inside instructorRanksAllClasses
    def processPair(classOne, classTwo, df):
      firstClass = df[self.CLASS_CODE_COLUMN] == classOne # looks at the whole dataframe and makes a series of true/false values, compares each value to the name of the first class (true if it is the same)
      firstClassEntries = df.loc[firstClass] # makes separate dataframe for first class
      secondClassEntries = df.loc[~firstClass] # makes separate dataframe for second class
      instructorDict = firstClassEntries[self.FACULTY_ID_COLUMN].value_counts().to_dict() # looks at all faculty that taught the first class, gets a number of students for each instructor
      instructors = {key:val for key, val in instructorDict.items() if val >= minStudents} # defines a map of values, removes any instructor that taught less than the minimum amount of students
      
      if not instructors: # if the threshold is not met, function ends
        return
      for instructor, count in instructors.items(): # loops through instructors that met the requirement
        tookInstructor = firstClassEntries.loc[firstClassEntries[self.FACULTY_ID_COLUMN] == instructor] # filters through the dataframe of the first class down to the students who took the specific instructor
        studentsWithInstructor = tookInstructor[self.STUDENT_ID_COLUMN].unique() # gets list of students that took the specific instructor
        secondClassWithPastInstructor = secondClassEntries[self.STUDENT_ID_COLUMN].isin(studentsWithInstructor) # takes all students that took the instructor for the first class and filters through the dataframe of the second class down to the same students
        newCount = sum(secondClassWithPastInstructor) # finds how many students took both classes
        nonStudents = len(secondClassWithPastInstructor.index) - newCount # finds how many students took the second class but did not have the first instructor
       
      if nonStudents > 0: # ensures that there are some students that did not take the instructor
        stdDev = secondClassEntries[self.FINAL_GRADE_COLUMN].std()
         
        if stdDev > 0: # ensures that every student did not recieve the same grade
            entriesWithPastInstructor = secondClassEntries.loc[secondClassWithPastInstructor] # makes a dataframe from the second class dataframe of people who took the instructor  
            entriesWithoutPastInstructor = secondClassEntries.loc[~secondClassWithPastInstructor] # makes a dataframe from the second class dataframe of people who did not take the instructor 
            AverageGradeWithInstructor = entriesWithPastInstructor[self.FINAL_GRADE_COLUMN].mean() # gets mean grade from group of people in the second class who took the instructor
            AverageGradeWithoutInstructor = entriesWithoutPastInstructor[self.FINAL_GRADE_COLUMN].mean() # gets mean grade from group of people in the second class who did not take the instructor
            rowDict = {}
            rowDict['Instructor'] = instructor
            rowDict['courseTaught'] = classOne
            rowDict['futureCourse'] = classTwo
            rowDict['normBenefit'] = entriesWithPastInstructor[self.NORMALIZATION_COLUMN].mean() - entriesWithoutPastInstructor[self.NORMALIZATION_COLUMN].mean() # subtracts the mean of students who did not take the instructor from the mean of students who took the instructor
            rowDict['gradeBenefit'] = (AverageGradeWithInstructor - AverageGradeWithoutInstructor) / stdDev # subtracts the average of students who did not take the instructor from the average of students who took the instructor and divides by standard deviation (because it is not centered at 0)
            
            if otherRank is not None: # otherRank allows one to include some other value in a different column
              rowDict[otherRank] = entriesWithPastInstructor[otherRank].mean() - entriesWithoutPastInstructor[otherRank].mean()
            rowDict['#students'] = newCount
            rowDict['#nonStudents'] = nonStudents
            rowList.append(rowDict) # rowList is a list of dictionaries
    print('here')
    classes = self.df[self.CLASS_CODE_COLUMN].unique().tolist() # gets all classes
    numClasses = len(classes)
    grouped = self.df.groupby(self.CLASS_CODE_COLUMN) # makes a specific dataframe for each class 
    
    for name, group in grouped:
      group.sort_values(self.TERM_COLUMN, inplace = True) # sorts classes by the term they occurred in
      group.drop_duplicates(self.STUDENT_ID_COLUMN, keep='last', inplace=True) # gets rid of any duplicates of a student taking a class more than once, keeps the last time
    for i in range(numClasses - 1): # loops over all classes to compare pairs
      print('class ' + str(i+1) + '/' + str(numClasses))
      classOne = classes[i]
      oneDf = grouped.get_group(classOne)
      start_time = time.time()
      for j in range(i + 1, numClasses): # loops through the classes again, to get specific data frames for both classes
        classTwo = classes[j]
        twoDf = grouped.get_group(classTwo)
        studentInClass = np.intersect1d(oneDf[self.STUDENT_ID_COLUMN].values,twoDf[self.STUDENT_ID_COLUMN].values) # finds how many students took both classes at some point
        if len(studentInClass) >= minStudents: # makes sure it is over the threshold
          combinedEntries = pd.concat([oneDf, twoDf], ignore_index=True) # combines the two data frames for the classes
          relevantEntries = combinedEntries.loc[combinedEntries[self.STUDENT_ID_COLUMN].isin(studentInClass)] # filters down to the common students 
          relevantEntries.sort_values(self.TERM_COLUMN, inplace = True) # sorts the entries for both classes by term
          firstEntries = relevantEntries[[self.STUDENT_ID_COLUMN, self.CLASS_CODE_COLUMN]].drop_duplicates(self.STUDENT_ID_COLUMN) # drops duplicates (might be redundant?)
          classOneFirstCount = sum(firstEntries[self.CLASS_CODE_COLUMN] == classOne) # finds how many students took class one first
          directionOne = classOneFirstCount / (len(firstEntries.index)) # compares how many students took class one first to the total, in order to find directionality
          if directionOne >= directionality: 
            processPair(classOne, classTwo, relevantEntries)
          if (1.0 - directionOne) >= directionality: # if it is not over the threshold, the classes orders get reversed
            processPair(classTwo, classOne, relevantEntries)
      # print('outerEnd: ' + str(time.time() - start_time))      
    
    if otherRank is None:
      completeDf = pd.DataFrame(rowList, columns=['Instructor','courseTaught','futureCourse','normBenefit','gradeBenefit','#students', '#nonStudents'])
    else:
      completeDf = pd.DataFrame(rowList, columns=['Instructor','courseTaught','futureCourse','normBenefit','gradeBenefit', otherRank, '#students', '#nonStudents'])
    completeDf.sort_values(by=['futureCourse','courseTaught','Instructor']) # sorts columns alphabetically
    completeDf['Instructor'].replace(' ', np.nan, inplace=True)
    completeDf.dropna(subset=['Instructor'], inplace=True)
    completeDf.reset_index(inplace = True, drop=True)
    completeDf['totalStudents'] = (completeDf['#students'].apply(int)) + (completeDf['#nonStudents'].apply(int))
    completeDf['%ofStudents'] = ((completeDf['#students'].apply(float)) / (completeDf['totalStudents'].apply(float))) * 100
    completeDf['grade*Norm*Sign(norm)'] = completeDf['gradeBenefit'] * completeDf['normBenefit'] * np.sign(completeDf['normBenefit'])
    completeDf['normBenefit' + pvalSuffix] = pvalOfSeries(completeDf['normBenefit'])
    completeDf['gradeBenefit' + pvalSuffix] = pvalOfSeries(completeDf['gradeBenefit'])
    completeDf['grade*Norm*Sign(norm)' + pvalSuffix] = pvalOfSeries(completeDf['grade*Norm*Sign(norm)'])
  
    if otherRank is not None:
      completeDf[otherRank + pvalSuffix] = pvalOfSeries(completeDf[otherRank])
    if not fileName.endswith('.csv'):
      fileName = "".join((fileName, '.csv'))
    completeDf.to_csv(fileName, index=False)
    if outputSubjectAverages:
      instructorAveraging(completeDf, subjectFileName)
    return completeDf # outputs completed dataframe

  #Note: Has an error when sequenceDetails = True if numexpr is not installed
  def getCorrelationsWithMinNSharedStudents(self, nSharedStudents = 20, directed = False, classDetails = False, sequenceDetails = False, semsBetweenClassesLimit = -1, compoundNormalization = False):
    """Returns a pandas dataframe with correlations between all available classes based on grades, after normalization.

    Args:
        nSharedStudents (:obj:`int`, optional): Minimum number of shared students a pair of classes must have to compute a correlation. Defaults to 20.
        directed (:obj:`bool`, optional): Whether or not to include data specific to students who took class A before B, vice versa, and concurrently. Defaults to 'False'.
        classDetails (:obj:`bool`, optional): Whether or not to include means of student grades, normalized grades, and standard deviations used. Defaults to 'False'.
        semsBetweenClassesLimit (:obj:`int`, optional): Maximum number of semesters that a student can take between two classes. If negative there is no limit. Defaults to -1.
        compundNormalization (:obj:`bool`, optional): When true normalizes grades by class and by student. When false normalizes grades by class only. Defaults to 'False'.

    Returns:
        :obj:`pandas.dataframe`: Pandas dataframe with at least columns "course1", "course2", "corr", "P-value", and "#students", which store class names, their correlation coefficient (0 least to 1 most), the P-value of this calculation, and the number of students shared between these two classes.

    """
    print("Getting correlations...")
    #Keep track of time to output how long the function takes at the end
    start_time = time.time()
    #Change nSharedStudents to 2 if it is less than 2
    nSharedStudents = max(nSharedStudents, 2)

    #Create a column of normalized grades in self
    if self.NORMALIZATION_COLUMN not in self.df.columns:
      self.getNormalizationColumn()
    if "semNumber" not in self.df.columns:
      self.makeSemesterNumberColumn()
    if "studentGPA" not in self.df.columns:
      self.makeStudentGPAColumn()
    if compoundNormalization:
      if self.STUDENT_CLASS_NORMALIZATION_COLUMN not in self.df.columns:
        self.getNormalizationByStudentByClass()
      self.df["norm"] = self.df["normByStudentByClass"]

    #The following code ensures that all required columns exist before continuing
    if not self._gradeDataHelper__requiredColumnPresent(self.NORMALIZATION_COLUMN):
      return
    if not self._gradeDataHelper__requiredColumnPresent(self.STUDENT_ID_COLUMN):
      return
    if not self._gradeDataHelper__requiredColumnPresent(self.STUDENT_ID_COLUMN):
      return
    if directed:
      if not self._gradeDataHelper__requiredColumnPresent(self.TERM_COLUMN):
        return
      self.df[self.TERM_COLUMN] = pd.to_numeric(self.df[self.TERM_COLUMN],errors='ignore')
    if sequenceDetails:
      if not self._gradeDataHelper__requiredColumnPresent(self.STUDENT_YEAR_COLUMN):
        return
      self.df[self.STUDENT_YEAR_COLUMN] = pd.to_numeric(self.df[self.STUDENT_YEAR_COLUMN],errors='ignore')

    #Create a list classes with each unique class
    print("Getting classes...")
    classes = self.getListOfClassCodes()
    #d is a dictionary with keys as classes and values as dataframes with all the data of students in that respective class
    d={}
    print("Organizing classes...")
    #Check that a class code column exists (not sure why this isn't with the other checks at the top)
    if not self._gradeDataHelper__requiredColumnPresent(self.CLASS_CODE_COLUMN):
      return
    
    if classDetails:
      #Create dictionaries for grades, normalized grades, and standard deviations of grades
      rawGrades = {}
      normalizedGrades = {}
      stdDevGrade = {}
      #Convert grade columns to numeric values
      self.convertColumnToNumeric(self.FINAL_GRADE_COLUMN)
      self.convertColumnToNumeric(self.NORMALIZATION_COLUMN)
      self.convertColumnToNumeric(self.GPA_STDDEV_COLUMN)
        
    #Fill out the keys of d as classes and the values of d as a dataframe of all students in that class
    for n, group in self.df.groupby(self.CLASS_CODE_COLUMN):

      d["df{0}".format(n)] = group
      d["df{0}".format(n)] = d["df{0}".format(n)].drop_duplicates(subset="SID", keep=False)
      if classDetails:
        raw = d["df{0}".format(n)][self.FINAL_GRADE_COLUMN].tolist()
        rawGrades[n] = str(sum(raw) / len(raw))
        stdDevGrade[n] = str(d["df{0}".format(n)][self.GPA_STDDEV_COLUMN].iloc[0])
        # print(rawGrades[n])
        # print(normalizedGrades[n])
        # print(stdDevGrade[n])

    f = []
    classCount = 0
    totalClasses = len(classes)
    print("Sorting classes...")
    classes.sort()
    #Declare classesProcessed as an empty set
    classesProcessed = set()
    #Cycle through each n in classes
    for n in classes:
      classCount = classCount + 1
      print("class " + str(classCount) + "/" + str(totalClasses))
      tim = time.time()
      #Cycle through each class m in classes
      for m in classes:
        # skip pair if there is less than N students in a class (Filter to make the program run faster)
        if (len(d["df{0}".format(m)]) < nSharedStudents) or (len(d["df{0}".format(n)]) < nSharedStudents):
          continue
        # Skip all pairs of identical classes
        if (m == n):
          continue
        #Checks to make sure this is a new pair of classes
        if m not in classesProcessed:
          #If not directed, use corAlg to generate correlation r, p-value p, and number of students c, then append them all to f
          if not directed:
            classesProcessed.add(n)
            result = self.corrAlg(d["df{0}".format(n)],d["df{0}".format(m)], nSharedStudents, directed, classDetails, sequenceDetails, semsBetweenClassesLimit)
            r, p, c = result[0], result[1], result[2]
            if not math.isnan(r):
              f.append((n, m, r, p, c))
          #If directed, use corrAlgDirected to generate a lot of data and add it to f
          else:
            classesProcessed.add(n)
            result = self.corrAlgDirected(d["df{0}".format(n)],d["df{0}".format(m)], nSharedStudents, directed, classDetails, sequenceDetails, semsBetweenClassesLimit)
            r, p, c, r1, p1, c1, r2, p2, c2, r3, p3, c3, ag, bg, asg, bsg, an, bn, asn, bsn, abadevg, abbdevg, baadevg, babdevg, abadevn, abbdevn, baadevn, babdevn = result[:28]
            #Only gets the last columns of the data if the correlation exists
            if not math.isnan(r):
              if classDetails and sequenceDetails:
                abA, abANorm, abB, abBNorm, baA, baANorm, baB, baBNorm, concA, concANorm, concB, concBNorm, abAGPAMean, baBGPAMean = result[28:42]
                avNormDifCrs1Fresh, avNormDifCrs2Fresh, avNormDifCrs1Soph, avNormDifCrs2Soph, avNormDifCrs1Jun, avNormDifCrs2Jun, avNormDifCrs1Sen, avNormDifCrs2Sen, avGradeDifCrs1Fresh, avGradeDifCrs2Fresh, avGradeDifCrs1Soph, avGradeDifCrs2Soph, avGradeDifCrs1Jun, avGradeDifCrs2Jun, avGradeDifCrs1Sen, avGradeDifCrs2Sen, crs1FreshMin, crs2FreshMin, crs1SophMin, crs2SophMin, crs1JunMin, crs2JunMin, crs1SenMin, crs2SenMin = result[42:]
                f.append((n, m, r, p, c, ag, bg, asg, bsg, an, bn, asn, bsn, abadevg, abbdevg, baadevg, babdevg, abadevn, abbdevn, baadevn, babdevn, r1, p1, c1, abA, abANorm, abB, abBNorm, r2, p2, c2, baB, baBNorm, baA, baANorm, r3, p3, c3, concA, concANorm, concB, concBNorm, abAGPAMean, baBGPAMean, avNormDifCrs1Fresh, avNormDifCrs2Fresh, avNormDifCrs1Soph, avNormDifCrs2Soph, avNormDifCrs1Jun, avNormDifCrs2Jun, avNormDifCrs1Sen, avNormDifCrs2Sen, avGradeDifCrs1Fresh, avGradeDifCrs2Fresh, avGradeDifCrs1Soph, avGradeDifCrs2Soph, avGradeDifCrs1Jun, avGradeDifCrs2Jun, avGradeDifCrs1Sen, avGradeDifCrs2Sen, crs1FreshMin, crs2FreshMin, crs1SophMin, crs2SophMin, crs1JunMin, crs2JunMin, crs1SenMin, crs2SenMin))
                if n != m:
                  f.append((m, n, r, p, c, bg, ag, bsg, asg, bn, an, bsn, asn, babdevg, baadevg, abbdevg, abadevg, babdevn, baadevn, abbdevn, abadevn,r2, p2, c2, baB, baBNorm, baA, baANorm, r1, p1, c1, abA, abANorm, abB, abBNorm, r3, p3, c3, concB, concBNorm, concA, concANorm, baBGPAMean, abAGPAMean, -avNormDifCrs2Fresh, -avNormDifCrs1Fresh, -avNormDifCrs2Soph, -avNormDifCrs1Soph, -avNormDifCrs2Jun, -avNormDifCrs1Jun, -avNormDifCrs2Sen, -avNormDifCrs1Sen, -avGradeDifCrs2Fresh, -avGradeDifCrs1Fresh, -avGradeDifCrs2Soph, -avGradeDifCrs1Soph, -avGradeDifCrs2Jun, -avGradeDifCrs1Jun, -avGradeDifCrs2Sen, -avGradeDifCrs1Sen, crs2FreshMin, crs1FreshMin, crs2SophMin, crs1SophMin, crs2JunMin, crs1JunMin, crs2SenMin, crs1SenMin))
              elif classDetails:
                abA, abANorm, abB, abBNorm, baA, baANorm, baB, baBNorm, concA, concANorm, concB, concBNorm, abAGPAMean, baBGPAMean = result[28:]
                # print(n + " " + m + " " + str(r))
                f.append((n, m, r, p, c, ag, bg, asg, bsg, an, bn, asn, bsn, abadevg, abbdevg, baadevg, babdevg, abadevn, abbdevn, baadevn, babdevn,r1, p1, c1, abA, abANorm, abB, abBNorm, r2, p2, c2, baB, baBNorm, baA, baANorm, r3, p3, c3, concA, concANorm, concB, concBNorm, abAGPAMean, baBGPAMean))
                if n != m:
                  f.append((m, n, r, p, c, bg, ag, bsg, asg, bn, an, bsn, asn, babdevg, baadevg, abbdevg, abadevg, babdevn, baadevn, abbdevn, abadevn, r2, p2, c2, baB, baBNorm, baA, baANorm, r1, p1, c1, abA, abANorm, abB, abBNorm, r3, p3, c3, concB, concBNorm, concA, concANorm, baBGPAMean, abAGPAMean))
              else:
                f.append((n, m, r, p, c, ag, bg, asg, bsg, an, bn, asn, bsn, r1, p1, c1, r2, p2, c2, r3, p3, c3))
                if n != m:
                  f.append((m, n, r, p, c, bg, ag, bsg, asg, bn, an, bsn, asn, r2, p2, c2, r1, p1, c1, r3, p3, c3))
      #add n to classesProcessed and print the time it took to generate the correlation data            
      classesProcessed.add(n)
      print(str(time.time() - tim))
    #After cycling through each class
    f[:] = [x for x in f if isinstance(x[0], str)]
    f.sort(key = lambda x: x[1])
    f.sort(key = lambda x: x[0])
    #Create a DataFrame using data in f based on the function arguments
    #Updated with the correct number of columns for some cases
    if not directed:
      normoutput = pd.DataFrame(f, columns=('course1', 'course2', 'corr', 'P-value', '#students'))
    else:
      if not classDetails:
        normoutput = pd.DataFrame(f, columns=('course1', 'course2', 'corr', 'P-value', '#students','avGrade1','avGrade2','stdDevGrade1','stdDevGrade2','avNorm1','avNorm2','stdDevNorm1','stdDevNorm2', 'corrCourse1->2', 'P-valueCrs1->2','#studentsCrs1->2', 'corrCourse2->1', 'P-valueCrs2->1','#studentsCrs2->1', 'corrCoursesConcurrent', 'P-valueCrsConcurrent','#studentsCrsConcurrent'))
      else:
        if not sequenceDetails:
          normoutput = pd.DataFrame(f, columns=('course1', 'course2', 'corr', 'P-value', '#students','avGrade1','avGrade2','stdDevGrade1','stdDevGrade2','avNorm1','avNorm2','stdDevNorm1','stdDevNorm2', 'crs1->2grdStdDev(crs1)','crs1->2grdStdDev(crs2)','crs2->1grdStdDev(crs1)','crs2->1grdStdDev(crs2)','crs1->2nrmStdDev(crs1)','crs1->2nrmStdDev(crs2)','crs2->1nrmStdDev(crs1)','crs2->1nrmStdDev(crs2)', 'corrCourse1->2', 'P-valueCrs1->2','#studentsCrs1->2', 'Av.GradeCrs1->2(crs1)', 'Av.NormCrs1->2(crs1)', 'Av.GradeCrs1->2(crs2)','Av.NormCrs1->2(crs2)','corrCourse2->1', 'P-valueCrs2->1','#studentsCrs2->1', 'Av.GradeCrs2->1(crs2)','Av.NormCrs2->1(crs2)', 'Av.GradeCrs2->1(crs1)', 'Av.NormCrs2->1(crs1)', 'corrCoursesConcurrent', 'P-valueCrsConcurrent','#studentsCrsConcurrent','Av.GradeConcurrent(crs1)','Av.NormConcurrent(crs1)','Av.GradeConcurrent(crs2)','Av.NormConcurrent(crs2)', 'Av.GPACrs1->2', 'Av.GPACrs2->1'))
        else:
          normoutput = pd.DataFrame(f, columns=('course1', 'course2', 'corr', 'P-value', '#students','avGrade1','avGrade2','stdDevGrade1','stdDevGrade2','avNorm1','avNorm2','stdDevNorm1','stdDevNorm2', 'crs1->2grdStdDev(crs1)','crs1->2grdStdDev(crs2)','crs2->1grdStdDev(crs1)','crs2->1grdStdDev(crs2)','crs1->2nrmStdDev(crs1)','crs1->2nrmStdDev(crs2)','crs2->1nrmStdDev(crs1)','crs2->1nrmStdDev(crs2)','corrCourse1->2', 'P-valueCrs1->2','#studentsCrs1->2', 'Av.GradeCrs1->2(crs1)', 'Av.NormCrs1->2(crs1)', 'Av.GradeCrs1->2(crs2)','Av.NormCrs1->2(crs2)','corrCourse2->1', 'P-valueCrs2->1','#studentsCrs2->1', 'Av.GradeCrs2->1(crs2)','Av.NormCrs2->1(crs2)', 'Av.GradeCrs2->1(crs1)', 'Av.NormCrs2->1(crs1)', 'corrCoursesConcurrent', 'P-valueCrsConcurrent','#studentsCrsConcurrent','Av.GradeConcurrent(crs1)','Av.NormConcurrent(crs1)','Av.GradeConcurrent(crs2)','Av.NormConcurrent(crs2)', 'Av.GPACrs1->2', 'Av.GPACrs2->1', 'Av.NormCrs1->2-Av.NormCrs2->1(crs1_fresh)', 'Av.NormCrs1->2-Av.NormCrs2->1(crs2_fresh)', 'Av.NormCrs1->2-Av.NormCrs2->1(crs1_soph)', 'Av.NormCrs1->2-Av.NormCrs2->1(crs2_soph)', 'Av.NormCrs1->2-Av.NormCrs2->1(crs1_jun)', 'Av.NormCrs1->2-Av.NormCrs2->1(crs2_jun)', 'Av.NormCrs1->2-Av.NormCrs2->1(crs1_sen)', 'Av.NormCrs1->2-Av.NormCrs2->1(crs2_sen)', 'Av.GradeCrs1->2-Av.GradeCrs2->1(crs1_fresh)', 'Av.GradeCrs1->2-Av.GradeCrs2->1(crs2_fresh)', 'Av.GradeCrs1->2-Av.GradeCrs2->1(crs1_soph)', 'Av.GradeCrs1->2-Av.GradeCrs2->1(crs2_soph)', 'Av.GradeCrs1->2-Av.GradeCrs2->1(crs1_jun)', 'Av.GradeCrs1->2-Av.GradeCrs2->1(crs2_jun)', 'Av.GradeCrs1->2-Av.GradeCrs2->1(crs1_sen)', 'Av.GradeCrs1->2-Av.GradeCrs2->1(crs2_sen)', 'crs1FreshMin', 'crs2FreshMin', 'crs1SophMin', 'crs2SophMin', 'crs1JunMin', 'crs2JunMin', 'crs1SenMin', 'crs2SenMin'))
    if classDetails:
      #Add more columns for classDetails
      normoutput['course1GradeMean'] = normoutput['course1'].apply(lambda x: rawGrades[x])
      normoutput['course2GradeMean'] = normoutput['course2'].apply(lambda x: rawGrades[x])
      normoutput['course1StdDev'] = normoutput['course1'].apply(lambda x: stdDevGrade[x])
      normoutput['course2StdDev'] = normoutput['course2'].apply(lambda x: stdDevGrade[x])
      normoutput['(Av.GradeCrs1->2(crs1)) - (Av.GradeCrs2->1(crs1))'] = normoutput['Av.GradeCrs1->2(crs1)'] - normoutput['Av.GradeCrs2->1(crs1)']
      normoutput['(Av.GradeCrs1->2(crs2)) - (Av.GradeCrs2->1(crs2))'] = normoutput['Av.GradeCrs1->2(crs2)'] - normoutput['Av.GradeCrs2->1(crs2)']
      #The following 2 columns are what is used in Tess's paper
      normoutput['(Av.NormCrs1->2(crs1)) - (Av.NormCrs2->1(crs1))'] = normoutput['Av.NormCrs1->2(crs1)'] - normoutput['Av.NormCrs2->1(crs1)']
      normoutput['(Av.NormCrs1->2(crs2)) - (Av.NormCrs2->1(crs2))'] = normoutput['Av.NormCrs1->2(crs2)'] - normoutput['Av.NormCrs2->1(crs2)']
      normoutput['(Av.GradeCrs1->2(crs1)) - (Av.GradeConcurrent(crs1))'] = normoutput['Av.GradeCrs1->2(crs1)'] - normoutput['Av.GradeConcurrent(crs1)']
      normoutput['(Av.GradeCrs1->2(crs2)) - (Av.GradeConcurrent(crs2))'] = normoutput['Av.GradeCrs1->2(crs2)'] - normoutput['Av.GradeConcurrent(crs2)']
      normoutput['(Av.GradeCrs2->1(crs1)) - (Av.GradeConcurrent(crs1))'] = normoutput['Av.GradeCrs2->1(crs1)'] - normoutput['Av.GradeConcurrent(crs1)']
      normoutput['(Av.GradeCrs2->1(crs2)) - (Av.GradeConcurrent(crs2))'] = normoutput['Av.GradeCrs2->1(crs2)'] - normoutput['Av.GradeConcurrent(crs2)']
      normoutput['(Av.NormCrs1->2(crs1)) - (Av.NormConcurrent(crs1))'] = normoutput['Av.NormCrs1->2(crs1)'] - normoutput['Av.NormConcurrent(crs1)']
      normoutput['(Av.NormCrs1->2(crs2)) - (Av.NormConcurrent(crs2))'] = normoutput['Av.NormCrs1->2(crs2)'] - normoutput['Av.NormConcurrent(crs2)']
      normoutput['(Av.NormCrs2->1(crs1)) - (Av.NormConcurrent(crs1))'] = normoutput['Av.NormCrs2->1(crs1)'] - normoutput['Av.NormConcurrent(crs1)']
      normoutput['(Av.NormCrs2->1(crs2)) - (Av.NormConcurrent(crs2))'] = normoutput['Av.NormCrs2->1(crs2)'] - normoutput['Av.NormConcurrent(crs2)']
      normoutput['(#studentsCrs1->2) - (#studentsCrs2->1)'] = normoutput['#studentsCrs1->2'] - normoutput['#studentsCrs2->1']
      normoutput['(#studentsCrs1->2) - (#studentsCrsConcurrent)'] = normoutput['#studentsCrs1->2'] - normoutput['#studentsCrsConcurrent']
      normoutput['(#studentsCrs1->2) / (Total # students)'] = normoutput['#studentsCrs1->2'] / normoutput['#students']
      normoutput['(#studentsCrs2->1) / (Total # students)'] = normoutput['#studentsCrs2->1'] / normoutput['#students']
      normoutput['(#studentsCrsConcurrent) / (Total # students)'] = normoutput['#studentsCrsConcurrent'] / normoutput['#students']

    #Remake normalization column to single normalization
    if compoundNormalization:
      self.getNormalizationColumn()

    #print details about the function
    print(str((totalClasses ** 2) - len(normoutput.index)) + ' class correlations dropped out of ' 
    + str(totalClasses ** 2) + ' from ' + str(nSharedStudents) + ' shared student threshold.')
    print(str(len(normoutput.index)) + ' correlations calculated. ' + str(time.time() - start_time) + ' seconds.')
    return normoutput


  def exportCorrelationsWithMinNSharedStudents(self, filename = 'CorrelationOutput_EDMLIB.csv', nStudents = 20, directedCorr = False, detailed = False, sequenced = False, semesterLimit = -1, compound = False):
    """Exports CSV file with all correlations between classes with the given minimum number of shared students. File format has columns 'course1', 'course2', 'corr', 'P-value', '#students'.

    Args:
        fileName (:obj:`str`, optional): Name of CSV to output. Default 'CorrelationOutput_EDMLIB.csv'.
        nStudents (:obj:`int`, optional): Minimum number of shared students a pair of classes must have to compute a correlation. Defaults to 20.
        directedCorr (:obj:`bool`, optional): Whether or not to include data specific to students who took class A before B, vice versa, and concurrently. Defaults to 'False'.
        detailed (:obj:`bool`, optional): Whether or not to include means of student grades, normalized grades, and standard deviations used. Defaults to 'False'.
        semesterLimit (:obj:`int`, optional): Maximum number of semesters that a student can take between two classes. If negative there is no limit. Defaults to -1.
        compound (:obj:`bool`, optional): When true normalizes grades by class and by student. When false normalizes grades by class only. Defaults to 'False'.

    """
    if not filename.endswith('.csv'):
      filename = "".join((filename, '.csv'))
    self.getCorrelationsWithMinNSharedStudents(nSharedStudents=nStudents, directed=directedCorr, classDetails = detailed, sequenceDetails = sequenced, semsBetweenClassesLimit = semesterLimit, compoundNormalization = compound).to_csv(filename, index=False)

  def exportCorrelationsWithAvailableClasses(self, filename = 'CorrelationOutput_EDMLIB.csv'):
    result = self.getCorrelationsWithMinNSharedStudents()
    if result:
      result.to_csv(filename, index=False)

  def getCorrelationsOfAvailableClasses(self):
    return classCorrelationData(self.getCorrelationsWithMinNSharedStudents())

  def exportCSV(self, fileName = 'csvExport.csv'):
    """Export the current state of the dataset to a :obj:`.CSV` file.
    
    Args:
        fileName (:obj:`str`, optional): Name of the file to export. Defaults to 'csvExport.csv'.

    """
    if not fileName.endswith('.csv'):
      fileName = "".join((fileName, '.csv'))
    self.df.to_csv(fileName, index=False)

