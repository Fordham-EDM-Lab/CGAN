"""
Library created by Michael Riad Zaky mriadzaky@fordham.edu (2020), with Daniel Leeds, Gary Weiss, Mavis Zhang at Fordham 
University. Library free for use provided you cite https://github.com/MichaelRZ/EDMLib in any resulting publications.  
Library free for redistribution provided you retain the author attributions above.
The following packages are required for installation before use: 
"""
import numpy as np
import pandas as pd
import csv
import sys
import math
import re, os
import networkx as nx
import itertools
import holoviews as hv
from holoviews import opts, dim
from bokeh.plotting import show, save, output_file
from bokeh.io import export_png
from bokeh.models import Title
from edmlib.edmlib import edmApplication, outDir

class classCorrelationData:
  """Class for manipulating and visualizing pearson correlations generated by the gradeData class.

    Attributes:
        df (:obj:`pandas.dataframe`): dataframe containing all correlational data.
        sourceFile (:obj:`str`): Name of source .CSV file with correlational data.

  """
  df = None
  sourceFile = ""

  def __init__(self, sourceFileOrDataFrame, copyDataFrame=True):
    """Class constructor, creates an instance of the class given a .CSV file or pandas dataframe. Typically should only be used manually with correlation files outputted by the gradeData class.

    Used with classCorrelationData('fileName.csv') or classCorrelationData(dataFrameVariable).

    Args:
        sourceFileOrDataFrame (:obj:`object`): name of the .CSV file (extension included) in the same path or pandas dataframe variable. Dataframes are copied so as to not affect the original variable.

    """
    # if the sourceFileOrDataFrame variable is of type str, then set sourceFile variable to sourceFileOrDataFrame 
    # and df variable to the contents of sourceFile
    if type(sourceFileOrDataFrame).__name__ == 'str':
      self.sourceFile = sourceFileOrDataFrame
      self.df = pd.read_csv(self.sourceFile)

    elif type(sourceFileOrDataFrame).__name__ == 'DataFrame':
    # JH: Make a copy if the caller doesn't want to change the dataFrame
    #     if not edmApplication:
      if not copyDataFrame:
        self.df = sourceFileOrDataFrame
      else:
        self.df = sourceFileOrDataFrame.copy()
  
  # M: returns the unique values in the column 'course1'
  def getClassesUsed(self):
    return self.df['course1'].unique()

  # M: returns the unique values in column 'course1' (Does not include NaN)
  def getNumberOfClassesUsed(self):
    return self.df['course1'].nunique()

  # M: returns the unique values in the column 'column'
  def printUniqueValuesInColumn(self, column):
    print(self.df[column].unique())

  def printClassesUsed(self):
    """Prints to console the classes included in the correlations.
    
    """
    self.printUniqueValuesInColumn('course1')

  # M: gets the number of rows
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
  
  def printMajors(self):
    """Prints to console the majors of the classes present in the correlational data.
    
    Note:
        The 'classDept' column as set by :obj:`defineWorkingColumns` must have been defined in your dataset to print majors.

    """
    courses = self.getClassesUsed().tolist()
    # below is a regex expression to only include non-numeric characters, inside a set comprehension
    majors = {re.findall('\A\D+', course)[0] for course in courses}
    print(majors)

  # TEST
  def filterColumnToValues(self, col, values = []):
    """Filters dataset to only include rows that contain any of the given values in the given column.

    Args:
        col (:obj:`str`): Name of the column to filter.
        values (:obj:`list`): Values to filter to.
        
    """
    if not self.__requiredColumnPresent(col):
        return
    
    self.printEntryCount()

    # M:  if all the elements in values array are strings:
    #       makes 'possibilities', lowercased elements in values array
    #       separated by '|'. Then, sets df equal to the specified 
    #       column(as str), with values conforming with possibilities(separated by '|')
    #       
    #       else, make the df equal to the specified column, checking 
    #       the values without changing case (idea: can make this not case sensitive maybe)

    if all([isinstance(x,str) for x in values]):
      lowered = [x.lower() for x in values]
      possibilities = "|".join(lowered)
      loweredCol = self.df[col].str.lower()
      self.df = self.df.loc[loweredCol.str.contains(possibilities)]
    else:
      self.df = self.df.loc[np.in1d(self.df[col],values)]
    
    # M: changes original df to have integer indices starting from 0
    self.df.reset_index(inplace=True, drop=True)
    self.printEntryCount()

  # Test
  def exportCSV(self, fileName = 'csvExport.csv'):
    """Export the current state of the dataset to a :obj:`.CSV` file.
    
    Args:
        fileName (:obj:`str`, optional): Name of the file to export. Defaults to 'csvExport.csv'.

    """
    self.df.to_csv(fileName, index=False)

  # TEST
  def filterToMultipleMajorsOrClasses(self, majors = [], classes = [], twoWay = True):
    """Reduces the dataset to only include entries of certain classes and/or classes in certain majors. This function is 
    inclusive; if a class in 'classes' is not of a major defined in 'majors', the class will still be included, and 
    vice-versa.

    Note:
        The 'classDept' column as set by :obj:`defineWorkingColumns` must have been defined in your dataset to filter by major.

    Args:
        majors (:obj:`list`, optional): List of majors to include. Filters by the 'classDept' column in the original dataset.
        classes (:obj:`list`, optional): List of classes to include. Filters by the 'classCode' column in the original dataset, or the conjoined version of 'classDept' and 'classNumber' columns.
        twoWay (:obj:`bool`, optional): Whether both classes in the correlation must be in the given majors / classes, or only one of them. Set to :obj:`True`, or both classes, by default.

    """
    if twoWay:
      self.df = self.df.loc[((self.df['course1'].isin(classes)) | (self.df['course1'].apply(lambda course: re.findall('\A\D+', course)[0])).isin(majors)) & ((self.df['course2'].isin(classes)) | (self.df['course2'].apply(lambda course: re.findall('\A\D+', course)[0]).isin(majors)))]
    else:
      self.df = self.df.loc[((self.df['course1'].isin(classes)) | (self.df['course1'].apply(lambda course: re.findall('\A\D+', course)[0])).isin(majors)) | ((self.df['course2'].isin(classes)) | (self.df['course2'].apply(lambda course: re.findall('\A\D+', course)[0]).isin(majors)))]
    self.df.reset_index(inplace=True, drop=True)

  # TEST
  def substituteSubStrInColumn(self, column, subString, substitute):
    """Replace a substring in a given column.

      Args:
        column (:obj:`str`): Column to replace substring in.
        subString (:obj:`str`): Substring to replace.
        substitute (:obj:`str`): Replacement of the substring.

    """
    self.convertColumnToString(column)
    self.df[column] = self.df[column].str.replace(subString, substitute)

  # TEST
  def chordGraphByMajor(self, coefficient = 0.5, pval = 0.05, outputName = 'majorGraph', outputSize = 200, imageSize = 300, showGraph = True, outputImage = True):
    """Creates a chord graph between available majors through averaging and filtering both correlation coefficients and P-values. Outputs to an html file, PNG file, and saves the underlying data by default.

    Note:
        The 'classDept' column as set by :obj:`defineWorkingColumns` must have been defined in your dataset to filter by major.

    Args:
        coefficient (:obj:`float`, optional): Minimum correlation coefficient to filter correlations by.
        pval (:obj:`float`, optional): Maximum P-value to filter correlations by. Defaults to 0.05 (a standard P-value limit used throughout the sciences)
        outputName (:obj:`str`, optional): First part of the outputted file names, e.g. fileName.csv, fileName.html, etc.
        outputSize (:obj:`int`, optional): Size (units unknown) of html graph to output. 200 by default.
        imageSize (:obj:`int`, optional): Size (units unknown) of image of the graph to output. 300 by default. Increase this if node labels are cut off.
        showGraph (:obj:`bool`, optional): Whether or not to open a browser and display the interactive graph that was created. Defaults to :obj:`True`.
        outputImage (:obj:`bool`, optional): Whether or not to export an image of the graph. Defaults to :obj:`True`. 
    
    """
    # M: The parameters should usually be changed when the function is called!!
    
    # M: initialized holoview of size outputSize
    hv.output(size=outputSize)
    
    # M:  creates a copy of df and sets course1 and course2 to the elements in the respective rows w 
    #     substring index 0 to the first number, exclusive (if number is first, element would be empty)
    majorFiltered = self.df.copy()
    # M: added the makeMissingValuesNanInColumn so that none of the entries are empty
    # majorFiltered.removeNanInColumn('course1')
    # majorFiltered.removeNanInColumn('course2')
    majorFiltered['course1'] = majorFiltered['course1'].apply(lambda course: re.findall('\A\D+', course)[0])
    majorFiltered['course2'] = majorFiltered['course2'].apply(lambda course: re.findall('\A\D+', course)[0])
    
    # sets majors to the unique remaining tuples of course1
    majors = majorFiltered['course1'].unique().tolist()
    majors.sort()


    majorCorrelations = []
    usedMajors = []

    # M: Makes the data in corr, P-value, and #students attributes numeric
    majorFiltered['corr'] = pd.to_numeric(majorFiltered['corr'])
    majorFiltered['P-value'] = pd.to_numeric(majorFiltered['P-value'])
    majorFiltered['#students'] = pd.to_numeric(majorFiltered['#students'])

    count = 0
    # M: loops through unique majors in course 1(those w/o numerical beginning)
    for major in majors:
      # Adds 1 to count then prints the number of elements in majors
      count += 1
      print(str(count) + ' / ' + str(len(majors)) + ' majors')
      
      # M: sets filteredToMajor to the majorFiltered where course 1 column is equal to 'major' in the majors list
      filteredToMajor = majorFiltered.loc[majorFiltered['course1'] == major]
      # M: sets connectedMajors to the unique values in course2 column
      connectedMajors = filteredToMajor['course2'].unique().tolist()

      # M: loops through the unique majors in course 2 (those w/o numerical beginning)
      for targetMajor in connectedMajors:
        # M: Sets filteredToMajorPair to the tuple(s) where course 1 is 'major' and course 2 is 'targetMajor'
        filteredToMajorPair = filteredToMajor.loc[filteredToMajor['course2'] == targetMajor]
        # M: Finds means for corr, PVal, and Students
        avgCorr = int(filteredToMajorPair['corr'].mean() * 100)
        avgPVal = filteredToMajorPair['P-value'].mean()
        avgStudents = filteredToMajorPair['#students'].mean()
        
        # M: ensures no corr following the constraints are counted twice and adds it to the list of correlations 
        if avgCorr > (coefficient * 100) and major != targetMajor and avgPVal < pval:
          if (targetMajor, major) not in usedMajors:
            usedMajors.append((major, targetMajor))
            majorCorrelations.append((major, targetMajor, avgCorr, avgPVal, avgStudents))

    # M: Tells us how many correlations found
    if len(majorCorrelations) == 0:
      print('Error: no valid correlations found.')
      return
    print(str(len(majorCorrelations)) + ' valid major correlations found.')
    
    # M: Sets output to majorCorrelations and sets the column names
    output = pd.DataFrame(majorCorrelations, columns=('source', 'target', 'corr', 'P-value', '#students'))
    # M: Sets newMajors to have the unique sources and targets (by putting them in a set) 
    newMajors = set(output['source'])
    newMajors.update(output['target'])
    # M: Sets sortedMajors to one list of sources and targets, all sorted
    sortedMajors = sorted(list(newMajors))

    # M: sets 'nodes' to be sortedMajors w/ column name 'name'
    nodes = pd.DataFrame(sortedMajors, columns = ['name'])
    
    # M: added this to check the value of output source and target before the apply
    print("source before:", output['source'])
    print("target before:", output['target'])

    # M: output source and target are changed to numeric instead of string objects to represent the sources and targets
    output['source'] = output['source'].apply(lambda major: nodes.index[nodes['name'] == major][0])
    output['target'] = output['target'].apply(lambda major: nodes.index[nodes['name'] == major][0])

    # M: Added this to check what each value would be set to after the apply
    print("index:", nodes.index[nodes['name'] == major][0])
    print("source now:", output['source'])
    print("target now:", output['target'])
    print(output['source'].dtype)

    # M: constructs the chord graph
    output.to_csv(outputName + '.csv', index=False)
    hvNodes = hv.Dataset(nodes, 'index')
    chord = hv.Chord((output, hvNodes)).select(value=(5, None))
    chord.opts(
        opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=dim('source').str(), 
                  labels='name', node_color=dim('index').str()))
    graph = hv.render(chord)
    output_file(outDir +outputName + '.html', mode='inline')
    # M: Saves and shows graph if showGraph true
    save(graph)
    if showGraph:
      show(graph)
    chord.opts(toolbar=None)
    # M: changes size to imageSize then saves it to outDir +outputName + '.png'
    if outputImage:
      hv.output(size=imageSize)
      export_png(hv.render(chord), filename=outDir +outputName + '.png')
    # M: changes size to outputSize
    hv.output(size=outputSize)


  def getNxGraph(self, minCorr = None):
    """Returns a NetworkX graph of the correlational data, where the nodes are classes and the weights are the correlations.

      Args:
        minCorr (:obj:`float`, optional): Minimum correlation between classes for an edge to be included on the graph. Should be in the 0.0-1.0 range. Defaults to :obj:`None` (or do not filter).

    """
# M:  if no filter, return graph containing edgelist correlations btwn course1 and course2
    if minCorr is None:
      print('minCorr none')
      return nx.from_pandas_edgelist(self.df, 'course1', 'course2', 'corr')
# M: filters out those with correlation < minCorr, then return graph containing edgelist correlations btwn course1 and course2
    self.df['corr'] = pd.to_numeric(self.df['corr'])
    filtered = self.df.loc[self.df['corr'] >= minCorr]
    return nx.from_pandas_edgelist(filtered, 'course1', 'course2', 'corr')

  def getCliques(self, minCorr = None, minSize = 2):
    """Returns a list of lists / cliques present in the correlational data. Cliques are connected sub-graphs in the larger overall graph.

    Args:
        minCorr (:obj:`None` or :obj:`float`, optional): Minimum correlation to consider a correlation an edge on the graph. 'None', or ignored, by default.
        minSize (:obj:`int`, optional): Minimum number of nodes to look for in a clique. Default is 2.

    """
# M:  First, gets graph with minCorr, then finds a list of cliques,
#     and finally, returns a sorted list of cliques of size >= minSize
    graph = self.getNxGraph(minCorr)
    cliques = list(nx.find_cliques(graph))
    return sorted(filter(lambda clique: len(clique) >= minSize, cliques))

  def outputCliqueDistribution(self, minCorr = None, countDuplicates = False, makeHistogram = False, fileName = 'cliqueHistogram', graphTitle = 'Class Correlation Cliques', logScale = False, exportPNG=True):
    """Outputs the clique distribution from the given correlation data. Prints to console by default, but can also optionally export a histogram.

    Args:
        minCorr (:obj:`None` or :obj:`float`, optional): Minimum correlation to consider a correlation an edge on the graph. 'None', or ignored, by default.
        countDuplicates (:obj:`bool`, optional): Whether or not to count smaller sub-cliques of larger cliques as cliques themselves. :obj:`False` by default.
        makeHistogram (:obj:`bool`, optional): Whether or not to generate a histogram. False by default.
        fileName (:obj:`str`, optional): File name to give exported histogram files. 'cliqueHistogram' by default.
        graphTitle (:obj:`str`, optional): Title displayed on the histogram. 'Class Correlation Cliques' by default.
        logScale (:obj:`bool`, optional): Whether or not to output graph in Log 10 scale on the y-axis. Defaults to :obj:`False`.

    """
# M:  Gets a list of cliques, then the largest clique length
    cliques = self.getCliques(minCorr = minCorr)
    largestClique = len(max(cliques, key = len))

# M:  makes a 'weight' array: contains counts of cliques of every size from 2 to 'largestClique'
#     (also includes count of smaller sub-cliques of larger cliques if 'countDuplicates' is true)
    weight = []
    for k in range(2, largestClique+1):
      count = 0
      for clique in cliques:
        if len(clique) == k:
            count += 1
        elif countDuplicates and len(clique) > k:
            count += len(list(itertools.combinations(clique, k)))
      weight.append(count)
      print('Size ' + str(k) + ' cliques: ' + str(count))


    if makeHistogram:
      # cliqueCount = [len(clique) for clique in cliques]
      # frequencies, edges = np.histogram(cliqueCount, largestClique - 1, (2, largestClique))
      cliqueCount = range(2, largestClique+1)
      frequencies, edges = np.histogram(a=cliqueCount, bins=largestClique - 1, range=(2, largestClique), weights=weight)
      #print('Values: %s, Edges: %s' % (frequencies.shape[0], edges.shape[0]))
      ylbl = 'Number of Cliques'
      # M: sets a log scale if specified
      if logScale:
        frequencies = [math.log10(freq) for freq in frequencies]
        ylbl += ' (log 10 scale)'
      
      # M: creates histogram using (edges, frequencies) and customizes it
      histo = hv.Histogram((edges, frequencies))
      histo.opts(opts.Histogram(xlabel='Number of Classes in Clique', ylabel=ylbl, title=graphTitle))
      hv.output(size=125)
      subtitle = 'n = ' + str(self.getEntryCount())
      if minCorr:
        subtitle = 'corr >= ' + str(minCorr) + ', ' + subtitle

      # M: creates the histogram, saves it, then shows it
      graph = hv.render(histo)
      graph.add_layout(Title(text=subtitle, text_font_style="italic", text_font_size="10pt"), 'above')
      output_file(outDir +fileName + '.html', mode='inline')
      save(graph)
      show(graph)
# JH: Use exportPng=True to get rid of the global variable.
#      if not edmApplication:
      # M: png version
      if exportPNG:
        histo.opts(toolbar=None)
        graph = hv.render(histo)
        graph.add_layout(Title(text=subtitle, text_font_style="italic", text_font_size="10pt"), 'above')
        export_png(graph, filename=outDir +fileName + '.png')

  # TEST
  def makeMissingValuesNanInColumn(self, column):
    """Replaces the ' ' values in 'column' w nan
    """
    if not self.__requiredColumnPresent(column):
      return
    self.df[column].replace(' ', np.nan, inplace=True)

  # TEST
  def removeNanInColumn(self, column):
    """drops na in 'column' then resets the indices(changes original df)
    """
    if not self.__requiredColumnPresent(column):
      return
    self.df.dropna(subset=[column], inplace=True)
    self.df.reset_index(inplace = True, drop=True)

  # TEST
  def dropMissingValuesInColumn(self, column):
    """Removes rows in the dataset which have missing data in the given column.

      Args:
        column (:obj:`str`): Column to check for missing values in.

    """
    if not self.__requiredColumnPresent(column):
      return
    self.makeMissingValuesNanInColumn(column)
    self.removeNanInColumn(column)      

  def convertColumnToString(self, column):
    """Converts 'column' to string type
    """
    if not self.__requiredColumnPresent(column):
      return
    self.df.astype({column:str}, copy=False)
  
  def __requiredColumnPresent(self, column):
    """Checks if 'column' present
        if not, prints error message depending on if 
        
        QUESTION: WHAT IS edmlib.edmApplication?
    """
    if column not in self.df.columns:
  # JH: Should this error be here or can we use two separate decorations?
      if edmlib.edmApplication:
        print("Error: required column '" + column + "' not present in dataset. Fix by right clicking / setting columns.")
      else:
        print("Error: required column '" + column + "' not present in dataset. Fix or rename with the 'defineWorkingColumns' function.")
      return False
    return True

