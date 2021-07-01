"""
Library created by Michael Riad Zaky mriadzaky@fordham.edu (2020), with Daniel Leeds, Gary Weiss, Mavis Zhang at Fordham 
University. Library free for use provided you cite https://github.com/MichaelRZ/EDMLib in any resulting publications.  
Library free for redistribution provided you retain the author attributions above.

The following packages are required for installation before use: numpy, pandas, csv, scipy, holoviews
"""
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
#from edmlib import gradeData, classCorrelationData
numLibInstalled = True
try:
  import numexpr
except:
  numLibInstalled = False
  pass

pd.options.mode.chained_assignment = None 
hv.extension('bokeh')
def disable_logo(plot, element):
    plot.state.toolbar.logo = None
hv.plotting.bokeh.ElementPlot.hooks.append(disable_logo)

pvalSuffix = 'PValue'
ttestSuffix = '_ttest'
edmApplication = False

def makeExportDirectory(directory):
  global outDir
  if directory[-1] == '/':
    outDir = directory
  else:
    outDir = directory + '/'
  if not os.path.isdir(outDir[:-1]):
    os.mkdir(outDir[:-1])

outDir = 'exports/'  
makeExportDirectory(outDir)

def instructorAveraging(data, filename = 'instructorAverages', minPercentage = 5.0, maxPercentage = 90.0, weighting = '#students', extraNorm = None):
  """
    Reads given file with instructor averages, create a new weighted data frame and return it.
    Returns None if any error is found.
  """
  cols = ['Instructor','courseTaught','futureCourse','normBenefit','gradeBenefit','#students', '#nonStudents', 'totalStudents', '%ofStudents']
  if not all(x in data.columns for x in cols):
    print('Error: Columns missing. Did you use the instructorRanksAllClasses method?')
    return None
  if weighting not in data.columns and weighting is not None:
    print('Error: Weight not present in columns. Check spelling.')
    return None
  filteredData = data.loc[(data['courseTaught'].str.replace('\d+', '') == data['futureCourse'].str.replace('\d+', '')) & 
    (data['%ofStudents'].apply(float) <= maxPercentage) & (data['%ofStudents'].apply(float) >= minPercentage)]
  filteredData['courseTaught'] = data['courseTaught'].str.replace('\d+', '')
  filteredData['futureCourse'] = data['futureCourse'].str.replace('\d+', '')
  uniqueInstructors = filteredData['Instructor'].unique()
  grouped = filteredData.groupby('Instructor')
  rowlist = []
  for instructor in uniqueInstructors:
    entries = grouped.get_group(instructor)
    for subject in entries['courseTaught'].unique():
      subEntries = entries.loc[entries['courseTaught'] == subject]
      rowdict = {'Instructor' : instructor}
      rowdict['Subject'] = subject
      if weighting is not None:
        rowdict['avNormBenefit'] = np.average(subEntries['normBenefit'].apply(float), weights=subEntries[weighting].apply(float))
        rowdict['avGradeBenefit'] = np.average(subEntries['gradeBenefit'].apply(float), weights=subEntries[weighting].apply(float))
        if extraNorm is not None:
          rowdict['av' + extraNorm] = np.average(subEntries[extraNorm].apply(float), weights=subEntries[weighting].apply(float))
      else:
        rowdict['avNormBenefit'] = np.average(subEntries['normBenefit'].apply(float))
        rowdict['avGradeBenefit'] = np.average(subEntries['gradeBenefit'].apply(float))
        if extraNorm is not None:
          rowdict['av' + extraNorm] = np.average(subEntries[extraNorm].apply(float))
      rowdict['students/entries'] = sum(subEntries['#students'].apply(int))
      rowdict['avStudents'] = np.average(subEntries['#students'].apply(int))
      rowdict['av%ofStudents'] = np.average(subEntries['%ofStudents'].apply(float))
      rowlist.append(rowdict)
  if extraNorm is not None:
    completeDf = pd.DataFrame(rowlist, columns=['Instructor','Subject','avNormBenefit','avGradeBenefit','av' + extraNorm,'students/entries','avStudents', 'av%ofStudents'])
  else:
    completeDf = pd.DataFrame(rowlist, columns=['Instructor','Subject','avNormBenefit','avGradeBenefit','students/entries','avStudents', 'av%ofStudents'])
  completeDf.sort_values(by=['Subject','Instructor'])
  completeDf.reset_index(inplace = True, drop=True)
  completeDf['grade*Norm*Sign(norm)'] = completeDf['avGradeBenefit'] * completeDf['avNormBenefit'] * np.sign(completeDf['avNormBenefit'])
  completeDf['avNormBenefit' + pvalSuffix] = pvalOfSeries(completeDf['avNormBenefit'])
  completeDf['avGradeBenefit' + pvalSuffix] = pvalOfSeries(completeDf['avGradeBenefit'])
  completeDf['grade*Norm*Sign(norm)' + pvalSuffix] = pvalOfSeries(completeDf['grade*Norm*Sign(norm)'])
  if extraNorm is not None:
    completeDf['av' + extraNorm + pvalSuffix] = pvalOfSeries(completeDf['av' + extraNorm])

  if not filename.endswith('.csv'):
    filename = "".join((filename, '.csv'))
  completeDf.to_csv(filename, index=False)
  return completeDf

def instructorAveraging2(data, filename = 'instructorAverages', minPercentage = 5.0, maxPercentage = 90.0, weighting = '#students', extraNorm = None):
  """
    Reads given file with instructor averages, create a new weighted data frame.
  """
  cols = ['Instructor','courseTaught','futureCourse','normBenefit','gradeBenefit','#students', '#nonStudents', 'totalStudents', '%ofStudents']
  if not all(x in data.columns for x in cols):
    print('Error: Columns missing. Did you use the instructorRanksAllClasses method?')
    return None
  if weighting not in data.columns and weighting is not None:
    print('Error: Weight not present in columns. Check spelling.')
    return None
  filteredData = data.loc[(data['%ofStudents'].apply(float) <= maxPercentage) & (data['%ofStudents'].apply(float) >= minPercentage)]
  filteredData['courseTaught'] = data['courseTaught'].str.replace('\d+', '')
  filteredData['futureCourse'] = data['futureCourse'].str.replace('\d+', '')
  uniqueInstructors = filteredData['Instructor'].unique()
  grouped = filteredData.groupby('Instructor')
  rowlist = []
  for instructor in uniqueInstructors:
    entries = grouped.get_group(instructor)
    for subject in entries['courseTaught'].unique():
      subEntries = entries.loc[entries['courseTaught'] == subject]
      for subject2 in subEntries['futureCourse'].unique():
        sub2Entries = subEntries.loc[subEntries['futureCourse'] == subject2]
        rowdict = {'Instructor' : instructor}
        rowdict['firstSubject'] = subject
        rowdict['secondSubject'] = subject2
        if weighting is not None:
          rowdict['avNormBenefit'] = np.average(sub2Entries['normBenefit'].apply(float), weights=sub2Entries[weighting].apply(float))
          rowdict['avGradeBenefit'] = np.average(sub2Entries['gradeBenefit'].apply(float), weights=sub2Entries[weighting].apply(float))
          if extraNorm is not None:
            rowdict['av' + extraNorm] = np.average(sub2Entries[extraNorm].apply(float), weights=sub2Entries[weighting].apply(float))
        else:
          rowdict['avNormBenefit'] = np.average(sub2Entries['normBenefit'].apply(float))
          rowdict['avGradeBenefit'] = np.average(sub2Entries['gradeBenefit'].apply(float))
          if extraNorm is not None:
            rowdict['av' + extraNorm] = np.average(sub2Entries[extraNorm].apply(float))
        rowdict['students/entries'] = sum(sub2Entries['#students'].apply(int))
        rowdict['avStudents'] = np.average(sub2Entries['#students'].apply(int))
        rowdict['av%ofStudents'] = np.average(sub2Entries['%ofStudents'].apply(float))
      rowlist.append(rowdict)
  if extraNorm is not None:
    completeDf = pd.DataFrame(rowlist, columns=['Instructor','firstSubject','secondSubject','avNormBenefit','avGradeBenefit','av' + extraNorm,'students/entries','avStudents', 'av%ofStudents'])
  else:
    completeDf = pd.DataFrame(rowlist, columns=['Instructor','firstSubject','secondSubject','avNormBenefit','avGradeBenefit','students/entries','avStudents', 'av%ofStudents'])
  completeDf.sort_values(by=['secondSubject','firstSubject','Instructor'])
  completeDf.reset_index(inplace = True, drop=True)
  completeDf['grade*Norm*Sign(norm)'] = completeDf['avGradeBenefit'] * completeDf['avNormBenefit'] * np.sign(completeDf['avNormBenefit'])
  completeDf['avNormBenefit' + pvalSuffix] = pvalOfSeries(completeDf['avNormBenefit'])
  completeDf['avGradeBenefit' + pvalSuffix] = pvalOfSeries(completeDf['avGradeBenefit'])
  completeDf['grade*Norm*Sign(norm)' + pvalSuffix] = pvalOfSeries(completeDf['grade*Norm*Sign(norm)'])
  if extraNorm is not None:
    completeDf['av' + extraNorm + pvalSuffix] = pvalOfSeries(completeDf['av' + extraNorm])

  if not filename.endswith('.csv'):
    filename = "".join((filename, '.csv'))
  completeDf.to_csv(filename, index=False)
  return completeDf

def analyzePairs(data):
  """
    Not currently used, was it requested?
  """
  data[data.columns[1]] = data[data.columns[1]].str.replace('\d+', '')
  data[data.columns[2]] = data[data.columns[2]].str.replace('\d+', '')
  same = sum(data[data.columns[1]] == data[data.columns[2]])
  # different = len(data.index) - same
  res = same / len(data.index)
  print(str(same) + ' / ' + str(len(data.index)) + ' or ' + str(round(res*100,3)) + '%')
  return res

def pvalOfSeries(data):
  # data.astype(float, copy=False)
  data = data.apply(float)
  normMean, normStd = data.mean(), data.std()
  # zval = lambda x: (x - normMean) / normStd
  # pval = lambda x: (1 - sciNorm.cdf(abs(x))) * 2
  # return data.apply(zval).apply(pval)
  # print('converted, did mean and std')
  # zval = lambda x: (x - normMean) / normStd
  pval = lambda x: (1 - sciNorm.cdf(abs((x - normMean) / normStd))) * 2
  return pval(data.values)

def tTestOfTwoSeries(data1,data2):
  data1,data2 = data1.apply(float),data2.apply(float)
  data1m, data1std = data1.mean(), data1.std()
  data2m, data2std = data2.mean(), data2.std()
  data1n, data2n = data1.size, data2.size
  tTest = (data1m - data2m) / math.sqrt(((data1std**2)/data1n) + ((data2std**2)/data2n))
  return tTest

def seriesToHistogram(data, fileName = 'histogram', graphTitle='Distribution', sortedAscending = True, logScale = False,xlbl='Value',ylbl = 'Frequency'):
  data2 = data.replace(' ', np.nan)
  data2.dropna(inplace=True)
  # data2.sort_values(inplace=True)
  try:
    histData = pd.to_numeric(data2, errors='raise')
    numericData = True
  except:
    histData = data2
    numericData = False
  if numericData:
    # frequencies, edges = np.histogram(gpas, int((highest - lowest) / 0.1), (lowest, highest))
    
    dataList = histData.tolist()
    frequencies, edges = np.histogram(dataList, (int(math.sqrt(len(dataList))) if (len(dataList) > 30) else (max(len(dataList) // 3 , 1))), (min(dataList), max(dataList)))
    #print('Values: %s, Edges: %s' % (frequencies.shape[0], edges.shape[0]))
    
    if logScale:
      frequencies = [math.log10(freq) if freq > 0 else freq for freq in frequencies]
      ylbl += ' (log 10 scale)'
    histo = hv.Histogram((edges, frequencies))
    histo.opts(opts.Histogram(xlabel=xlbl, ylabel=ylbl, title=graphTitle, fontsize={'title': 40, 'labels': 20, 'xticks': 20, 'yticks': 20}))
    subtitle= 'mean: ' + str(round(sum(dataList) / len(dataList), 3))+ ', n = ' + str(len(dataList))
    hv.output(size=250)
    graph = hv.render(histo)
    graph.add_layout(Title(text=subtitle, text_font_style="italic", text_font_size="30pt"), 'above')
    output_file(outDir +fileName + '.html', mode='inline')
    save(graph)
    show(graph)
# JH: Adds some specific display components when not in a graphical program.
# JH: Consider a separate function for the two cases.
    if not edmApplication:
      hv.output(size=300)
      histo.opts(toolbar=None)
      graph = hv.render(histo)
      graph.add_layout(Title(text=subtitle, text_font_style="italic", text_font_size="30pt"), 'above')
      export_png(graph, filename=outDir +fileName + '.png')
  else:
    barData = histData.value_counts(dropna=False)
    dictList = sorted(zip(barData.index, barData.values), key = lambda x: x[sortedAscending])
    # print(dictList)
    bar = hv.Bars(dictList)
    bar.opts(opts.Bars(xlabel=xlbl, ylabel=ylbl, title=graphTitle))
    subtitle= 'n = ' + str(len(dictList))
    hv.output(size=250)
    graph = hv.render(bar)
    graph.add_layout(Title(text=subtitle, text_font_style="italic", text_font_size="30pt"), 'above')
    output_file(outDir +fileName + '.html', mode='inline')
    save(graph)
    show(graph)
# JH: Consider a bool exportPng=True when calling from outside edmAppliation
    if not edmApplication:
      hv.output(size=300)
      bar.opts(toolbar=None)
      graph2 = hv.render(bar)
      graph2.add_layout(Title(text=subtitle, text_font_style="italic", text_font_size="30pt"), 'above')
      export_png(graph2, filename=outDir +fileName + '.png')
  hv.output(size=125)
    
