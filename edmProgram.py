import csv, sys, os, io, getpass, webbrowser, traceback
#import inspect, time
import PyQt5
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
# from PyQt5.QtWebEngineWidgets import *
import pandas as pd
import numpy as np
from param import Filename
import edmlib
from edmlib import *
class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        # self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.message = QtWidgets.QLabel()
        self.statusBar().addWidget(self.message)
        self.settings = QSettings('EDM Lab', 'EDM Program')
        self.setWindowTitle("EDM Program")
        # self.settings.setValue("lastFile", None)
        if self.settings.value(getpass.getuser()+"lastFile", self.settings.value("lastFile", None)):
          if os.path.isfile(self.settings.value(getpass.getuser()+"lastFile", self.settings.value("lastFile", None))):
            self.setWindowTitle("EDM - " + self.settings.value("lastFile",None))
        if self.settings.value(getpass.getuser()+"ExportDir", None):
          edmlib.makeExportDirectory(self.settings.value(getpass.getuser()+"ExportDir", None))
        self.terminal = sys.stdout
        sys.stdout = self
        self.resize(self.settings.value("size", QSize(640, 480)))
        self.move(self.settings.value("pos", QPoint(50, 50)))
        self.correlationFile = False
        self.grades = False
        self.first = False
        self.reload()
        self.first = True
        self.setCentralWidget(self.tableView)
        self.menubar = self.menuBar()
        self.setGeneralButtons()
        
        # menubar = QMenuBar()
        
        # exitAct.setShortcut('Ctrl+Q')
        # exitAct.setStatusTip('Exit application')
        # exitAct.triggered.connect(self.close)


        # self.pushButtonLoad = QtWidgets.QPushButton(self)
        # self.pushButtonLoad.setText("Load Csv File")
        # self.pushButtonLoad.clicked.connect(self.on_pushButtonLoad_clicked)

        # self.pushButtonWrite = QtWidgets.QPushButton(self)
        # self.pushButtonWrite.setText("Save Csv File")
        # self.pushButtonWrite.clicked.connect(self.on_pushButtonWrite_clicked)

        # self.layoutVertical = QtWidgets.QVBoxLayout(self)
        # self.layoutVertical.addWidget(self.tableView)
        # self.layoutVertical.addWidget(self.pushButtonLoad)
        # self.layoutVertical.addWidget(self.pushButtonWrite)

        if self.settings.value(getpass.getuser()+"lastFile", self.settings.value("lastFile", None)):
          if os.path.isfile(self.settings.value(getpass.getuser()+"lastFile", self.settings.value("lastFile", None))):
            self.loadCsv(self.settings.value(getpass.getuser()+"lastFile", self.settings.value("lastFile", None)))
            print('File opened.')
            self.doColumnThings()
            self.setGeneralButtons()
            # print('File loaded')

        self.threadpool = QThreadPool()
        self.show()


    def eventFilter(self, source, event):
        if (event.type() == QtCore.QEvent.KeyPress and
            event.matches(QtGui.QKeySequence.Copy)):
            self.copySelection()
            return True
        return super(MyWindow, self).eventFilter(source, event)

    def copySelection(self):
        selection = self.tableView.selectedIndexes()
        if selection:
            rows = sorted(index.row() for index in selection)
            columns = sorted(index.column() for index in selection)
            rowcount = rows[-1] - rows[0] + 1
            colcount = columns[-1] - columns[0] + 1
            table = [[''] * colcount for _ in range(rowcount)]
            for index in selection:
                row = index.row() - rows[0]
                column = index.column() - columns[0]
                table[row][column] = index.data()
            stream = io.StringIO()
            csv.writer(stream, delimiter='\t').writerows(table)
            QtWidgets.qApp.clipboard().setText(stream.getvalue())  

    def reload(self, skipPVal = False):
        # ts = time.time()
        self.model = QtGui.QStandardItemModel(self)
        # for key in self.settings.allKeys():
        #   print(str(key) + ' ' + str(self.settings.value(key, None)))
        def resizeEvent(self, event):
          super(QtWidgets.QTableView, self).resizeEvent(event)
          header = self.horizontalHeader()
          for column in range(header.count()):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
            width = header.sectionSize(column)
            header.setSectionResizeMode(column, QHeaderView.Interactive)
            header.resizeSection(column, width)
        QtWidgets.QTableView.resizeEvent = resizeEvent
        self.tableView = QtWidgets.QTableView(self)
        self.tableView.installEventFilter(self)
        self.tableView.setModel(self.model)
        self.tableView.setSortingEnabled(True)
        self.tableView.horizontalHeader().setStretchLastSection(True)
        self.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tableView.horizontalHeader().setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tableView.horizontalHeader().customContextMenuRequested.connect(self.onColumnRightClick)
        if isinstance(self, MyWindow):
          self.setCentralWidget(self.tableView)
        elif isinstance(self, csvPreview):
          self.layout.itemAt(0).widget().setParent(None)
          self.layout.insertWidget(0,self.tableView)
          # self.layout.addWidget(self.tableView)
          # self.layout.addWidget(self.buttonBox)
        if self.first:
            try:
                if not skipPVal:
                  if len(self.grades.df.columns) > 1 and any(x.endswith(edmlib.pvalSuffix) or x.endswith(edmlib.ttestSuffix) for x in self.grades.df.columns):
                    for col in self.grades.df.columns:
                      # print(col + edmlib.pvalSuffix)
                      if col + edmlib.pvalSuffix in self.grades.df.columns:
                        self.grades.df[col + edmlib.pvalSuffix] = edmlib.pvalOfSeries(self.grades.df[col])
                      if col.endswith(edmlib.ttestSuffix):
                        c1,c2=col[:-len(edmlib.ttestSuffix)].split('|')
                        self.grades.df[col]=edmlib.tTestOfTwoSeries(c1,c2)
                self.df = self.grades.df
                self.model = TableModel(self.grades.df)
                self.tableView.setModel(self.model)
                self.getLastKnownColumns()
            except Exception as e: 
                # print(e) 
                pass
        # te = time.time()
        # print('reload time (s)')
        # print(te - ts)

    # GUI upper menu
    def setGeneralButtons(self):
        """
          Set up the GUI upper menu bars
        """
        self.menubar.clear()
        self.menubar = self.menuBar()
        # File menu
        self.fileMenu = self.menubar.addMenu('File')
        # Each block is one button of File menu
        openFile = QAction('Open', self)
        openFile.triggered.connect(self.on_pushButtonLoad_clicked)
        self.fileMenu.addAction(openFile)

        graphOpen = QAction('Open Graph', self)
        graphOpen.triggered.connect(self.openGraph)
        self.fileMenu.addAction(graphOpen)

        setColumns = QAction('Set Columns', self)
        setColumns.triggered.connect(self.designateColumns)
        self.fileMenu.addAction(setColumns)

        saveAs = QAction('Save As', self)
        saveAs.triggered.connect(self.on_pushButtonWrite_clicked)
        self.fileMenu.addAction(saveAs)

        self.menubar.setNativeMenuBar(False)
        self.setMenuBar(self.menubar)

        # Menu for non correlation matrix file (e.g. original dataset)
        if self.settings.value(getpass.getuser()+"lastFile", self.settings.value("lastFile", None)) and not self.correlationFile:
          if os.path.isfile(self.settings.value(getpass.getuser()+"lastFile", self.settings.value("lastFile", None))):
            # Export stats menu
            self.exportMenu = self.menubar.addMenu('Export')
            # Each block is one button of the Stats menu
            exportCorr = QAction('Class Correlations', self)
            exportCorr.triggered.connect(self.exportCorrelations)
            self.exportMenu.addAction(exportCorr)

            exportGPAS = QAction('GPA Distribution', self)
            exportGPAS.triggered.connect(self.exportGPADistribution)
            self.exportMenu.addAction(exportGPAS)

            exportPairGrades = QAction('Class Pair Grades', self)
            exportPairGrades.triggered.connect(self.pairGraph)
            self.exportMenu.addAction(exportPairGrades)

            sankeyTrackNew = QAction('Course Track Graph', self)
            sankeyTrackNew.triggered.connect(self.makeSankeyTrackNew)
            self.exportMenu.addAction(sankeyTrackNew)

            sankeyTrack = QAction('Course Track Graph (alternate)', self)
            sankeyTrack.triggered.connect(self.makeSankeyTrack)
            self.exportMenu.addAction(sankeyTrack)

            sankeyTrack2 = QAction('Track Graph (Experimental)', self)
            sankeyTrack2.triggered.connect(self.makeSankeyTrackAdvanced)
            self.exportMenu.addAction(sankeyTrack2)

            currentState = QAction('Current State of the DataFrame', self)
            currentState.triggered.connect(self.exportCSV)
            self.exportMenu.addAction(currentState)

            # Filter menu
            self.filterMenu = self.menubar.addMenu('Filters')
            # Each block is one button of the Filter menu
            byClassOrMajor = QAction('by Classes / Class Depts', self)
            byClassOrMajor.triggered.connect(self.filterByClassOrMajor)
            self.filterMenu.addAction(byClassOrMajor)

            byStdntMajor = QAction('by Student Majors', self)
            byStdntMajor.triggered.connect(self.filterByStudentMajor)
            self.filterMenu.addAction(byStdntMajor)

            byGPADev = QAction('Classes by Grade Deviation', self)
            byGPADev.triggered.connect(self.filterGPADeviation)
            self.filterMenu.addAction(byGPADev)

            # Calculation menu
            self.calcMenu = self.menubar.addMenu('Calculations')
            # Each block is one button of the Calculation menu
            gpaMean = QAction('Class Grade Means', self)
            gpaMean.triggered.connect(self.gpaMeanCol)
            self.calcMenu.addAction(gpaMean)

            gpaDev = QAction('Class Grade Std Deviations', self)
            gpaDev.triggered.connect(self.gpaDevCol)
            self.calcMenu.addAction(gpaDev)

            norms = QAction('Grade Normalizations', self)
            norms.triggered.connect(self.normCol)
            self.calcMenu.addAction(norms)

            norms2 = QAction('Normalization by Student GPA', self)
            norms2.triggered.connect(self.normByGPACol)
            self.calcMenu.addAction(norms2)

            norms3 = QAction('Normalization by Student by Class', self)
            norms3.triggered.connect(self.normByStudByClassCol)
            self.calcMenu.addAction(norms3)

            instructorEffect = QAction('Instructor Effectiveness', self)
            instructorEffect.triggered.connect(self.instructorEffectiveness)
            self.calcMenu.addAction(instructorEffect)

            instructorEffectAll = QAction('Instructor Effectiveness (All)', self)
            instructorEffectAll.triggered.connect(self.instructorEffectivenessAll)
            self.calcMenu.addAction(instructorEffectAll)

            ttest = QAction('t-test', self)
            ttest.triggered.connect(self.ttestCalc)
            self.calcMenu.addAction(ttest)

            predict = QAction('Grade Predictions', self)
            predict.triggered.connect(self.gradePredict)
            self.calcMenu.addAction(predict)

            stats = QAction('Unique Values', self)
            stats.triggered.connect(self.getStats)
            self.calcMenu.addAction(stats)
        # Menu for correlation matrix file
        elif self.settings.value(getpass.getuser()+"lastFile", self.settings.value("lastFile", None)):
          if os.path.isfile(self.settings.value(getpass.getuser()+"lastFile", self.settings.value("lastFile", None))):
            self.correlationMenu = self.menubar.addMenu('Correlations')

            majorChord = QAction('Export Chord Graph by Major', self)
            majorChord.triggered.connect(self.exportMajorChord)
            self.correlationMenu.addAction(majorChord)

            cliqueHisto = QAction('Export Clique Histogram', self)
            cliqueHisto.triggered.connect(self.exportCliqueHisto)
            self.correlationMenu.addAction(cliqueHisto)

            self.filterMenu = self.menubar.addMenu('Filters')

            byClassOrMajor = QAction('Filter to Classes / Class Depts', self)
            byClassOrMajor.triggered.connect(self.filterByClassOrMajorCorr)
            self.filterMenu.addAction(byClassOrMajor)

            self.calcMenu = self.menubar.addMenu('Calculations')

            ttest = QAction('Calculate t-test', self)
            ttest.triggered.connect(self.ttestCalc)
            self.calcMenu.addAction(ttest)
        if self.settings.value(getpass.getuser()+"lastFile", self.settings.value("lastFile", None)):
          if os.path.isfile(self.settings.value(getpass.getuser()+"lastFile", self.settings.value("lastFile", None))):
            getOriginal = QAction('Reload Original File', self)
            getOriginal.triggered.connect(self.originalReload)
            self.fileMenu.addAction(getOriginal)

            setExports = QAction('Set User Export Directory', self)
            setExports.triggered.connect(self.setExportDir)
            self.fileMenu.addAction(setExports)

    def doColumnThings(self):
        self.columnSelected = None
        self.rcMenu=QMenu(self)
        if not self.correlationFile:
            self.setMenu = self.rcMenu.addMenu('Set Column')
            setDept = QAction('Set Class Department Column', self)
            setDept.triggered.connect(self.designateDept)
            self.setMenu.addAction(setDept)
            setClassNumber = QAction('Set Class Number Column', self)
            setClassNumber.triggered.connect(self.designateClssNmbr)
            self.setMenu.addAction(setClassNumber)
            setCID = QAction('Set Class ID Column', self)
            setCID.triggered.connect(self.designateCID)
            self.setMenu.addAction(setCID)
            setSID = QAction('Set Student ID Column', self)
            setSID.triggered.connect(self.designateSID)
            self.setMenu.addAction(setSID)
            setGrades = QAction('Set Numeric Grade Column', self)
            setGrades.triggered.connect(self.designateGrades)
            self.setMenu.addAction(setGrades)
            setTerm = QAction('Set Term Column', self)
            setTerm.triggered.connect(self.designateTerms)
            self.setMenu.addAction(setTerm)
            setStdntMjr = QAction('Set Student Major Column', self)
            setStdntMjr.triggered.connect(self.designateStdntMjr)
            self.setMenu.addAction(setStdntMjr)
            setStdntYr = QAction('Set Student Year Column (optional)', self)
            setStdntYr.triggered.connect(self.designateStdntYr)
            self.setMenu.addAction(setStdntYr)
            setCredits = QAction('Set Class Credits Column (optional)', self)
            setCredits.triggered.connect(self.designateCredits)
            self.setMenu.addAction(setCredits)
            setFID = QAction('Set Faculty ID Column (optional)', self)
            setFID.triggered.connect(self.designateFID)
            self.setMenu.addAction(setFID)
            setClassCode = QAction('Set Class Code Column (optional)', self)
            setClassCode.triggered.connect(self.designateClassCode)
            self.setMenu.addAction(setClassCode)
            self.getLastKnownColumns()
            renColumn = QAction('Rename Column...', self)
            renColumn.triggered.connect(self.renameColumn)
            self.rcMenu.addAction(renColumn)
        self.substituteMenu = self.rcMenu.addMenu('Substitute')
        substituteInColumn = QAction('Substitute in Column...', self)
        substituteInColumn.triggered.connect(self.strReplace)
        self.substituteMenu.addAction(substituteInColumn)
        dictReplaceInColumn = QAction('Substitute Many Values...', self)
        dictReplaceInColumn.triggered.connect(self.dictReplace)
        self.substituteMenu.addAction(dictReplaceInColumn)
        dictReplaceTxt = QAction('Use substitution file...', self)
        dictReplaceTxt.triggered.connect(self.dictReplaceFile)
        self.substituteMenu.addAction(dictReplaceTxt)
        self.fNumMenu = self.rcMenu.addMenu('Filter / Numeric Operations')
        valFilter = QAction('Filter Column to Value(s)...', self)
        valFilter.triggered.connect(self.filterColByVals)
        self.fNumMenu.addAction(valFilter)
        NumericFilter = QAction('Filter Column Numerically...', self)
        NumericFilter.triggered.connect(self.filterColNumeric)
        self.fNumMenu.addAction(NumericFilter)
        absFilter = QAction('Make Absolute Values', self)
        absFilter.triggered.connect(self.absCol)
        self.fNumMenu.addAction(absFilter)
        avStats = QAction('Get Mean / Med. / Mode', self)
        avStats.triggered.connect(self.avCol)
        self.fNumMenu.addAction(avStats)
        pval = QAction('Calculate P-Value', self)
        pval.triggered.connect(self.pValCol)
        self.fNumMenu.addAction(pval)
        roundFilter = QAction('Round Column...', self)
        roundFilter.triggered.connect(self.roundCol)
        self.fNumMenu.addAction(roundFilter)
        NAFilter = QAction('Drop Undefined Values in Column', self)
        NAFilter.triggered.connect(self.removeNaInColumn)
        self.rcMenu.addAction(NAFilter)
        # if not self.correlationFile:
        deleteColumn = QAction('Delete Column (permanent)', self)
        deleteColumn.triggered.connect(self.delColumn)
        self.rcMenu.addAction(deleteColumn)
        histExp = QAction('Export Histogram', self)
        histExp.triggered.connect(self.exportHisto)
        self.rcMenu.addAction(histExp)

    def onColumnRightClick(self, QPos=None):       
        parent=self.sender()
        pPos=parent.mapToGlobal(QtCore.QPoint(0, 0))
        mPos=pPos+QPos
        column = self.tableView.horizontalHeader().logicalIndexAt(QPos)
        label = self.model.headerData(column, Qt.Horizontal, Qt.DisplayRole)
        self.columnSelected = label
        self.rcMenu.move(mPos)
        self.rcMenu.show()

    def getLastKnownColumns(self):
      if self.settings.value("ClassID", None):
        self.designateCID(self.settings.value("ClassID", None))
      if self.settings.value("ClassCode", None):
        self.designateClassCode(self.settings.value("ClassCode", None))
      if self.settings.value("ClassNumber", None):
        self.designateClssNmbr(self.settings.value("ClassNumber", None))
      if self.settings.value("ClassDept", None):
        self.designateDept(self.settings.value("ClassDept", None))
      if self.settings.value("Grades", None):
        self.designateGrades(self.settings.value("Grades", None))
      if self.settings.value("StudentID", None):
        self.designateSID(self.settings.value("StudentID", None))
      if self.settings.value("StudentMajor", None):
        self.designateStdntMjr(self.settings.value("StudentMajor", None))
      if self.settings.value("StudentYear", None):
        self.designateStdntYr(self.settings.value("StudentYear", None))
      if self.settings.value("Terms", None):
        self.designateTerms(self.settings.value("Terms", None))
      if self.settings.value("Credits", None):
        self.designateCredits(self.settings.value("Credits", None))
      if self.settings.value("FID", None):
        self.designateFID(self.settings.value("FID", None))

    def designateColumns(self):
      dlg = columnDialog(self, list(self.grades.df.columns.values))
      dlg.setWindowTitle("Set Columns")
      if dlg.exec():
        try:
          result = dlg.getInputs()
          self.designateDept(result[0])
          self.designateClssNmbr(result[1])
          self.designateCID(result[2])
          self.designateSID(result[3])
          self.designateGrades(result[4])
          self.designateTerms(result[5])
          self.designateStdntMjr(result[6])
          self.designateStdntYr(result[7])
          self.designateCredits(result[8])
          self.designateFID(result[9])
          self.designateClassCode(result[10])

        except Exception as e: 
          print(e)
          pass

    def gradePredict(self):
        if ((self.grades.CLASS_CODE_COLUMN in self.grades.df.columns) or ((self.grades.CLASS_DEPT_COLUMN in self.grades.df.columns) and (self.grades.CLASS_NUMBER_COLUMN in self.grades.df.columns) )) and (self.grades.FINAL_GRADE_COLUMN in self.grades.df.columns):
          dlg = gradePredictDialogue(self)
          dlg.setWindowTitle("Predict Grades")
          if dlg.exec():
            prior, future, mode = dlg.getInputs()
            deTranslate = {v: k for k, v in dlg.translate.items()}
            if mode == 'tryAll':
              print('Trying all grade modes...')
              modeNames = dlg.modes[:-1]
              results = []
              for method in [dlg.translate[x] for x in modeNames]:
                results.append(self.grades.gradePredict(prior, future, method))
              resultValues = [list(x.values()) for x in results]
              resultValues.insert(0, list(results[0].keys()))
              dlg2 = csvPreview(self, pd.DataFrame(zip(*resultValues), columns=['Class'] + modeNames))
              dlg2.resize(QSize(640, 400))
              # dlg2.tableView.sortByColumn(0)
              dlg2.setWindowTitle('Grade Predictions (All)')
              dlg2.exec()
            else:
              vals = self.grades.gradePredict(prior, future, mode)
              dlg2 = csvPreview(self, pd.DataFrame(vals.items(), columns=['Class', 'Predicted Grade']))
              dlg2.resize(QSize(400, 480))
              # dlg2.tableView.sortByColumn(0)
              dlg2.setWindowTitle('Grade Predictions (' + deTranslate[mode] + ')')
              dlg2.exec()

        else:
          print('Error: Class dept/num or code and grade columns required. Fix with "set columns" in the top menu.')

    def getStats(self):
        dlg = statsDialogue(self)
        dlg.setWindowTitle("Dataframe Statistics - " + str(len(self.grades.df.index)) + " rows")
        dlg.exec()
            # try:
            #   corr, pval, name = dlg.getInputs()
            #   worker = Worker(self.grades.chordGraphByMajor, corr, pval, name)
            #   self.threadpool.start(worker) 
            # except Exception as e: 
            #   print(e)
            #   pass

    def renameColumn(self):
        oldName = self.columnSelected
        dlg = renameColumnDialogue(self)
        dlg.setWindowTitle("Rename Column")
        if dlg.exec():
            try:
                newName = dlg.getInputs()
                self.grades.df.rename({oldName: newName}, axis=1, inplace=True)
                if self.settings.value("ClassID", None) == oldName:
                  self.designateCID(newName)
                if self.settings.value("ClassCode", None) == oldName:
                  self.designateClassCode(newName)
                if self.settings.value("ClassNumber", None) == oldName:
                  self.designateClssNmbr(newName)
                if self.settings.value("ClassDept", None) == oldName:
                  self.designateDept(newName)
                if self.settings.value("Grades", None) == oldName:
                  self.designateGrades(newName)
                if self.settings.value("StudentID", None) == oldName:
                  self.designateSID(newName)
                if self.settings.value("StudentMajor", None) == oldName:
                  self.designateStdntMjr(newName)
                if self.settings.value("StudentYear", None) == oldName:
                  self.designateStdntYr(newName)
                if self.settings.value("Terms", None) == oldName:
                  self.designateTerms(newName)
                if self.settings.value("Credits", None) == oldName:
                  self.designateCredits(newName)
                if self.settings.value("FID", None) == oldName:
                  self.designateFID(newName)
            except Exception as e: 
                print(e)
                return
    @QtCore.pyqtSlot()
    def removeNaInColumn(self, col=None):
        try:
            if col is None:
              col = self.columnSelected
            firstNum = len(self.grades.df.index)
            if isinstance(col, list):
              for c in col:
                self.grades.dropMissingValuesInColumn(c)
            else:
              self.grades.dropMissingValuesInColumn(col)
        except Exception as e: 
            print(e)
            return
        self.reload(firstNum == len(self.grades.df.index))
        print(str(firstNum - len(self.grades.df.index)) + ' undefined values dropped in ' + str(self.columnSelected) + ' column.')

    def exportHisto(self):
        try:
            dlg = histoDialogue(self)
            dlg.setWindowTitle("Histogram of "+self.columnSelected)
            if dlg.exec():
              flnm,ttl,ylabel,xlabel,asc,ylg = dlg.getInputs()
              edmlib.seriesToHistogram(self.grades.df[self.columnSelected], fileName=flnm, graphTitle=ttl,sortedAscending=asc,logScale=ylg,xlbl=xlabel,ylbl=ylabel)
              print(str(self.columnSelected) + ' histogram exported.')
        except Exception as e: 
            print(e)

    @QtCore.pyqtSlot()
    def filterColNumeric(self):
            try:
              # print('prefiltering')
              self.removeNaInColumn()
              # print('removed na')
              self.grades.df[self.columnSelected] = self.grades.df[self.columnSelected].apply(float)
              # print('tonumeric ran')
            except:
              print('Error: Column is not numeric. Check values.')
              return
            dlg = filterNumsDialogue(self)
            dlg.setWindowTitle("Filter Column Numerically")
            if dlg.exec():
                try:
                    firstNum = len(self.grades.df.index)
                    minVal, maxVal = dlg.getInputs()
                    # self.model.layoutAboutToBeChanged.emit()
                    filterOutput = 'Values in column '+str(self.columnSelected)
                    if not((minVal is None) or (maxVal is None)):
                      self.grades.df = self.grades.df.loc[(self.grades.df[self.columnSelected] >= minVal) & (self.grades.df[self.columnSelected] <= maxVal)]
                      filterOutput += ' filtered to min ' + str(minVal) + ' and max ' + str(maxVal)
                    elif not(minVal is None):
                      self.grades.df = self.grades.df.loc[self.grades.df[self.columnSelected] >= minVal]
                      filterOutput += ' filtered to min ' + str(minVal)
                    elif not(maxVal is None):
                      self.grades.df = self.grades.df.loc[self.grades.df[self.columnSelected] <= maxVal]
                      filterOutput += ' filtered to max ' + str(maxVal)
                    else:
                      filterOutput += ' not filtered.'
                    self.grades.df.reset_index(inplace = True, drop=True)
                    self.reload((firstNum == len(self.grades.df.index)) or self.columnSelected.endswith(edmlib.pvalSuffix) or self.columnSelected.endswith(edmlib.ttestSuffix))
                    # self.model.layoutChanged.emit()
                    print(filterOutput)
                except Exception as e: 
                    print(e)
                    return
    @QtCore.pyqtSlot()
    def roundCol(self):
            try:
              self.model.layoutAboutToBeChanged.emit()
              self.removeNaInColumn()
              self.grades.df[self.columnSelected] = pd.to_numeric(self.grades.df[self.columnSelected], errors='raise')
              self.model.layoutChanged.emit()
            except:
              print('Error: Column is not numeric. Check values.')
              return
            dlg = roundDialogue(self)
            dlg.setWindowTitle("Round Column Values")
            if dlg.exec():
                try:
                    self.model.layoutAboutToBeChanged.emit()
                    decPlaces = dlg.getInputs()
                    self.grades.df[self.columnSelected] = self.grades.df[self.columnSelected].round(decPlaces)
                    # self.reload()
                    self.model.layoutChanged.emit()
                    print('Values in column '+str(self.columnSelected) +' rounded to ' + str(decPlaces) + ' decimal places.')
                except Exception as e: 
                    print(e)
                    return

    @QtCore.pyqtSlot()
    def avCol(self):
        try:
          self.model.layoutAboutToBeChanged.emit()
          self.removeNaInColumn()
          self.grades.df[self.columnSelected] = pd.to_numeric(self.grades.df[self.columnSelected], errors='raise')
          self.model.layoutChanged.emit()
          mean = self.grades.df[self.columnSelected].mean()
          med = self.grades.df[self.columnSelected].median()
          mode = self.grades.df[self.columnSelected].mode()
          stRnd = lambda x : str(round(x, 3))
          print(self.columnSelected + ' mean: ' + stRnd(mean) + '    median: ' + stRnd(med) + '    mode: ' + stRnd(mode.iloc[0]))
        except:
          print(self.columnSelected + ' mode: ' + str(self.grades.df[self.columnSelected].mode().iloc[0]))
          return

    @QtCore.pyqtSlot()
    def absCol(self):
        try:
          self.model.layoutAboutToBeChanged.emit()
          self.removeNaInColumn()
          self.grades.df[self.columnSelected] = pd.to_numeric(self.grades.df[self.columnSelected], errors='raise')
          self.grades.df[self.columnSelected] = self.grades.df[self.columnSelected].abs()
          self.model.layoutChanged.emit()
        except:
          print('Error: Column is not numeric. Check values.')
          self.model.layoutChanged.emit()
          return

    @QtCore.pyqtSlot()
    def pValCol(self):
        try:
          self.model.layoutAboutToBeChanged.emit()
          self.removeNaInColumn()
          self.grades.df[self.columnSelected] = pd.to_numeric(self.grades.df[self.columnSelected], errors='raise')
          newCol = self.columnSelected + edmlib.pvalSuffix
          self.grades.df[newCol] = edmlib.pvalOfSeries(self.grades.df[self.columnSelected])
          self.model.layoutChanged.emit()
          print('P-Values added to ' + newCol + ' column.')
        except:
          print('Error: Column is not numeric. Check values.')
          self.model.layoutChanged.emit()
          return

    def filterColByVals(self):
        dlg = filterValsDialogue(self)
        dlg.setWindowTitle("Filter Column to Specific Value(s)")
        if dlg.exec():
            try:
                vals = dlg.getInputs()
                firstNum = len(self.grades.df.index)
                self.grades.convertColumnToString(self.columnSelected)
                self.grades.filterColumnToValues(self.columnSelected, vals)
                self.reload(firstNum == len(self.grades.df.index))
                print('Values in '+str(self.columnSelected) +' filtered to: ' + str(vals))
            except Exception as e: 
                print(e)
                return

    def gpaMeanCol(self):
        try:
            self.grades.getGPAMeans()
            self.reload()
            print('Class grade standard deviations available in ' + self.grades.GPA_MEAN_COLUMN + ' column.')
        except Exception as e: 
            print(e)
            return

    def gpaDevCol(self):
        try:
            self.grades.getGPADeviations()
            self.reload()
            print('Class grade standard deviations available in ' + self.grades.GPA_STDDEV_COLUMN + ' column.')
        except Exception as e: 
            print(e)
            return

    def normCol(self):
        try:
            self.grades.getNormalizationColumn()
            self.reload()
            print('Normalized grades ((student grade - class mean) / stdDev) available in ' + self.grades.NORMALIZATION_COLUMN + ' column.')
        except Exception as e: 
            print(e)
            return

    def normByGPACol(self):
        try:
            print('Calculating Normalizations by Student GPA...')
            worker = Worker(self.grades.getNormalizationByGPA)
            worker.signals.finished.connect(self.normByGPAColPrintOut())
            self.threadpool.start(worker)
        except Exception as e: 
            print(e)
            return
    def normByGPAColPrintOut(self):
        def outp():
          print('Normalized grades by GPA ((student grade - student GPA) / studentStdDev) available in ' + self.grades.GPA_NORMALIZATION_COLUMN + ' column.')
          self.reload()
        return outp

    def normByStudByClassCol(self):
        try:
            print('Calculating Normalizations by Student by Class...')
            worker = Worker(self.grades.getNormalizationByStudentByClass)
            worker.signals.finished.connect(self.normByStudByClassColPrintOut())
            self.threadpool.start(worker)
        except Exception as e: 
            print(e)
            return
    def normByStudByClassColPrintOut(self):
        def outp():
          print('Normalized grades by student and class ((normalized student grade - mean student norm) / studentNormStdDev) available in ' + self.grades.STUDENT_CLASS_NORMALIZATION_COLUMN + ' column.')
          self.reload()
        return outp

    def filterGPADeviation(self):
        dlg = filterGPADeviationDialogue(self)
        dlg.setWindowTitle("Filter Classes by GPA Deviation")
        if dlg.exec():
            try:
                # self.model.layoutAboutToBeChanged.emit()
                dev, out, fileNm = dlg.getInputs()
                self.grades.filterByGpaDeviationMoreThan(dev, out, fileNm + '.csv')
                self.reload()
                # self.model.layoutChanged.emit()
            except Exception as e: 
                print(e)
                return
            # print('Filtered data successfully.')

    @QtCore.pyqtSlot()
    def filterByStudentMajor(self):
        dlg = filterStudentMajorDialog(self)
        dlg.setWindowTitle("Filter to Classes and/or Departments")
        if dlg.exec():
            try:
                # self.model.layoutAboutToBeChanged.emit()
                majors = dlg.getInputs()
                self.grades.filterStudentsByMajors(majors)
                self.reload()
                # self.model.layoutChanged.emit()
                print('Majors filtered to: ' + str(majors))
            except Exception as e: 
                print(e)
                return
            print('Filtered data successfully.')

    @QtCore.pyqtSlot()
    def filterByClassOrMajorCorr(self):
        dlg = filterClassesDeptsDialog(self, True)
        dlg.setWindowTitle("Filter to Classes and/or Departments")
        if dlg.exec():
            try:
                # self.model.layoutAboutToBeChanged.emit()
                courses, depts, twoWay = dlg.getInputs()
                self.grades.filterToMultipleMajorsOrClasses(depts, courses, twoWay)
                self.reload()
                # self.model.layoutChanged.emit()
                print('Classes filtered to: ' + str(courses))
                print('Departments filtered to: ' + str(depts))
            except Exception as e: 
                print(e)
                return
            print('Filtered data successfully.')

    @QtCore.pyqtSlot()
    def filterByClassOrMajor(self):
        dlg = filterClassesDeptsDialog(self)
        dlg.setWindowTitle("Filter to Classes and/or Departments")
        if dlg.exec():
            try:
                # self.model.layoutAboutToBeChanged.emit()
                courses, depts = dlg.getInputs()
                self.grades.filterToMultipleMajorsOrClasses(depts, courses)
                # worker = Worker(self.grades.filterToMultipleMajorsOrClasses, majors=depts, classes=courses)
                # worker.signals.finished.connect(self.filterDone)
                # self.threadpool.start(worker)
                # self.model = TableModel(self.grades.df)
                self.reload()
                # self.model.layoutChanged.emit()
                print('Classes filtered to: ' + str(courses))
                print('Departments filtered to: ' + str(depts))
                print('Filtered data successfully.')
            except Exception as e: 
                print(e)
                pass

    def filterDone(self):
        self.model.layoutAboutToBeChanged.emit()
        self.model.layoutChanged.emit()
        print('Filtered to specific classes / departments.')

    def makeSankeyTrackNew(self):
        dlg = sankeyTrackInputNew(self)
        dlg.setWindowTitle("Export Class Track Graph")
        if dlg.exec():
            try:
                grTitle, fileTitle, group, required, minEdge = dlg.getInputs()
                worker = Worker(self.grades.sankeyGraphByCourseTracksOneGroup, courseGroup=group, requiredCourses=required, graphTitle=grTitle, outputName=fileTitle, minEdgeValue=minEdge)
                self.threadpool.start(worker)
            except Exception as e: 
                print(e)
                pass

    def makeSankeyTrack(self):
        dlg = sankeyTrackInput(self)
        dlg.setWindowTitle("Export Class Track Graph (alternate)")
        if dlg.exec():
            try:
                if dlg.orderedCheck.isChecked():
                  grTitle, fileTitle, minEdge, groups, thres = dlg.getInputs()
                  worker = Worker(self.grades.sankeyGraphByCourseTracks, courseGroups=groups, graphTitle=grTitle, outputName=fileTitle, minEdgeValue=minEdge, termThreshold=thres)
                else:
                  grTitle, fileTitle, minEdge, groups = dlg.getInputs()
                  worker = Worker(self.grades.sankeyGraphByCourseTracks, courseGroups=groups, graphTitle=grTitle, outputName=fileTitle, minEdgeValue=minEdge)
                self.threadpool.start(worker)
            except Exception as e: 
                print(e)
                pass

    def makeSankeyTrackAdvanced(self):
        # if 'termNumber' not in self.grades.columns:
        dlg = termNumberInput(self)
        dlg.setWindowTitle("Set Term Numbers / Ordering")
        if dlg.exec():
            try:
                termToVals = dlg.getInputs()
                print(termToVals)
                self.grades.termMapping(termToVals)
                self.reload()
                self.makeSankeyTrack()
            except Exception as e: 
                print(e)
                pass

    def exportMajorChord(self):
        dlg = majorChordInput(self)
        dlg.setWindowTitle("Export Major Chord Graph")
        if dlg.exec():
            try:
              corr, pval, name = dlg.getInputs()
              worker = Worker(self.grades.chordGraphByMajor, corr, pval, name, 200, 230, True, False)
              # worker.signals.finished.connect(self.chordDone(name))
              self.threadpool.start(worker) 
            except Exception as e: 
              print(e)
              pass
    def chordDone(self, name):
      def done():
        self.pic = PaintPicture(self, name + '.png')
      return done

    def exportCliqueHisto(self):
        dlg = cliqueHistogramInput(self)
        dlg.setWindowTitle("Export Clique Histogram")
        if dlg.exec():
            try:
              corr, dup, logSc, title, name = dlg.getInputs()
              worker = Worker(self.grades.outputCliqueDistribution, minCorr=corr, countDuplicates=dup, makeHistogram=True, fileName=name, graphTitle=title, logScale=logSc)
              # worker.signals.finished.connect(self.cliqueDone(name))
              self.threadpool.start(worker) 
            except Exception as e: 
              print(e)
              pass
    def cliqueDone(self, name):
      def done():
        self.pic = PaintPicture(self, name + '.png')
        # file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), name + ".html"))
        # self.web = QWebEngineView()
        # self.web.load(QUrl.fromLocalFile(file_path))
        # self.web.show()
      return done
    
    @QtCore.pyqtSlot()
    def instructorEffectiveness(self):
        dlg = instructorEffectivenessDialog(self)
        dlg.setWindowTitle("Rank Instructor Effectiveness")
        if dlg.exec():
            try:
                classOne, classTwo, filename, minStud = dlg.getInputs()
                if not filename.endswith('.csv'):
                    filename = "".join((filename, '.csv'))
                self.grades.instructorRanks(classOne,classTwo,fileName = filename, minStudents = minStud)
                print('Instructors of ' + classOne + ' class ranked based on future student performance in ' + classTwo + '. Saved to ' + filename + '.')
                dlg2 = csvPreview(self, filename)
                dlg2.setWindowTitle("Instructor Ranking - " + filename)
                dlg2.tableView.sortByColumn(1,1)
                # print (inspect.getsource(QtWidgets.QTableView.setSortingEnabled))
                # print (inspect.getsource(QtWidgets.QTableView.sortByColumn))
                dlg2.exec()
            except Exception as e: 
              print(e)
              pass

    @QtCore.pyqtSlot()
    def instructorEffectivenessAll(self):
        dlg = instructorEffectivenessDialog(self, True)
        dlg.setWindowTitle("Rank Instructors (All classes, may take hours)")
        if dlg.exec():
            try:
                filename, minStud, direct = dlg.getInputs()
                if not filename.endswith('.csv'):
                    filename = "".join((filename, '.csv'))
                worker = Worker(self.grades.instructorRanksAllClasses,fileName = filename, minStudents = minStud, directionality = direct)
                worker.signals.finished.connect(self.instructorEffectivenessAllProc(filename))
                self.threadpool.start(worker)
            except Exception as e: 
              print(e)
              pass
    def instructorEffectivenessAllProc(self, filename):
        def proc():
            print('Instructor rankings saved to ' + filename + '.')
            dlg2 = csvPreview(self, filename)
            dlg2.setWindowTitle("Instructor Ranking - " + filename)
            dlg2.tableView.sortByColumn(1,1)
            dlg2.exec()
        return proc

    @QtCore.pyqtSlot()
    def ttestCalc(self):
        dlg = ttestDialog(self, list(self.grades.df.columns.values))
        dlg.setWindowTitle("Perform t-test")
        if dlg.exec():
            try:
                self.model.layoutAboutToBeChanged.emit()
                c1, c2 = dlg.getInputs()
                self.removeNaInColumn([c1,c2])
                self.grades.df[c1] = pd.to_numeric(self.grades.df[c1], errors='raise')
                self.grades.df[c2] = pd.to_numeric(self.grades.df[c2], errors='raise')
                newCol = c1+'|'+c2+edmlib.ttestSuffix
                self.grades.df[newCol] = edmlib.tTestOfTwoSeries(c1,c2)
                self.model.layoutChanged.emit()
                print('t-test values added to ' + newCol + ' column.')
            except:
                print('Error: Column is not numeric. Check values.')
                self.model.layoutChanged.emit()
                return

    @QtCore.pyqtSlot()
    def dictReplace(self):
        dlg = valuesReplaceDialogue(self)
        dlg.setWindowTitle("Substitute Values in Column")
        if dlg.exec():
            try:
                self.model.layoutAboutToBeChanged.emit()
                replace = dlg.getInputs()
                self.grades.df[str(self.columnSelected)] = self.grades.df[str(self.columnSelected)].map(replace).fillna(self.grades.df[str(self.columnSelected)])
                self.model.layoutChanged.emit()
                print('Replaced ' + str(len(replace)) + ' values in '+ str(self.columnSelected) +' column.')
                # self.reload()
            except Exception as e: 
              print(e)
              pass

    def dictReplaceFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Open", "","Text Files (*.txt)", options=options)
        if fileName:
            try:
                fh = open(fileName)
                lines = [line.rstrip('\n') for line in fh.readlines()]
                replace = {}
                for line in lines:
                  original, replacement = line.split(';')
                  replace[original] = replacement
                fh.close()
                self.model.layoutAboutToBeChanged.emit()
                self.grades.df[str(self.columnSelected)] = self.grades.df[str(self.columnSelected)].map(replace).fillna(self.grades.df[str(self.columnSelected)])
                self.model.layoutChanged.emit()
                print('Replaced ' + str(len(replace)) + ' values in '+ str(self.columnSelected) +' column.')
            except Exception as e: 
                print(e)
                pass

    @QtCore.pyqtSlot()
    def strReplace(self):
        dlg = substituteInput(self)
        dlg.setWindowTitle("Substitute String in Column " + str(self.columnSelected))
        if dlg.exec():
            try:
                self.model.layoutAboutToBeChanged.emit()
                subStr, replace = dlg.getInputs()
                self.grades.substituteSubStrInColumn(str(self.columnSelected), subStr, replace)
                self.model.layoutChanged.emit()
                print('Replaced ' + subStr + ' with ' + replace + ' in ' + str(self.columnSelected) + ' column.')
            except Exception as e: 
              print(e)
              pass

    @QtCore.pyqtSlot()
    def delColumn(self):
        try:
          if self.correlationFile and self.columnSelected in ['course1', 'course2', 'corr', 'P-value', '#students']:
              print('Error: column required for correlation calculations. Not deleted.')
          else:
              self.model.layoutAboutToBeChanged.emit()
              worker = Worker(self.grades.df.drop, self.columnSelected, axis=1, inplace=True)
              worker.signals.finished.connect(self.delColumnProc(self.columnSelected))
              self.threadpool.start(worker) 
        except Exception as e: 
          print(e)
          pass
    def delColumnProc(self, col):
        def proc():
            self.model.layoutChanged.emit()
            print('Removed ' + col + ' column.')
        return proc

    def write(self, text):
        self.terminal.write(text)
        text = text.strip()
        if len(text) > 0:
            self.message.setText(str(text))
    def flush(self):
        pass

    def originalReload(self):
      try:
        self.loadCsv(self.settings.value(getpass.getuser()+"lastFile", self.settings.value("lastFile", None)))
        self.getLastKnownColumns()
        self.doColumnThings()
        self.setGeneralButtons()
        print('File reloaded.')
      except Exception as e: 
          print(e)
          print('Failed to reload.')
          pass
    
    def loadCsv(self, fileName):
        try:
            self.df = pd.read_csv(fileName, dtype=str)
            if {'course1', 'course2', 'corr', 'P-value', '#students'}.issubset(self.df.columns):
              self.correlationFile = True
              self.grades = classCorrelationData(self.df, copyDataFrame=False)
            else:
              self.correlationFile = False
              self.grades = gradeData(self.df, copyDataFrame=False)
              self.getLastKnownColumns()
            self.model = TableModel(self.grades.df)
            self.tableView.setModel(self.model)
            
            self.settings.setValue("lastFile", fileName)
            self.settings.setValue(getpass.getuser()+"lastFile", fileName)
            self.setWindowTitle("EDM - " + self.settings.value("lastFile",None))
        except: 
            print("Unexpected error:", sys.exc_info())
            pass
        
    def writeCsv(self, fileName):
        self.grades.exportCSV(fileName)

    def closeEvent(self, e):
        # Write window size and position to config file
        self.settings.setValue("size", self.size())
        self.settings.setValue("pos", self.pos())

        e.accept()

    @QtCore.pyqtSlot()
    def exportGPADistribution(self):
        dlg = gpaDistributionInput(self)
        dlg.setWindowTitle("Export GPA Distribution Histogram")
        if dlg.exec():
            try:
                graphTtl, fileNm, minClsses = dlg.getInputs()
                worker = Worker(self.grades.outputGpaDistribution, makeHistogram=True, fileName=fileNm, graphTitle=graphTtl, minClasses=minClsses)
                self.threadpool.start(worker) 
            except Exception as e: 
              print(e)
              pass

    def gpaDone(self, name):
      def done():
        self.pic = PaintPicture(self, name)
      return done

    @QtCore.pyqtSlot()
    def pairGraph(self):
        dlg = pairGraphInput(self)
        dlg.setWindowTitle("Export GPA Distribution Histogram")
        if dlg.exec():
            try:
                classOne, classTwo, fileNm = dlg.getInputs()
                worker = Worker(self.grades.coursePairGraph, classOne, classTwo, fileNm)
                self.threadpool.start(worker)
            except Exception as e: 
              print(e)
              pass

    @QtCore.pyqtSlot()
    def exportCSV(self):
        dlg = getFilename(self)
        dlg.setWindowTitle("Export Current State of DataFrame") 
        if dlg.exec():
            try:
                fileNm = dlg.getInputs()
                worker = Worker(self.grades.exportCSV, fileName = fileNm)
                self.threadpool.start(worker)
            except Exception as e: 
              print(e)
              pass

    @QtCore.pyqtSlot()
    def exportCorrelations(self, arg = None):
        dlg = getCorrDialogue(self)
        dlg.setWindowTitle("Export Class Correlations (Warning: may take hours)")
        if dlg.exec():
            try:
                minStdnts, fileNm, directedData, details, byYear = dlg.getInputs()
                worker = Worker(self.grades.exportCorrelationsWithMinNSharedStudents,filename=fileNm, nStudents=minStdnts, directedCorr = directedData, detailed = details, sequenced = byYear)
                self.threadpool.start(worker)
            except Exception as e: 
              print(e)
              pass

    @QtCore.pyqtSlot()
    def designateGrades(self, arg = None):
      if self.grades and arg:
          if arg in self.grades.df.columns:
              self.grades.FINAL_GRADE_COLUMN = arg
              self.settings.setValue("Grades", arg)
          elif arg == ' ':
            self.grades.FINAL_GRADE_COLUMN = 'finalGrade'
            self.settings.setValue("Grades", None)
      elif self.grades and self.columnSelected:
          self.grades.FINAL_GRADE_COLUMN = self.columnSelected
          self.settings.setValue("Grades", self.columnSelected)
          print('Student Grade column set to ' + self.columnSelected)

    @QtCore.pyqtSlot()
    def designateCID(self, arg = None):
      if self.grades and arg:
          if arg in self.grades.df.columns:
              self.grades.CLASS_ID_COLUMN = arg
              self.settings.setValue("ClassID", arg)
          elif arg == ' ':
            self.grades.CLASS_ID_COLUMN = 'classID'
            self.settings.setValue("ClassID", None)
      elif self.grades and self.columnSelected:
          self.grades.CLASS_ID_COLUMN = self.columnSelected
          self.settings.setValue("ClassID", self.columnSelected)
          print('Class ID column set to ' + self.columnSelected)

    @QtCore.pyqtSlot()
    def designateFID(self, arg = None):
      if self.grades and arg:
          if arg in self.grades.df.columns:
              self.grades.FACULTY_ID_COLUMN = arg
              self.settings.setValue("FID", arg)
          elif arg == ' ':
            self.grades.FACULTY_ID_COLUMN = 'facultyID'
            self.settings.setValue("FID", None)
      elif self.grades and self.columnSelected:
          self.grades.FACULTY_ID_COLUMN = self.columnSelected
          self.settings.setValue("FID", self.columnSelected)
          print('Faculty ID column set to ' + self.columnSelected)

    @QtCore.pyqtSlot()
    def designateClssNmbr(self, arg = None):
      if self.grades and arg:
          if arg in self.grades.df.columns:
              self.grades.CLASS_NUMBER_COLUMN = arg
              self.settings.setValue("ClassNumber", arg)
          elif arg == ' ':
            self.grades.CLASS_CREDITS_COLUMN = 'classNumber'
            self.settings.setValue("ClassNumber", None)
      elif self.grades and self.columnSelected:
          self.grades.CLASS_NUMBER_COLUMN = self.columnSelected
          self.settings.setValue("ClassNumber", self.columnSelected)
          print('Class Number column set to ' + self.columnSelected)

    @QtCore.pyqtSlot()
    def designateCredits(self, arg = None):
      if self.grades and arg:
          if arg in self.grades.df.columns:
              self.grades.CLASS_CREDITS_COLUMN = arg
              self.settings.setValue("Credits", arg)
          elif arg == ' ':
            self.grades.CLASS_CREDITS_COLUMN = 'classCredits'
            self.settings.setValue("Credits", None)
      elif self.grades and self.columnSelected:
          self.grades.CLASS_CREDITS_COLUMN = self.columnSelected
          self.settings.setValue("Credits", self.columnSelected)
          print('Class Credits column set to ' + self.columnSelected)

    @QtCore.pyqtSlot()
    def designateDept(self, arg = None):
      if self.grades and arg:
          if arg in self.grades.df.columns:
              self.grades.CLASS_DEPT_COLUMN = arg
              self.settings.setValue("ClassDept", arg)
          elif arg == ' ':
            self.grades.CLASS_DEPT_COLUMN = 'classDept'
            self.settings.setValue("ClassDept", None)
      elif self.grades and self.columnSelected:
          self.grades.CLASS_DEPT_COLUMN = self.columnSelected
          self.settings.setValue("ClassDept", self.columnSelected)
          print('Class Department column set to ' + self.columnSelected)

    @QtCore.pyqtSlot()
    def designateSID(self, arg = None):
      if self.grades and arg:
          if arg in self.grades.df.columns:
              self.grades.STUDENT_ID_COLUMN = arg
              self.settings.setValue("StudentID", arg)
          elif arg == ' ':
            self.grades.STUDENT_ID_COLUMN = 'studentID'
            self.settings.setValue("StudentID", None)
      elif self.grades and self.columnSelected:
          self.grades.STUDENT_ID_COLUMN = self.columnSelected
          self.settings.setValue("StudentID", self.columnSelected)
          print('Student ID column set to ' + self.columnSelected)

    @QtCore.pyqtSlot()
    def designateStdntMjr(self, arg = None):
      if self.grades and arg:
          if arg in self.grades.df.columns:
              self.grades.STUDENT_MAJOR_COLUMN = arg
              self.settings.setValue("StudentMajor", arg)
          elif arg == ' ':
            self.grades.STUDENT_MAJOR_COLUMN = 'studentMajor'
            self.settings.setValue("StudentMajor", None)
      elif self.grades and self.columnSelected:
          self.grades.STUDENT_MAJOR_COLUMN = self.columnSelected
          self.settings.setValue("StudentMajor", self.columnSelected)
          print('Student Major column set to ' + self.columnSelected)

    @QtCore.pyqtSlot()
    def designateStdntYr(self, arg = None):
      if self.grades and arg:
          if arg in self.grades.df.columns:
              self.grades.STUDENT_YEAR_COLUMN = arg
              self.settings.setValue("StudentYear", arg)
          elif arg == ' ':
            self.grades.STUDENT_YEAR_COLUMN = 'studentYear'
            self.settings.setValue("StudentYear", None)
      elif self.grades and self.columnSelected:
          self.grades.STUDENT_YEAR_COLUMN = self.columnSelected
          self.settings.setValue("StudentYear", self.columnSelected)
          print('Student Year column set to ' + self.columnSelected)

    @QtCore.pyqtSlot()
    def designateClassCode(self, arg = None):
      if self.grades and arg:
          if arg in self.grades.df.columns:
              self.grades.CLASS_CODE_COLUMN = arg
              self.settings.setValue("ClassCode", arg)
          elif arg == ' ':
            self.grades.CLASS_CODE_COLUMN = 'classCode'
            self.settings.setValue("ClassCode", None)
      elif self.grades and self.columnSelected:
          self.grades.CLASS_CODE_COLUMN = self.columnSelected
          self.settings.setValue("ClassCode", self.columnSelected)
          print('Class Code column set to ' + self.columnSelected)

    @QtCore.pyqtSlot()
    def designateTerms(self, arg = None):
      if self.grades and arg:
          if arg in self.grades.df.columns:
              self.grades.TERM_COLUMN = arg
              self.settings.setValue("Terms", arg)
          elif arg == ' ':
            self.grades.TERM_COLUMN = 'term'
            self.settings.setValue("Terms", None)
      elif self.grades and self.columnSelected:
          self.grades.TERM_COLUMN = self.columnSelected
          self.settings.setValue("Terms", self.columnSelected)
          print('Term column set to ' + self.columnSelected)

    @QtCore.pyqtSlot()
    def on_pushButtonWrite_clicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"Save As","","CSV Files (*.csv)", options=options)
        if fileName:
            if not fileName.endswith('.csv'):
              fileName = fileName + '.csv'
            self.writeCsv(fileName)
            print('File saved.')

    @QtCore.pyqtSlot()
    def on_pushButtonLoad_clicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Open", "","CSV Files (*.csv)", options=options)
        if fileName:
            self.loadCsv(fileName)
            self.getLastKnownColumns()
            self.doColumnThings()
            self.setGeneralButtons()
            print('File opened.')
    
    @QtCore.pyqtSlot()
    def openGraph(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        directory = self.settings.value(getpass.getuser()+"ExportDir", "")
        fileName, _ = QFileDialog.getOpenFileName(self,"Open Graph", directory,"HTML Files (*.html)", options=options)
        if fileName:
            print('Graph opened.')
            webbrowser.open("file://" + fileName,new=2)

    @QtCore.pyqtSlot()
    def setExportDir(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        directory = QFileDialog.getExistingDirectory(self,"Set Export Directory", options=options)
        if directory:
            self.settings.setValue(getpass.getuser()+"ExportDir", directory)
            edmlib.makeExportDirectory(directory)
            print('Directory set to ' + directory + ' for user ' + getpass.getuser())

class TableModel(QtCore.QAbstractTableModel):

    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data
        self.moreDecimals = [i for i in range(len(self._data.columns)) if self._data.columns[i].endswith(edmlib.pvalSuffix) or self._data.columns[i].endswith(edmlib.ttestSuffix)]          

    def data(self, index, role):
        if role == Qt.DisplayRole:
            value = str(self._data.iloc[index.row(), index.column()])
            try:
              value = float(value)
              if value.is_integer():
                return str(int(value))
              if index.column() in self.moreDecimals:
                return str(round(value, 6))
              return str(round(value, 3))
            except ValueError:
              return value

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]
    
    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])

            if orientation == Qt.Vertical:
                return str(self._data.index[section])
    def sort(self, column, order):
        """Sort table by given column number.
        """
        # print('sort clicked col {} order {}'.format(column, order))
        self.layoutAboutToBeChanged.emit()
        # print(self._df.columns[column])
        self._data[self._data.columns[column]] = pd.to_numeric(self._data[self._data.columns[column]],errors='ignore')
        self._data.sort_values(self._data.columns[column], ascending=order == Qt.AscendingOrder, inplace=True, kind='mergesort')
        self._data.reset_index(inplace=True, drop=True) # <-- this is the change
        # print(self._df)
        self.layoutChanged.emit()

class statsDialogue(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        vlayout = QVBoxLayout()
        hLayout = QHBoxLayout()

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok, self)
        forms = []
        count = 0
        for column in parent.grades.df.columns.values:
          if count % 20 == 0:
            forms.append(QFormLayout())
            forms[-1].setVerticalSpacing(0)
          button = QPushButton(self)
          # self.buttons.append(button)
          vals = dict(parent.grades.df[column].value_counts())
          vals = {k: v for k, v in sorted(vals.items(), key=lambda item: item[1], reverse=True)} # sort keys by values in descending order
          num = parent.grades.df[column].nunique()
          button.setText('Show Values')
          button.clicked.connect(self.showdialog(num, vals, column))
          forms[-1].addRow('Unique values in ' + str(column) + ': ' + str(num), button)
          count += 1
        for form in forms:
          hLayout.addLayout(form)
        vlayout.addLayout(hLayout)
        vlayout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        self.setLayout(vlayout)

    def showdialog(self, num, vals, column):
        def dlg():
            dlg2 = csvPreview(self, pd.DataFrame(vals.items(), columns=['Value', 'Frequency']))
            dlg2.resize(QSize(500, 480))
            dlg2.tableView.sortByColumn(1,1)
            dlg2.setWindowTitle("Stats on column - " + column)
            dlg2.exec()
            # msg = MyMessageBox()
            # # msg.setIcon(QMessageBox.Information)
            
            # if num < 500:
            #   msg.setInformativeText(str(vals)[1:-1])
            #   msg.setText(str(num) + " unique values found: ")
            # else:
            #   msg.setDetailedText(str(vals)[1:-1])
            #   msg.setText(str(num) + " unique values found:\t\t\t\t\t")
            # msg.setWindowTitle(str(column) +  " Unique Values")
            # # msg.setDetailedText(str(vals)[1:-1])
            # msg.setStandardButtons(QMessageBox.Ok)
            # msg.exec_()
        return dlg

    # def getInputs(self):
    #     return (self.first.value(), self.second.value(), self.third.text())

class valuesReplaceDialogue(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        vlayout = QVBoxLayout()
        hLayout = QHBoxLayout()
        self.col = parent.columnSelected
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok, self)
        form = QFormLayout()
        form.setVerticalSpacing(0)
        self.inputs = []
        vals = sorted(list(parent.grades.df[self.col].astype(str).unique()))
        for val in vals:
          self.inputs.append((val, QLineEdit(self)))
          self.inputs[-1][1].setMaximumWidth(100)
          self.inputs[-1][1].setFixedWidth(100)
          form.addRow(str(val) + ' replacement: ', self.inputs[-1][1])
        hLayout.addLayout(form)
        scroll = QScrollArea() 
        widget = QWidget() 
        widget.setLayout(hLayout)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        vlayout.addWidget(scroll)
        form2 = QFormLayout()
        self.saveSubstitutions = QCheckBox()
        self.saveSubstitutions.setChecked(False)
        form2.addRow('Save substitutions to file ', self.saveSubstitutions)
        vlayout.addLayout(form2)
        vlayout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        self.setLayout(vlayout)

    def getInputs(self):
        replacements = {}
        for inputPair in self.inputs:
              if len(inputPair[1].text()) > 0:
                  replacements[inputPair[0]] = inputPair[1].text()
        if self.saveSubstitutions.isChecked() and len(replacements) > 0:
          with open(self.col+'_substitutions.txt', 'w') as f:
            for key, value in replacements.items():
              f.write(key + ';' + value + '\n')
        return (replacements)

class PaintPicture(QDialog):
    def __init__(self, parent=None, fileName=None):
        super(PaintPicture, self).__init__()

        layout = QVBoxLayout()
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok, self)

        image = QImage(fileName)

        self.imageLabel = QLabel()
        self.imageLabel.setPixmap(QPixmap.fromImage(image))

        layout.addWidget(self.imageLabel)
        layout.addWidget(buttonBox)

        self.setLayout(layout)
        self.setWindowTitle(fileName)
        self.show()
        buttonBox.accepted.connect(self.accept)



class MyMessageBox(QMessageBox):
    def __init__(self):
        self.isFirst = True
        QMessageBox.__init__(self)
        self.setSizeGripEnabled(True)

    def event(self, e):
        result = QMessageBox.event(self, e)

        self.setMinimumHeight(0)
        self.setMaximumHeight(16777215)
        self.setMinimumWidth(0)
        self.setMaximumWidth(16777215)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        textEdit = self.findChild(QTextEdit)
        if textEdit != None :
            textEdit.setMinimumHeight(0)
            textEdit.setMaximumHeight(16777215)
            textEdit.setMinimumWidth(0)
            textEdit.setMaximumWidth(16777215)
            textEdit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        if self.isFirst:
            button = self.findChild(QPushButton)
            if button != None:
                if button.text() == 'Show Details...':
                    button.clicked.emit()
                    self.isFirst = False

        return result

class columnDialog(QDialog):
    def __init__(self, parent=None, options = None):
        super().__init__(parent)
        def addOptions(combo):
          combo.addItem(' ')
          for option in options:
            combo.addItem(option)
        def setDefault(index, name):
          if parent.settings.value(name, None) in parent.grades.df.columns:
            self.combos[index].setCurrentIndex(parent.grades.df.columns.get_loc(parent.settings.value(name, None))+1)
        self.combos = []
        for i in range(11):
          self.combos.append(QComboBox(self))
          addOptions(self.combos[i])

        if parent.settings.value("ClassID", None):
          setDefault(2, "ClassID")
        if parent.settings.value("ClassCode", None):
          setDefault(10, "ClassCode")
        if parent.settings.value("FID", None):
          setDefault(9, "FID")
        if parent.settings.value("ClassNumber", None):
          setDefault(1, "ClassNumber")
        if parent.settings.value("ClassDept", None):
          setDefault(0, "ClassDept")
        if parent.settings.value("Grades", None):
          setDefault(4, "Grades")
        if parent.settings.value("StudentID", None):
          setDefault(3, "StudentID")
        if parent.settings.value("StudentMajor", None):
          setDefault(6, "StudentMajor")
        if parent.settings.value("StudentMajor", None):
          setDefault(7, "StudentYear")
        if parent.settings.value("Terms", None):
          setDefault(5, "Terms")
        if parent.settings.value("Credits", None):
          setDefault(8, "Credits")
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Class Department Column (Psych in Psych1000): ", self.combos[0])
        layout.addRow("Class Number Column (1000 in Psych1000): ", self.combos[1])
        layout.addRow("Class ID Column (number specific to class): ", self.combos[2])
        layout.addRow("Student ID Column: ", self.combos[3])
        layout.addRow("Student Grade Column (0.0 - 4.0+): ", self.combos[4])
        layout.addRow("Term Column (sortable): ", self.combos[5])
        layout.addRow("Student Major Column: ", self.combos[6])
        layout.addRow("Student Year Column: ", self.combos[7])
        layout.addRow("Class Credits Column (optional): ", self.combos[8])
        layout.addRow("Faculty ID Column (optional): ", self.combos[9])
        layout.addRow("Class Code Column (optional): ", self.combos[10])
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return [x.currentText() for x in self.combos]

class ttestDialog(QDialog):
    def __init__(self, parent=None, options = None):
        super().__init__(parent)
        def addOptions(combo):
          for option in options:
            combo.addItem(option)
        self.combos = []
        for i in range(2):
          self.combos.append(QComboBox(self))
          addOptions(self.combos[i])

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Column 1: ", self.combos[0])
        layout.addRow("Column 2: ", self.combos[1])

        layout.addWidget(buttonBox)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return [x.currentText() for x in self.combos]

class getFilename(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.first = QLineEdit(self)
        self.first.setText('csvExport')
        
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Output CSV file name: ", self.first)

        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return (self.first.text())

class getCorrDialogue(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.first = QSpinBox(self)
        self.first.setMinimum(0)
        self.first.setSingleStep(1)
        self.first.setValue(20)
        self.second = QLineEdit(self)
        self.second.setText('classCorrelations')
        self.third = QCheckBox(self)
        self.third.setChecked(False)
        self.fourth = QCheckBox(self)
        self.fourth.setChecked(False)
        self.fifth = QCheckBox(self)
        self.fifth.setChecked(False)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Minimum number of students shared between two classes: ", self.first)
        layout.addRow("Output CSV file name: ", self.second)
        layout.addRow("Include class order correlations (slower): ", self.third)
        layout.addRow("Include class grade details: ", self.fourth)
        layout.addRow("Include by-year details: ", self.fifth)

        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return (self.first.value(), self.second.text(), self.third.isChecked(), self.fourth.isChecked(), self.fifth.isChecked())

class filterGPADeviationDialogue(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.first = preciseSpinBox(self)
        self.first.setMinimum(0.0)
        self.first.setMaximum(1.0)
        self.first.setSingleStep(0.05)
        self.first.setValue(0.2)
        self.second = QCheckBox(self)
        self.second.setChecked(False)
        self.third = QLineEdit(self)
        self.third.setText('droppedData')
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Minimum class GPA Deviation: ", self.first)
        layout.addRow("Output dropped data to file: ", self.second)
        layout.addRow("Dropped data file name: ", self.third)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return (self.first.value(), self.second.isChecked(), self.third.text())

class filterNumsDialogue(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.firstBox = QGroupBox('Set Minimum')
        self.firstBox.setCheckable(True)
        self.secondBox = QGroupBox('Set Maximum')
        self.secondBox.setCheckable(True)
        # self.firstBox.setFlat(True)
        self.firstH = QHBoxLayout()
        self.secondH = QHBoxLayout()

        self.first = preciseSpinBox(self)
        # self.secondBox = QGroupBox('Minimum')
        self.second = preciseSpinBox(self)
        # self.firstCheck = QCheckBox(self)
        self.firstBox.setChecked(True)
        self.secondBox.setChecked(True)
        # self.secondCheck = QCheckBox(self)
        # self.secondCheck.setChecked(True)
        self.firstH.addWidget(self.first)
        self.secondH.addWidget(self.second)
        # self.firstH.addWidget(self.firstCheck)
        self.firstBox.setLayout(self.firstH)
        self.secondBox.setLayout(self.secondH)
        # self.firstBox.setStyleSheet("QGroupBox{padding-top:1px; margin-top:-20px}")
        # self.firstBox.setStyleSheet("border-radius: 0px; margin-top: 3ex;")
        # self.secondBox.setStyleSheet("border-radius:0px; margin-top:1ex;")

        # self.firstBox.setContentsMargins(0, 0, 0, 0)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QVBoxLayout(self)
        # layout.addRow("Minimum: ", self.firstBox)
        # layout.addRow("Apply Minimum: ", self.firstBox)
        layout.addWidget(self.firstBox)
        layout.addWidget(self.secondBox)
        # layout.addRow("Maximum: ", self.second)
        # layout.addRow("Apply Maximum: ", self.secondCheck)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        check = lambda x : x[0].value() if x[1].isChecked() else None           
        return (check([self.first, self.firstBox]), check([self.second, self.secondBox]))

class roundDialogue(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.first = QSpinBox(self)
        self.first.setMinimum(0)
        self.first.setSingleStep(1)
        self.first.setValue(2)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Decimal places: ", self.first)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
            return (self.first.value())

class histoDialogue(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.first = QLineEdit(self)
        self.first.setMaximumWidth(200)
        self.first.setFixedWidth(200)
        self.first.setText(parent.columnSelected)
        self.second = QLineEdit(self)
        self.second.setMaximumWidth(200)
        self.second.setFixedWidth(200)
        self.second.setText(parent.columnSelected)
        self.third = QLineEdit(self)
        self.third.setMaximumWidth(200)
        self.third.setFixedWidth(200)
        self.third.setText('Frequency')
        self.fourth = QLineEdit(self)
        self.fourth.setMaximumWidth(200)
        self.fourth.setFixedWidth(200)
        self.fourth.setText('Value')
        self.fifth = QCheckBox(self)
        self.fifth.setChecked(True)
        self.sixth = QCheckBox(self)
        self.sixth.setChecked(False)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Filename: ", self.first)
        layout.addRow("Graph Title: ", self.second)
        layout.addRow("Y-Axis Text: ", self.third)
        layout.addRow("X-Axis Text: ", self.fourth)
        layout.addRow("Sort Ascending: ", self.fifth)
        layout.addRow("Y-axis log-scale: ", self.sixth)

        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
            return ([x.text() for x in [self.first,self.second,self.third,self.fourth]]+[self.fifth.isChecked(),self.sixth.isChecked()])

class renameColumnDialogue(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.first = QLineEdit(self)
        self.first.setMaximumWidth(200)
        self.first.setFixedWidth(200)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Rename"+ parent.columnSelected +" to: ", self.first)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
            return (self.first.text())

class filterValsDialogue(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.first = QLineEdit(self)
        self.first.setMaximumWidth(400)
        self.first.setFixedWidth(400)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Value(s) (seperate by comma): ", self.first)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
            text = self.first.text()
            if text.endswith('\r\n'):
              text = text[:-len('\r\n')]
            comms = text.split(',')
            xl = text.split('\r\n')
            if len(comms) < len(xl):
              return xl
            return comms

class filterStudentMajorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.first = QLineEdit(self)
        self.first.setMaximumWidth(400)
        self.first.setFixedWidth(400)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Student Majors (seperate by comma): ", self.first)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
            return (self.first.text().split(','))

class csvPreview(QDialog):
    def __init__(self, parent = None, data=None):
        super().__init__(parent)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok, self)

        try:
            self.tableView = QtWidgets.QTableView(self)
            self.tableView.installEventFilter(self)

            if isinstance(data, str):
              self.df = pd.read_csv(data, dtype=str)
            elif isinstance(data, pd.DataFrame):
              self.df = data
            self.grades = gradeData(self.df, copyDataFrame=False)
            self.model = TableModel(self.grades.df)
            self.tableView.setModel(self.model)
            self.tableView.setSortingEnabled(True)
            self.tableView.horizontalHeader().setStretchLastSection(True)
            self.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            self.tableView.horizontalHeader().setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            self.tableView.horizontalHeader().customContextMenuRequested.connect(self.onColumnRightClick)
            self.first = True
            self.doColumnThings()
        except: 
            pass

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.tableView)
        self.layout.addWidget(self.buttonBox)

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.resize(QSize(700, 480))
        # self.setLayout(layout)
        # self.show()
    def onColumnRightClick(self, QPos=None):       
        parent=self.sender()
        pPos=parent.mapToGlobal(QtCore.QPoint(0, 0))
        mPos=pPos+QPos
        column = self.tableView.horizontalHeader().logicalIndexAt(QPos)
        label = self.model.headerData(column, Qt.Horizontal, Qt.DisplayRole)
        self.columnSelected = label
        self.rcMenu.move(mPos)
        self.rcMenu.show()

    def doColumnThings(self):
        self.columnSelected = None
        self.rcMenu=QMenu(self)
        self.substituteMenu = self.rcMenu.addMenu('Substitute')
        substituteInColumn = QAction('Substitute in Column...', self)
        substituteInColumn.triggered.connect(self.strReplace)
        self.substituteMenu.addAction(substituteInColumn)
        dictReplaceInColumn = QAction('Substitute Many Values...', self)
        dictReplaceInColumn.triggered.connect(self.dictReplace)
        self.substituteMenu.addAction(dictReplaceInColumn)
        dictReplaceTxt = QAction('Use substitution file...', self)
        dictReplaceTxt.triggered.connect(self.dictReplaceFile)
        self.substituteMenu.addAction(dictReplaceTxt)
        self.fNumMenu = self.rcMenu.addMenu('Filter / Numeric Operations')
        valFilter = QAction('Filter Column to Value(s)...', self)
        valFilter.triggered.connect(self.filterColByVals)
        self.fNumMenu.addAction(valFilter)
        NumericFilter = QAction('Filter Column Numerically...', self)
        NumericFilter.triggered.connect(self.filterColNumeric)
        self.fNumMenu.addAction(NumericFilter)
        absFilter = QAction('Make Absolute Values', self)
        absFilter.triggered.connect(self.absCol)
        self.fNumMenu.addAction(absFilter)
        avStats = QAction('Get Mean / Med. / Mode', self)
        avStats.triggered.connect(self.avCol)
        self.fNumMenu.addAction(avStats)
        pval = QAction('Calculate P-Value', self)
        pval.triggered.connect(self.pValCol)
        self.fNumMenu.addAction(pval)
        roundFilter = QAction('Round Column...', self)
        roundFilter.triggered.connect(self.roundCol)
        self.fNumMenu.addAction(roundFilter)
        NAFilter = QAction('Drop Undefined Values in Column', self)
        NAFilter.triggered.connect(self.removeNaInColumn)
        self.rcMenu.addAction(NAFilter)
        # if not self.correlationFile:
        deleteColumn = QAction('Delete Column (permanent)', self)
        deleteColumn.triggered.connect(self.delColumn)
        self.rcMenu.addAction(deleteColumn)

    @QtCore.pyqtSlot()
    def strReplace(self):
        MyWindow.strReplace(self)
    @QtCore.pyqtSlot()
    def dictReplace(self):
        MyWindow.dictReplace(self)
    @QtCore.pyqtSlot()
    def dictReplaceFile(self):
        MyWindow.dictReplaceFile(self)
    @QtCore.pyqtSlot()
    def filterColByVals(self):
        MyWindow.filterColByVals(self)
    @QtCore.pyqtSlot()
    def filterColNumeric(self):
        MyWindow.filterColNumeric(self)
    @QtCore.pyqtSlot()
    def absCol(self):
        MyWindow.absCol(self)
    @QtCore.pyqtSlot()
    def avCol(self):
        MyWindow.avCol(self)  
    @QtCore.pyqtSlot()
    def pValCol(self):
        MyWindow.pValCol(self)  
    @QtCore.pyqtSlot()
    def roundCol(self):
        MyWindow.roundCol(self)  
    @QtCore.pyqtSlot()
    def removeNaInColumn(self):
        MyWindow.removeNaInColumn(self)
    @QtCore.pyqtSlot()
    def delColumn(self):
        MyWindow.delColumn(self)
    def reload(self):
        MyWindow.reload(self)
    def getLastKnownColumns(self):
        MyWindow.getLastKnownColumns(self)
    def eventFilter(self, source, event):
        if (event.type() == QtCore.QEvent.KeyPress and
            event.matches(QtGui.QKeySequence.Copy)):
            self.copySelection()
            return True
        return super(csvPreview, self).eventFilter(source, event)
    def copySelection(self):
        MyWindow.copySelection(self)

    def getInputs(self):
        return

class instructorEffectivenessDialog(QDialog):
    def __init__(self, parent=None, allClasses= False):
        super().__init__(parent)
        self.allClass = allClasses
        if not allClasses:
          self.first = QLineEdit(self)
          self.first.setMaximumWidth(200)
          self.first.setFixedWidth(200)
          self.second = QLineEdit(self)
          self.second.setMaximumWidth(200)
          self.second.setFixedWidth(200)
        self.third = QLineEdit(self)
        self.third.setMaximumWidth(200)
        self.third.setFixedWidth(200)
        self.third.setText('instructorRanking')
        self.fourth = QSpinBox(self)
        self.fourth.setMinimum(1)
        self.fourth.setMaximum(999999)
        self.fourth.setSingleStep(1)
        self.fourth.setValue(1)
        self.fourth.setMaximumWidth(200)
        self.fourth.setFixedWidth(200)
        if allClasses:
          self.fifth = preciseSpinBox(self)
          self.fifth.setMinimum(0.5)
          self.fifth.setMaximum(1.0)
          self.fifth.setSingleStep(0.1)
          self.fifth.setValue(0.8)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        if not allClasses:
          layout.addRow("First course (instructors from here): ", self.first)
          layout.addRow("Second course (indicates benefit of instructor): ", self.second)
        layout.addRow("Output file name: ", self.third)
        layout.addRow("Minimum number of Students per instructor: ", self.fourth)
        if allClasses:
          layout.addRow("Minimum Class Directionality (0.5 - 1.0): ", self.fifth)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        if not self.allClass:
          return (self.first.text(), self.second.text(), self.third.text(), self.fourth.value())
        else:
          return (self.third.text(), self.fourth.value(), self.fifth.value())

class filterClassesDeptsDialog(QDialog):
    def __init__(self, parent=None, corr=False):
        super().__init__(parent)
        self.corrDialog = corr
        self.first = QLineEdit(self)
        self.first.setMaximumWidth(400)
        self.first.setFixedWidth(400)
        self.second = QLineEdit(self)
        self.second.setMaximumWidth(400)
        self.second.setFixedWidth(400)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Classes (dept+number or classcode, no spaces, sep. by comma): ", self.first)
        layout.addRow("Departments (seperate by comma): ", self.second)
        if corr:
          self.third = QCheckBox(self)
          self.third.setChecked(True)
          layout.addRow("Both classes must match requirements: ", self.third)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        if not self.corrDialog:
            return (self.first.text().split(','), self.second.text().split(','))
        else:
            return (self.first.text().split(','), self.second.text().split(','), self.third.isChecked())

class termNumberInput(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        vlayout = QVBoxLayout()
        hLayout = QHBoxLayout()

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        forms = []
        self.vals = []
        count = 0
        self.terms = sorted(list(parent.grades.df[parent.grades.TERM_COLUMN].unique()))
        for term in self.terms:
          if count % 15 == 0:
            forms.append(QFormLayout())
            forms[-1].setVerticalSpacing(0)
          self.vals.append(preciseSpinBox(dec=2))
          self.vals[-1].setMinimum(0.0)
          self.vals[-1].setSingleStep(0.05)
          self.vals[-1].setValue(round(count, 1))
          forms[-1].addRow(str(term) + ': ', self.vals[-1])
          count += 1
        for form in forms:
          hLayout.addLayout(form)
        # otherOpts = QFormLayout()
        
        vlayout.addLayout(hLayout)
        # vlayout.addLayout(otherOpts)
        vlayout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        self.setLayout(vlayout)

    def getInputs(self):
        termToVal = {}
        for i in range(len(self.terms)):
            termToVal[self.terms[i]] = round(self.vals[i].value(),2)
        return (termToVal)

class sankeyTrackInputNew(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.first = QLineEdit(self)
        self.first.setText('Class Tracks')
        self.second = QLineEdit(self)
        self.second.setText('sankeyTracks')
        self.third = QLineEdit(self)
        self.third.setMaximumWidth(300)
        self.third.setFixedWidth(300)
        self.fourth = QLineEdit(self)
        self.fourth.setMaximumWidth(300)
        self.fourth.setFixedWidth(300)
        self.fifth = QSpinBox(self)
        self.fifth.setMinimum(0)
        self.fifth.setSingleStep(1)
        self.fifth.setValue(0)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Graph Title: ", self.first)
        layout.addRow("File name: ", self.second)
        layout.addRow('Classes (seperate by comma): ', self.third)
        layout.addRow('Required Classes to Count Student (seperate by comma, all by default): ', self.fourth)
        layout.addRow('Minimum Edge Value: ', self.fifth)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return (self.first.text(), self.second.text(), self.third.text().split(","), self.fourth.text().split(","), self.fifth.value())

class gradePredictDialogue(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        parent.grades.getUniqueIdentifiersForSectionsAcrossTerms()
        self.classes = sorted(parent.grades.df[parent.grades.CLASS_CODE_COLUMN].unique().tolist())
        parent.grades.dropMissingValuesInColumn(parent.grades.FINAL_GRADE_COLUMN)
        parent.grades.convertColumnToNumeric(parent.grades.FINAL_GRADE_COLUMN)
        self.possibleGrades = [str(x) for x in sorted(parent.grades.df[parent.grades.FINAL_GRADE_COLUMN].unique().tolist(), reverse=True)]
        self.pastGradesBox = QGroupBox("Past Grades")
        self.form = QFormLayout()
        self.pastGradesCombos = []
        for x in range(5):
          self.addPastGrade()
        self.predictBox = QGroupBox("Classes to Predict")
        self.form2 = QFormLayout()
        self.predictCombos = []
        self.addPrediction()
        self.pastGradesBox.setLayout(self.form)
        self.predictBox.setLayout(self.form2)

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.pastGradesBox)
        self.plus = QPushButton(self)
        self.plus.setText("+")
        self.plus.clicked.connect(self.addPastGrade)
        self.minus = QPushButton(self)
        self.minus.setText("-")
        self.minus.clicked.connect(self.delPastGrade)
        firstButtons = QHBoxLayout()
        firstButtons.addWidget(self.minus)
        firstButtons.addWidget(self.plus)
        self.form.addRow(firstButtons)
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.predictBox)
        self.plus2 = QPushButton(self)
        self.plus2.setText("+")
        self.plus2.clicked.connect(self.addPrediction)
        self.minus2 = QPushButton(self)
        self.minus2.setText("-")
        self.minus2.clicked.connect(self.delPrediction)
        secondButtons = QHBoxLayout()
        secondButtons.addWidget(self.minus2)
        secondButtons.addWidget(self.plus2)
        self.form2.addRow(secondButtons)

        self.modeBox = QGroupBox('Mode')
        self.modeLayout = QVBoxLayout()
        self.modes = ['Nearest Neighbor', 'Mean of Three Nearest', 'Try All']
        self.modeButtons = [QRadioButton(x) for x in self.modes]
        self.modeButtons[0].setChecked(True)
        for modeButton in self.modeButtons:
          self.modeLayout.addWidget(modeButton)
        self.modeBox.setLayout(self.modeLayout)
        self.vbox.addWidget(self.modeBox)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.vbox.addWidget(self.buttonBox)
        self.layout.addLayout(self.vbox)
        self.setLayout(self.layout)

    def addPastGrade(self):
        self.addGroup(self.form, self.pastGradesCombos, self.classes, self.possibleGrades)

    def delPastGrade(self):
        self.delGroup(self.form, self.pastGradesBox, self.pastGradesCombos)
      
    def addPrediction(self):
        self.predictCombos.append(QComboBox(self))
        self.predictCombos[-1].addItem(' ')
        self.fillCombo(self.predictCombos[-1], self.classes)
        self.form2.insertRow(len(self.predictCombos) - 1, 'Class ' + str(len(self.predictCombos)), self.predictCombos[-1])

    def delPrediction(self):
        self.delGroup(self.form2, self.predictBox, self.predictCombos)

    def addOptions(self, combos, options, options2):
        combos[0].addItem(' ')
        self.fillCombo(combos[0], options)
        self.fillCombo(combos[1], options2)

    def fillCombo(self, combo, optionList):
        for option in optionList:
          combo.addItem(option)

    def addGroup(self, form, combos, options, options2):
        combos.append((QComboBox(self), QComboBox(self)))
        self.addOptions(self.pastGradesCombos[-1], options, options2)
        form.insertRow(len(combos) - 1, combos[-1][0], combos[-1][1])

    def delGroup(self, form, group, combos):
        if len(combos) > 1:
            form.removeRow(form.rowCount()-2)
            del combos[-1]
            group.adjustSize()
            self.adjustSize()

    def getInputs(self):
        past = {x[0].currentText():float(x[1].currentText()) for x in self.pastGradesCombos if (x[0].currentText() != ' ')}
        predict = [x.currentText() for x in self.predictCombos if (x.currentText() != ' ')]
        self.translate = {'Nearest Neighbor':'nearest', 'Mean of Three Nearest':'nearestThree', 'Try All':'tryAll'}
        for modeButton in self.modeButtons:
          if modeButton.isChecked():
            return (past, predict, self.translate[modeButton.text()])

class sankeyTrackInput(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ordered = 'termOrder' in parent.grades.df.columns
        self.formGroupBox = QGroupBox("Sankey Graph")
        self.form = QFormLayout(self)
        self.first = QLineEdit(self)
        self.first.setText('Class Tracks')
        self.form.addRow('Graph Title: ', self.first)
        self.second = QLineEdit(self)
        self.second.setText('sankeyTracks')
        self.form.addRow('File Name: ', self.second)
        self.third = QSpinBox(self)
        self.third.setValue(0)
        self.form.addRow('Minimum Edge Value: ', self.third)
        if self.ordered:
          self.orderedCheck = QCheckBox(self)
          self.orderedCheck.setChecked(True)
          self.form.addRow('Use designated term order column: ', self.orderedCheck)
          self.maxConsecutive = preciseSpinBox(dec=2)
          self.maxConsecutive.setMinimum(0.0)
          self.maxConsecutive.setSingleStep(0.05)
          self.maxConsecutive.setValue(round(1,0))
          self.form.addRow('Maximum Difference for a consecutive term (if ordered): ', self.maxConsecutive)
        self.groups = []
        self.formGroupBox.setLayout(self.form)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.formGroupBox)
        self.addGroup()
        self.addGroup()
        self.plus = QPushButton(self)
        self.plus.setText("+")
        self.plus.clicked.connect(self.addGroup)
        self.minus = QPushButton(self)
        self.minus.setText("-")
        self.minus.clicked.connect(self.delGroup)
        self.layout.addWidget(self.plus)
        self.layout.addWidget(self.minus)
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

    def addGroup(self):
        line = QLineEdit()
        line.setMaximumWidth(300)
        line.setFixedWidth(300)
        self.groups.append(line)
        self.form.addRow('Class Group ' + str(len(self.groups)) + ' (seperate by commas only): ', self.groups[-1])

    def delGroup(self):
        if len(self.groups) > 2:
            self.form.removeRow(self.form.rowCount()-1)
            del self.groups[-1]
        self.formGroupBox.adjustSize()
        self.adjustSize()

    def getInputs(self):
        classGroups = [group.text().split(",") for group in self.groups]
        if self.orderedCheck.isChecked():
          return (self.first.text(), self.second.text(), self.third.value(), classGroups, self.maxConsecutive.value())
        return (self.first.text(), self.second.text(), self.third.value(), classGroups)

class gpaDistributionInput(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.first = QLineEdit(self)
        self.first.setText('GPA Distribution')
        self.second = QLineEdit(self)
        self.second.setText('gpaHistogram')
        self.third = QSpinBox(self)
        self.third.setMinimum(0)
        self.third.setSingleStep(1)
        self.third.setValue(36)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Graph Title: ", self.first)
        layout.addRow("File name: ", self.second)
        layout.addRow("Minimum number of classes taken to count GPA: ", self.third)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return (self.first.text(), self.second.text(), self.third.value())

class pairGraphInput(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        if parent.grades.CLASS_CODE_COLUMN not in parent.grades.df.columns:
            parent.grades.getUniqueIdentifiersForSectionsAcrossTerms()
        self.classes = sorted(parent.grades.df[parent.grades.CLASS_CODE_COLUMN].unique().tolist())
        self.first = QComboBox(self)
        self.fillCombo(self.first, self.classes)
        self.second = QComboBox(self)
        self.fillCombo(self.second, self.classes)
        self.third = QLineEdit(self)
        self.third.setText('coursePairGraph')
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Class One: ", self.first)
        layout.addRow("Class Two: ", self.second)
        layout.addRow("File Name: ", self.third)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def fillCombo(self, combo, optionList):
        for option in optionList:
            combo.addItem(option)

    def getInputs(self):
        return (self.first.currentText(), self.second.currentText(), self.third.text())
    

class cliqueHistogramInput(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.first = preciseSpinBox(self)
        self.first.setMinimum(0.0)
        self.first.setMaximum(1.0)
        self.first.setSingleStep(0.05)
        self.first.setValue(0.5)
        self.second = QCheckBox(self)
        self.third = QCheckBox(self)
        self.third.setChecked(True)
        self.fourth = QLineEdit(self)
        self.fourth.setText('Class Correlation Cliques')
        self.fifth = QLineEdit(self)
        self.fifth.setText('cliqueHistogram')
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Minimum correlation (0.0 to 1.0): ", self.first)
        layout.addRow("Count duplicate sub-cliques: ", self.second)
        layout.addRow("Y-axis in log base 10 scale: ", self.third)
        layout.addRow("Graph Title: ", self.fourth)
        layout.addRow("File name: ", self.fifth)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return (self.first.value(), self.second.isChecked(), self.third.isChecked(), self.fourth.text(), self.fifth.text())

class preciseSpinBox(QDoubleSpinBox):
    def __init__(self, parent=None, dec = 15):
        super().__init__(parent)
        self.setFixedWidth(160)
        self.setDecimals(dec)
        self.setMaximum(999999999)
        self.setMinimum(-999999999)

    def textFromValue(self, val):
      return str(round(float(str(val)),self.decimals()))

class majorChordInput(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.first = preciseSpinBox(self)
        self.first.setMinimum(0.0)
        self.first.setMaximum(1.0)
        self.first.setSingleStep(0.05)
        self.first.setValue(0.5)
        self.second = preciseSpinBox(self)
        self.second.setMinimum(0.0)
        self.second.setMaximum(1.0)
        self.second.setSingleStep(0.05)
        self.second.setValue(0.05)
        self.third = QLineEdit(self)
        self.third.setText('majorGraph')
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Minimum correlation (0.0 to 1.0): ", self.first)
        layout.addRow("Maximum P-val (0.0 to 1.0): ", self.second)
        layout.addRow("File name: ", self.third)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return (self.first.value(), self.second.value(), self.third.text())

class substituteInput(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.first = QLineEdit(self)
        self.second = QLineEdit(self)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Find: ", self.first)
        layout.addRow("Replacement: ", self.second)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return (self.first.text(), self.second.text())

class TableWidgetCustom(QTableWidget):
    def __init__(self, parent=None):
        super(TableWidgetCustom, self).__init__(parent)

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Copy):
            self.copy()
        else:
            QTableWidget.keyPressEvent(self, event)

    def copy(self):
        selection = self.selectionModel()
        indexes = selection.selectedRows()
        if len(indexes) < 1:
            # No row selected
            return
        text = ''
        for idx in indexes:
            row = idx.row()
            for col in range(0, self.columnCount()):
                item = self.item(row, col)
                if item:
                    text += item.text()
                text += '\t'
            text += '\n'
        QApplication.clipboard().setText(text)

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data
    
    error
        `tuple` (exctype, value, traceback.format_exc() )
    
    result
        `object` data returned from processing, anything

    progress
        `int` indicating % progress 

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)

class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and 
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()    

        # Add the callback to our kwargs
        # self.kwargs['progress_callback'] = self.signals.progress        

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        
        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done

if __name__ == "__main__":
    import sys
    import os
    # put it **before** importing webbroser
    os.environ["BROWSER"] = "firefox"
    import webbrowser
    # BROWSER = 'firefox'
    edmlib.edmApplication = True
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName('EDM Program')
    stylesAvailable = PyQt5.QtWidgets.QStyleFactory.keys()
    # print(stylesAvailable)
    if 'Macintosh' in stylesAvailable:
      app.setStyle('Macintosh')
    elif 'Breeze' in stylesAvailable:
      app.setStyle('Breeze')
    # elif 'GTK+' in stylesAvailable:
    #   app.setStyle('Breeze')
    elif 'Fusion' in stylesAvailable:
      app.setStyle('Fusion')
    elif 'Windows' in stylesAvailable:
      app.setStyle('Windows')

    main = MyWindow()
    main.show()

    sys.exit(app.exec())
