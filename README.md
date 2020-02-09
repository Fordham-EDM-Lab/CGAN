# EDMLib
A library made for educational data mining at Fordham University.

Library created by Michael Riad Zaky mriadzaky@fordham.edu (2020), with Daniel Leeds, Gary Weiss, Mavis Zhang at Fordham 
University. Library free for use provided you cite https://github.com/MichaelRZ/EDMLib in any resulting publications. Library free for redistribution provided you retain the author attributions above.

Usage:

The library takes in grade data in the form of a CSV file or Pandas dataframe and manipulates the data according to certain constant column names for student IDs, class IDs, and a student’s grade in that class. These constants can be changed as need be, and by default work with Fordham’s data.

Importing the library (while edmlib.py is in the same directory):
```
import edmlib
from edmlib import gradeData
from edmlib import classCorrelationData
```
Using the gradeData class:
```
Initialization
data = gradeData(‘filename.csv’)
# or data = gradeData(pandasDataFrame)
```
Filter to class data that has a gpa deviation larger than X (ex. 0.2):
```
data.filterByGpaDeviationMoreThan(X)
```
Filter to specific classes or majors:
```
data.reduceToMultipleMajorsOrClasses(majorsToFilterTo, classesToFilterTo)
# where 'majorsToFilterTo' is a list of majors matching the class major column and 'classesToFilterTo' 
# is a list of classes matching the class defining column
```

Export correlation file:
```
data.exportCorrelationsWithAvailableClasses('outputfile.csv')
# Has format of "course1", "course2", "corr" (correlation between 0-1), "P-value" (value between 0-1), 
# "#students" (number of students shared between these classes). 
```

Change constants to match your file:
```
data.defineWorkingColumns(finalGradeInClassColumn, classDefiningColumn, courseNumberAndTerm, studentIdColumn, classMajorColumn)
```

See the example file for simple usage.
