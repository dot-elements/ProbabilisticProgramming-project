import random

class BooleanDistribution:
    def __init__(self):
        self.typeName = "Boolean"


class CategoricalDistribution:
    def __init__(self, values, typeName):
        self.values = values
        self.typeName = typeName


class IntegerDistribution:
    def __init__(self):
        self.typeName = "Integer"


class RealDistribution:
    def __init__(self):
        self.typeName = "Real"


# **********************************************************************
# Data structures for creating PPL ASTs
# **********************************************************************

def isInteger(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def isFloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_all_true(predicate, iterable):
    return all(predicate(item) for item in iterable)

def removeColumns(datasetLs, indexes):
    sorted_indices = sorted(indexes, reverse=True)

    for row in datasetLs:
        # Ensure that the row length is sufficient for index removal
        if max(sorted_indices, default=-1) < len(row):
            for index in sorted_indices:
                del row[index]
        else:
            # Handling cases where row length is less than the index to be removed
            raise ValueError("Index out of range for row length.")

    return datasetLs


class Dataset:
    def __init__(self, filename, eps):

        self.indexCount = 0
        self.perturbation_coefficient = eps
        #if True: print("making dataset", filename)

        f = open(filename, "r")
        lines = f.readlines()

        # first let's filter out any constant columns.  no need to waste time modeling that
        datasetLs = []
        for line in lines:
            datasetLs.append(line.strip().split(","))

        colsToRemove = []
        for i in range(len(datasetLs[0])):
            firstVal = datasetLs[1][i]
            allSame = True
            for row in datasetLs[1:]:  # is there no andmap?
                if row[i] != firstVal:
                    allSame = False
                    break
            if allSame:
                colsToRemove.append(i)
        datasetLs = removeColumns(datasetLs, colsToRemove)

        lineItems = datasetLs[0]
        names = {}
        indexes = {}
        numItems = 0
        for i in range(len(lineItems)):
            lineItem = lineItems[i]
            if lineItem != "":
                names[i] = lineItem.replace("(", "").replace(")", "")
                indexes[names[i]] = i
                numItems = i

        numItems += 1

        self.numColumns = numItems

        self.indexesToNames = names
        self.namesToIndexes = indexes

        columns = []
        columnValues = []
        for i in range(numItems):
            columns.append([])
            columnValues.append(set())
        rows = datasetLs[1:]
        for cells in rows:
            for i in range(numItems):
                cell = cells[i]
                columns[i].append(cell)
                columnValues[i].add(cell)

        self.columns = columns
        self.numRows = len(rows)

        columnDistributionInformation = []
        colTypes = []
        columnNumericColumns = []
        columnMaxes = {}
        columnMins = {}
        for i, currColumnValues in enumerate(columnValues):
            if currColumnValues == {"true", "false"}:
                columnDistributionInformation.append(BooleanDistribution())
                colTypes.append("BOOL")
                ls = [self.float_to_int_rounding(self.perturbeValue(1)) if x == "true" else self.float_to_int_rounding(self.perturbeValue(0)) for x in self.columns[i]]
                columnNumericColumns.append([ls])
                for row in rows:
                    row[i] = 1 if row[i] == "true" else 0
            elif is_all_true(isInteger, currColumnValues):
                columnDistributionInformation.append(IntegerDistribution())
                colTypes.append("INT")
                self.columns[i] = [int(self.perturbeValue(x)) for x in self.columns[i]]
                columnMaxes[names[i]] = max(self.columns[i])
                columnMins[names[i]] = min(self.columns[i])
                columnNumericColumns.append([self.columns[i]])
                for row in rows:
                    row[i] = int(row[i])
            elif is_all_true(isFloat, currColumnValues):
                columnDistributionInformation.append(RealDistribution())
                colTypes.append("FLOAT")
                self.columns[i] = [float(self.perturbeValue(x)) for x in self.columns[i]]
                columnMaxes[names[i]] = max(self.columns[i])
                columnMins[names[i]] = min(self.columns[i])
                columnNumericColumns.append([self.columns[i]])
                for row in rows:
                    row[i] = float(row[i])
            else:
                columnDistributionInformation.append(CategoricalDistribution(list(currColumnValues), names[i] + "Type"))
                longestStrLength = max(len(x) for x in self.columns[i])
                colTypes.append(f"CHAR({longestStrLength})")
                lists = [[1 * (x == val) for x in self.columns[i]] for val in currColumnValues]
                columnNumericColumns.append(lists)
        self.colTypes = colTypes
        self.rows = rows

        self.columnDistributionInformation = columnDistributionInformation
        self.columnNumericColumns = columnNumericColumns

        self.columnMaxes = columnMaxes
        self.columnMins = columnMins

        # Call to perturb data

    def perturbeValue(self, val):
        return float(val) + random.uniform(-self.perturbation_coefficient, self.perturbation_coefficient)

    def float_to_int_rounding(self, val):
        if int(round(val)) >= 1:
            return 1
        else:
            return 0