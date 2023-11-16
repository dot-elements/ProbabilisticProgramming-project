from itertools import combinations
from ND import *
from scipy.stats import spearmanr
import networkx as nx
from dataset import Dataset
import matplotlib.pyplot as plt


def draw_graph(graph):
    # Draw the graph
    pos = nx.spring_layout(graph)  # positions for all nodes
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_labels(graph, pos)

    # Show the plot
    plt.show()
def print_adjacency_list(graph):
    for line in nx.generate_adjlist(graph):
        print(line)
class Graph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def addNode(self, node):
        self.graph.add_node(node.name, data=node)

    def getNode(self, name, distribInfo):
        if self.graph.has_node(name):
            return self.graph.nodes[name]['data']
        else:
            newNode = Node(name, distribInfo)
            self.addNode(newNode)
            return newNode

    def addEdge(self, parent, child):
        self.graph.add_edge(parent.name, child.name)

    def getNodesInDependencyOrder(self):
        return list(nx.topological_sort(self.graph))

    def isDescendedFrom(self, name1, info1, name2, info2):
        n1 = self.getNode(name1, info1)
        n2 = self.getNode(name2, info2)
        return nx.has_path(self.graph, n1.name, n2.name)

class Node:
    def __init__(self, name, distribInfo):
        self.name = name
        self.parents = []
        self.children = []
        self.distribInfo = distribInfo
    def __init__(self, name, distribInfo):
        self.name = name
        self.distribInfo = distribInfo

    def __repr__(self):
        return f"Node({self.name})"

# **********************************************************************
# Helper functions
# **********************************************************************

def makeProgram(scriptStrings, holes, numHoles):
    outputString = scriptStrings[0]
    for i in range(0, numHoles):
        string = scriptStrings[i + 1]
        holeFiller = holes[i]
        outputString += str(holeFiller)
        outputString += string
    return outputString


# **********************************************************************
# Evaluate generated programs' proximity to spec
# **********************************************************************

def summarizeDataset(fileName):
    f = open(fileName, "r")
    lines = f.readlines()

    line = lines[0].strip()
    lineItems = line.split(",")
    sums = {}
    names = {}
    numItems = 0
    for i in range(len(lineItems)):
        lineItem = lineItems[i]
        if lineItem != "":
            sums[i] = 0
            names[i] = lineItem
            numItems = i
    numItems += 1

    for line in lines[1:]:
        entries = line.strip().split(",")
        for i in range(numItems):
            entry = entries[i]
            if entry == "true":
                entry = 1
            else:
                entry = 0
            sums[i] = sums[i] + entry

    numLines = len(lines) - 1
    means = {}
    for i in range(numItems):
        means[names[i]] = float(sums[i]) / numLines
    return means


def distance(summary1, summary2):
    res = 0
    for key in summary1:
        v1 = summary1[key]
        v2 = summary2[key]
        res += abs(v1 - v2)
    return res

debug = True


def correlationHelper(dataset, i, j):
    iCols = dataset.columnNumericColumns[i]
    jCols = dataset.columnNumericColumns[j]
    correlations = []
    correlations_2 = []
    for iCol in iCols:
        for jCol in jCols:
            res = spearmanr(iCol, jCol)
            # res2 = pearsonr(iCol, jCol)
            correlations.append(res)
        # correlations_2.append(res2[0])
    correlation1 = max(correlations, key=lambda item: item[1])
    correlation2 = min(correlations, key=lambda item: item[1])
    correlation = correlation1
    if abs(correlation2[0]) > abs(correlation1[0]):
        correlation = correlation2

    # correlation1_2 = max(correlations_2)
    # correlation2_2 = min(correlations_2)
    # correlation_2 = correlation1_2
    # if abs(correlation2_2) > abs(correlation1_2):
    #	correlation_2 = correlation2_2
    return correlation


def generateStructureFromDatasetNetworkDeconvolution(dataset, connectionThreshold):
    correlationsMatrix = [[0 for i in range(dataset.numColumns)] for j in range(dataset.numColumns)]
    for i in range(dataset.numColumns):
        for j in range(i + 1, dataset.numColumns):
            correlation = correlationHelper(dataset, i, j)[0]

            # correlation = pearsonr(dataset.columns[i], dataset.columns[j])
            correlationsMatrix[i][j] = correlation
            correlationsMatrix[j][i] = correlation


    a = np.array(correlationsMatrix)
    x = ND(a)

    g = Graph()
    for i in range(dataset.numColumns):
        name1 = dataset.indexesToNames[i]
        a1 = g.getNode(name1, dataset.columnDistributionInformation[i])
        for j in range(i + 1, dataset.numColumns):
            if x[i][j] > connectionThreshold:
                name2 = dataset.indexesToNames[j]
                a2 = g.getNode(name2, dataset.columnDistributionInformation[j])
                g.addEdge(a1, a2)
    return g


def generateReducibleStructuresFromDataset(dataset):
    g = Graph()
    for i in range(dataset.numColumns):
        name1 = dataset.indexesToNames[i]
        a1 = g.getNode(name1, dataset.columnDistributionInformation[i])
        for j in range(i + 1, dataset.numColumns):
            name2 = dataset.indexesToNames[j]
            a2 = g.getNode(name2, dataset.columnDistributionInformation[j])
            g.addEdge(a1, a2)

    return g


statisticalSignificanceThreshold = 0.05
correlationThreshold = 0.01


def generatePotentialStructuresFromDataset(dataset):
    global statisticalSignificanceThreshold, correlationThreshold
    columns = range(dataset.numColumns)
    combos = combinations(columns, 2)
    correlations = []
    for combo in combos:
        correlationPair = correlationHelper(dataset, combo[0], combo[1])
        # correlationPair = pearsonr(dataset.columns[combo[0]], dataset.columns[combo[1]])
        correlations.append((combo, correlationPair))
    sortedCorrelations = sorted(correlations, key=lambda x: abs(x[1][0]), reverse=True)
    # print sortedCorrelations

    g = Graph()
    # make sure we add all nodes
    for i in range(dataset.numColumns):
        name1 = dataset.indexesToNames[i]
        a1 = g.getNode(name1, dataset.columnDistributionInformation[i])
    # now add relationships
    for correlation in sortedCorrelations:
        i = correlation[0][0]
        j = correlation[0][1]
        name1 = dataset.indexesToNames[i]
        name2 = dataset.indexesToNames[j]
        statisticalSignificance = correlation[1][1]
        correlationAmount = abs(correlation[1][0])
        # print name1, name2, correlationAmount, statisticalSignificance
        if statisticalSignificance > statisticalSignificanceThreshold:
            # print "non sig:", statisticalSignificance
            continue
        if correlationAmount < correlationThreshold:
            # print "not cor:", correlationAmount
            break
        if not g.isDescendedFrom(name1, dataset.columnDistributionInformation[i], name2,
                                 dataset.columnDistributionInformation[j]):
            # we don't yet have an explanation from the connection between these two.  add one.
            a1 = g.getNode(name1, dataset.columnDistributionInformation[i])
            a2 = g.getNode(name2, dataset.columnDistributionInformation[j])
            # for now we'll assume the causation goes from left to right in input dataset
            g.addEdge(a1, a2)

    return g


def jaccard_index(graph1, graph2):
    edges_intersection = set(graph1.edges()).intersection(set(graph2.edges()))
    edges_union = set(graph1.edges()).union(set(graph2.edges()))
    return len(edges_intersection) / len(edges_union) if edges_union else 1.0

def hamming_distance(graph1, graph2):
    if set(graph1.nodes()) != set(graph2.nodes()):
        raise ValueError("Graphs must have the same node set for Hamming distance")

    graph1_edges = set(graph1.edges())
    graph2_edges = set(graph2.edges())

    return len(graph1_edges.symmetric_difference(graph2_edges))

def plot_findings_single(inputFile, generateGraph=generatePotentialStructuresFromDataset, metric=jaccard_index):
    original_dataset = Dataset(inputFile, 0)  # No perturbation
    original_graph = generateGraph(original_dataset).graph

    perturbation_coefficients = [i/10 for i in range(11)]  # 0, 0.1, 0.2, ..., 1.0
    jaccard_indices = []

    for eps in perturbation_coefficients:
        print(eps)
        perturbed_dataset = Dataset(inputFile, eps)
        perturbed_graph = generateGraph(perturbed_dataset).graph
        jaccard_indices.append(metric(original_graph, perturbed_graph))

    plt.plot(perturbation_coefficients, jaccard_indices)
    plt.xlabel('Perturbation Coefficient')
    plt.ylabel('Jaccard Index')
    plt.title('Robustness of Structure Generation Strategy')
    plt.show()

def plot_findings(inputFile, strategies, metric=jaccard_index):

    perturbation_coefficients = [i / 10 for i in range(11)]  # 0, 0.1, 0.2, ..., 1.0
    original_dataset = Dataset(inputFile, 0)  # No perturbation
    for generateGraph, strategy_name in strategies:
        print(f"Processing strategy: {strategy_name}")
        if generateGraph == generateStructureFromDatasetNetworkDeconvolution:
            original_graph = generateGraph(original_dataset, .1).graph
        else:
            original_graph = generateGraph(original_dataset).graph
        perturbed_datasets = {eps: Dataset(inputFile, eps) for eps in perturbation_coefficients}
        metrics = []
        for eps in perturbation_coefficients:
            print(f"Perturbation Coefficient: {eps}")
            perturbed_dataset = perturbed_datasets[eps]
            if generateGraph == generateStructureFromDatasetNetworkDeconvolution:
                perturbed_graph = generateGraph(perturbed_dataset, .1).graph
            else:
                perturbed_graph = generateGraph(perturbed_dataset).graph
            metrics.append(metric(original_graph, perturbed_graph))

        plt.plot(perturbation_coefficients, metrics, label=strategy_name)

    plt.xlabel('Perturbation Coefficient')
    plt.ylabel('Hamming Distance')
    plt.title('Robustness of Structure Generation Strategies')
    plt.legend()
    plt.show()
def main():
    inputFile = "airlineDelayDataProcessed.csv"
    strategies = [
        (generateReducibleStructuresFromDataset, "Complete"),
        (generateStructureFromDatasetNetworkDeconvolution, "ND"),
        (generatePotentialStructuresFromDataset, "Correlation")
    ]
    plot_findings(inputFile, strategies, hamming_distance)

if __name__ == "__main__":
    main()