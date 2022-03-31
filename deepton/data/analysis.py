
# Find the min and max values for each column
def minmax(data):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*data)]
    return stats