from math import sqrt
def euc_dist(row1,row2):
    'calculates eucledian distance between two vectors'
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euc_dist(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

def predict_classification(train,test_row,num_neighbors):
    neighbors = get_neighbors(train,test_row,num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key= output_values.count)
    return prediction
def k_nearest_neighbors(train, test, num_neighbors):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    return(predictions) 
