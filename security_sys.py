import tensorflow as tf
import csv as csv
import numpy as np
from numpy import genfromtxt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import random

def main():
    """['time_stamp', 'node', 'light', 'sound', 'temp', 'smoke', 'corrected_smoke', 'label']"""
    file = open('Forest_fire_data.csv')
    csv_reader = csv.reader(file)

    #my_data = genfromtxt('Forest_fire_data.csv', delimiter=',')
    #print(my_data)

    # Declared arrays
    csv_data = []
    node_info = []
    data = []
    #target = []

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    x_valid = []
    y_valid = []

    # Transfer information from csv
    for i in csv_reader:
        csv_data.append(i)

    #print(csv_data)

    csv_data.pop(0)

    train_size = int(0.7 * len(csv_data))
    test_size = int(0.1 * len(csv_data))
    
    # Adds information to arrays
    for i in range(len(csv_data)):
        if (i > 0):
            event = []
            node = []

            time_data = csv_data[i][0].split(":")
            time_sec = (int(time_data[0]) * 3600) + (int(time_data[1]) * 60) + int(time_data[2])

            match csv_data[i][7]:
                case 'normal_event':
                    event = [1, 0, 0, 0, 0, 0, 0]
                case 'fire_dying':
                    event = [0, 1, 0, 0, 0, 0, 0]
                case 'fire_increasing':
                    event = [0, 0, 1, 0, 0, 0, 0]
                case 'fie_increasing':
                    event = [0, 0, 1, 0, 0, 0, 0]
                case 'fire':
                    event = [0, 0, 0, 1, 0, 0, 0]
                case 'about_to_intense_fire':
                    event = [0, 0, 0, 0, 1, 0, 0]
                case 'malfunction':
                    event = [0, 0, 0, 0, 0, 1, 0]
                case 'sensor_about_to_die':
                    event = [0, 0, 0, 0, 0, 0, 1]

            match csv_data[i][1]:
                case '1':
                    node = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                case '2':
                    node = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                case '3':
                    node = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                case '4':
                    node = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                case '5':
                    node = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                case '6':
                    node = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                case '7':
                    node = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                case '8':
                    node = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
                case '9':
                    node = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                case '10':
                    node = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
                case '11':
                    node = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
                case '12':
                    node = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
                case '13':
                    node = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
                case '14':
                    node = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
                case '15':
                    node = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            
            data.append([csv_data[i][2], csv_data[i][3], csv_data[i][4], 
                        csv_data[i][5], csv_data[i][6], time_sec, 
                        event[0], event[1], event[2], event[3], event[4], event[5], event[6],
                        node[0], node[1], node[2], node[3], node[4], node[5], 
                        node[6], node[7], node[8], node[9], node[10], node[11], 
                        node[12], node[13], node[14], 0
                        ])
            
            node_info.append(time_sec)

    #max_n_min(data, 4)
    
    intruded_data, target = simulate_intrusion(data, [0, 1, 2, 3, 4])
    #print(target)
    

    #print(node_info)
    #print(len(intruded_data))
    #print(len(target))

    # Convert all data into integers and scale
    for i in range(len(intruded_data)):
        try:
            intruded_data[i][0] = int(intruded_data[i][0])/1015
        except:
            pass
        try:
            intruded_data[i][1] = int(intruded_data[i][1])/1006
        except:
            pass
        try:
            intruded_data[i][2] = int(intruded_data[i][2])/944
        except:
            pass
        try:
            intruded_data[i][3] = int(intruded_data[i][3])/1000
        except:
            pass
        try:
            intruded_data[i][4] = int(intruded_data[i][4])/1000
        except:
            pass
        intruded_data[i][5] = (int(intruded_data[i][5]) - 82241)/984
        #intruded_data[i][6] = (int(intruded_data[i][6]) - 1)/14
        #intruded_data[i][7] = int(intruded_data[i][7])/6

    #print(intruded_data)
    for i in intruded_data:
        for j in i:
            j = float(j)

    #random.shuffle(intruded_data)

    # min-max scaling for input features
    #scaler = MinMaxScaler()
    #intruded_data = scaler.fit_transform(intruded_data)

    # Split information into traning, testing, and validation data
    for i in range(len(intruded_data)):
        if (i <= train_size):
            x_train.append(intruded_data[i])
            y_train.append(target[i])
        elif (i > train_size and i <= (train_size + test_size)):
            x_test.append(intruded_data[i])
            y_test.append(target[i])
        else:
            # Rest of data is validation data
            x_valid.append(intruded_data[i])
            y_valid.append(target[i])

    #print(x_train)
    # Convert data sets into numpy arrays
    x_test = np.array([np.array(i) for i in x_test])
    x_train = np.array([np.array(i) for i in x_train])
    x_valid = np.array([np.array(i) for i in x_valid])
    y_test = np.array(y_test)
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)

    # Encoding data in one-hot format
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)
    y_valid = tf.keras.utils.to_categorical(y_valid, num_classes=2)
    #print(intruded_data)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, )),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation='sigmoid') ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['Accuracy']
    )

    model.evaluate(x_test, y_test, verbose=1)

    model.fit(
        x_train, y_train, verbose=1, epochs=50, batch_size=32, validation_data=(x_valid, y_valid)
    )

    y_pred = model.predict(x_test, verbose=1, batch_size=4)

    # Chooses the label value with the greatest prediction probability
    max_indices = np.argmax(y_pred, axis = 1)
    csv_data, cols = y_pred.shape
    for i in range(csv_data):
        for j in range(cols):
            if (j == max_indices[i]):
                y_pred[i, j] = 1.0
            else:
                y_pred[i, j] = 0.0

    y_test = to_multiclass_list(y_test)
    y_pred = to_multiclass_list(y_pred)

    print(
        f"Classification report for classifier {model}:\n"
        f"{metrics.classification_report(y_test, y_pred)}\n"
    )
                        
    # Compute and plot ROC curve and auc for each class
    # One vs All approach
    #fpr = dict()
    #tpr = dict()
    #auc_roc = dict()

    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    auc_roc = metrics.auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, label='class {}'.format(i))

    """ plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.3f)' % auc_roc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()"""
    print(str(i) + " AUC-ROC (Area Under the Receiver Operating Characteristic Curve): " + str(auc_roc))
    
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="best")
    plt.title("Receiver Operating Characteristic Curve")
    plt.show()


    # Compute and plot precision recall curve and auc for each class
    #precision = dict()
    #recall = dict()

    precision, recall, _ = metrics.precision_recall_curve(y_test, y_pred)
    plt.plot(recall, precision, lw=2, label='class {}'.format(i))

    auc_precision_recall = metrics.auc(recall, precision)
    print(str(i) + " AUC-PR (Area Under the Precision-Recall Curve): " + str(auc_precision_recall))

    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("Precision-Recall Curve")
    plt.show()


    # Compute and print confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    print(cm)
    #print(y_pred)

    # Compute confusion matrix for each class and print tn, fp, fn, and tp
    cmi = metrics.confusion_matrix(y_test, y_pred).ravel()
    print(cmi.ravel())
    print(str(i) + " True negatives: " + str(cmi[0]))
    print(str(i) + " False positives: " + str(cmi[1]))
    print(str(i) + " False negatives: " + str(cmi[2]))
    print(str(i) + " True positives: " + str(cmi[3]))

    file.close()

def simulate_intrusion(data, intrusion_col_arr):
    old = data
    new =[]
    intrusion_y = []
    res = []

    random.shuffle(old)
    for j in intrusion_col_arr:
        start = random.randrange(14774)
        length = random.randrange(2000, 2500)
        node_num = old[start][6]

        for i in range(len(data)):
            if (i >= start and i <= (start + length)):
                rand_val = random.randrange(1000)

                old[i][j] = rand_val
                old[i][28] = 1
                new.append(old[i])
            elif (old[i][28] == 1):
                old[i][28] = 1
            else:
                old[i][28] = 0
                new.append(old[i])

    # Interpolate empty data
    list_data_interp(new, 0)
    list_data_interp(new, 1)
    list_data_interp(new, 2)
    list_data_interp(new, 3)
    list_data_interp(new, 4)
        
    random.shuffle(new)
    #print(new)

    for i in range(len(new)):
        #print(new)
        intrusion_y.append(new[i][28])
        #print(new[i][27])
        res.append([new[i][0], new[i][1], new[i][2], new[i][3], new[i][4], new[i][5], new[i][6], new[i][7], new[i][8], new[i][9], 
        new[i][10], new[i][11], new[i][12], new[i][13], new[i][14], new[i][15], new[i][16], new[i][17], new[i][18], new[i][19], new[i][20], 
        new[i][21], new[i][22], new[i][23], new[i][24], new[i][25], new[i][26], new[i][27]])
        
    #print(intrusion_y)
    return res, intrusion_y

"""
Definition: Converts an input multilabel format numpy array into a multiclass 2d list
@param input: The numpy array to be converted
return: A new multiclass list based on the input array
"""
def to_multiclass_list(input):
    input_list = input.tolist()
    new_list = []
    for i in range(len(input_list)):
        for j in range(len(input_list[i])):
            if (input_list[i][j] == 1.0):
                new_list.append(j)
                break
    return new_list

"""
Definition: Linearly interpolates and inserts new data into the empty data of a dataset
@param input_list: The input dataset with empty data entries in need of interpolation
@param column: The colum of the dataset with data in need of interpolation
"""
def list_data_interp(input_list, column):
    gap_len = 0
    gap_found = False
    start = 500
    
    for i in range(len(input_list)):
        if (input_list[i][column] == ''):
            gap_found = True
            gap_len += 1
        if (gap_found and input_list[i][column] != ''):
            
            try:
                start = float(input_list[i - gap_len - 1][column])
            except:
                start = 500

            end = float(input_list[i][column])
            difference = end - start

            for j in range(gap_len):
                input_list[i - gap_len + j][column] = float(start + ((difference / (gap_len + 1)) * (j + 1)))

            gap_found = False
            gap_len = 0

def max_n_min(data, column):
    arr = []
    for i in range(len(data)):
        #if (isinstance(data[i][column], int)):
        try:    
            arr.append(int(data[i][column]))
        except:
            pass

    print("min: " + str(min(arr)))
    print("max: " + str(max(arr)))

if __name__ == "__main__":
    main()