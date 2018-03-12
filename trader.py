import numpy as np
import pandas as pd

def model(X_train, y_train):
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM

    # Initialising the RNN
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1)))

    # Adding the output layer
    classifier.add(Dense(units=1))

    # Compiling the RNN
    classifier.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    # Fitting the RNN to the Training set
    classifier.fit(X_train, y_train, batch_size=10, nb_epoch=200)

    return classifier

def solve(classifier, train_test):
    # Initialize Stock Holding
    stock = 0
    money = 0
    sell = 0
    buy = 0
    output = []
    # Importing the dataset of testing
    dataset_real = pd.read_csv(args.testing, names=["Open", "High", "Low", "Close"])
    dataset_real = dataset_real.values

    for data in dataset_real[:-1]:
        price_real = data[0]
        train_test_last = train_test[-1]
        train_test_scaled = price_real / train_test_last
        price_pred = float(sc.inverse_transform(classifier.predict(np.reshape(train_test_scaled, (1, 1, 1)))))
        # print ("real: " +str(price_real)+" pred: "+str(price_pred))
        train_test = np.append(train_test, price_pred)
        if buy == 1:
            stock += 1
            money -= price_real
            buy = 0
            # print("buy: 1 " + str(money))
        if sell == 1:
            stock -= 1
            money += price_real
            sell = 0
            # print("sell: 1 " + str(money))
        if stock == 0:
            if train_test_last > price_pred:
                output = np.append(output, 0)

            if train_test_last <= price_pred:
                buy = 1
                output = np.append(output, 1)

        if stock == 1:
            if train_test_last > price_pred:
                sell = 1
                output = np.append(output, -1)

            if train_test_last <= price_pred:
                output = np.append(output, 0)
    else:
        if (stock == 1):
            price_real = dataset_real[-1][0]
            money += price_real
            stock = 0
    np.savetxt(args.output, output, delimiter=",", fmt='%1.f')


# You can write code above the if-main block.
if __name__ == '__main__':
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                        default='training_data.csv',
                        help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.

    # Importing the dataset
    dataset = pd.read_csv(args.training, names=["Open", "High", "Low", "Close"])

    # Dimensions of dataset
    n = dataset.shape[0]
    p = dataset.shape[1]

    dataset_scaled = dataset.iloc[:, 0:1].values

    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler()
    dataset_scaled = sc.fit_transform(dataset_scaled)

    X = dataset_scaled[0:n - 1]
    y = dataset_scaled[1:n]

    X_train = np.reshape(X, (n - 1, 1, 1))
    y_train = y

    classifier = model(X_train, y_train)

    # get trained data
    train_test = classifier.predict(X_train)
    train_test = sc.inverse_transform(train_test)

    solve(classifier, train_test)
