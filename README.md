# Trader with NASDAQ stock
### 1. Get the Data
By using panda library to parse data 
```python
dataset = pd.read_csv(args.training, names=["Open", "High", "Low", "Close"])
```
### 2. Get the MinMax to scale the data
X = current open price <br/>
y = future open price
```python
from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler()
    dataset_scaled = sc.fit_transform(dataset_scaled)

    X = dataset_scaled[0:n - 1]
    y = dataset_scaled[1:n]
```

### 3. Start Using RNN and add layers
Adding the input layer and the first hidden layer
```python
classifier.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1)))
```
Fitting the RNN to the Training set
```python
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=200)
```

### 4. Using loop to update RNN and predict next data
```python
train_test_last = train_test[-1]
        train_test_scaled = price_real / train_test_last
        price_pred = float(sc.inverse_transform(classifier.predict(np.reshape(train_test_scaled, (1, 1, 1)))))
```