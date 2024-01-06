import numpy as np
from MLP import MLP

if __name__ == "__main__":
    # data
    X = np.random.uniform(-1, 1, size=(500, 4))
    X_T = np.transpose(X)
    y = np.sin(X_T[0] - X_T[1] + X_T[2] - X_T[3])

    # train
    mlp = MLP(4,  1, [12],lr=0.01)
    mlp.fit(X[:400], y[:400], 300, batch_size=50)
    mlp.write_train_data("sin train.txt")
    mlp.show_error_plot()

    # predict
    pre = mlp.predict(X[400:])
    print("Prediction:")
    print(pre)
    err = np.sum((y[400:] - pre.reshape((1,pre.shape[0]))) ** 2 / 2)/100
    print("Error: {}".format(err))


