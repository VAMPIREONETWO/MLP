from MLP import MLP
import numpy as np

if __name__ == "__main__":
    # data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    # construct MLP and train
    mlp = MLP(2,  1, [4],lr=1)
    mlp.fit(X, y, 1000, batch_size=4)
    mlp.write_train_data("XOR train.txt")
    mlp.show_error_plot()

    # predict
    pre = mlp.predict(X)
    print("Prediction:")
    print(pre)
    err = np.sum((pre-y.reshape(4,1))**2/2)
    print("Error: {}".format(err))

