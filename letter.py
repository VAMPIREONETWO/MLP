import pandas as pd
from MLP import MLP, Adam, Momentum, RMSProp

if __name__ == "__main__":
    # data
    df = pd.read_csv("letter+recognition\letter-recognition.data", header=None)
    y = df[0].to_numpy()
    X = df.drop([0], axis=1).to_numpy()

    # train
    num = int(len(X) / 5 * 4)
    # mlp = MLP(16, 26, [256, 128, 64], lr=0.001, optimizer=None)
    # mlp = MLP(16, 26, [256, 128, 64], lr=0.001, batch_size=100, optimizer=Momentum(mf=0.9))  # recommended mf is 0.9
    # mlp = MLP(16, 26, [256, 128, 64], lr=0.001, batch_size=100, optimizer=RMSProp(df = 0.999))  # recommended df is 0.999
    mlp = MLP(16, 26, [256, 128, 64], lr=0.001, optimizer=Adam(mf=0.9, df=0.999))
    mlp.fit(X[:num], y[:num], 1000, batch_size=100)
    mlp.write_train_data("letter train.txt")
    mlp.show_error_plot()

    # prediction
    pre = mlp.predict(X[num:])
    accuracy = 0
    for i in range(len(pre)):
        if pre[i] == y[num+i]:
            accuracy += 1
    accuracy = accuracy / len(pre)
    print("Accuracy: {}".format(accuracy))


