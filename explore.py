import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
df = pd.read_csv("saas.csv")
df['Month_Invoiced'] = pd.to_datetime(df['Month_Invoiced'])
df = df.sort_values('Month_Invoiced').set_index('Month_Invoiced')
df_re = df.Amount.resample('W').sum().reset_index()
X = df_re.index
y = df_re.Amount
tss = TimeSeriesSplit(n_splits=5, max_train_size=None)
for train_index, test_index in tss.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    train, test = [train_index, [test_index]]
def split_data(df, train_prop=.66): 
    train_size = int(len(df) * train_prop)
    train, test = df[0:train_size].reset_index(), df[train_size:len(df)].reset_index()
    return train, test


def evaluate(target_var, train = train, test = test, output=True):
    mse = metrics.mean_squared_error(test[target_var], yhat[target_var])
    rmse = math.sqrt(mse)

    if output:
        print('MSE:  {}'.format(mse))
        print('RMSE: {}'.format(rmse))
    else:
        return mse, rmse

def plot_and_eval(target_vars, train = train, test = test, metric_fmt = '{:.2f}', linewidth = 4):
    if type(target_vars) is not list:
        target_vars = [target_vars]

    plt.figure(figsize=(16, 8))
    plt.plot(train[target_vars],label='Train')
    plt.plot(test[target_vars], label='Test')

    for var in target_vars:
        mse, rmse = evaluate(target_var = var, train = train, test = test, output=False)
        plt.plot(yhat[var], linewidth=linewidth)
        print(f'{var} -- MSE: {metric_fmt} RMSE: {metric_fmt}'.format(mse, rmse))

    plt.show()


train, test = split_data(df)
train = train.set_index('Month_Invoiced')
test = test.set_index('Month_Invoiced')
plt.figure(figsize=(12, 4))
plt.plot(train)
plt.plot(test)
print('Observations: %d' % (len(df)))
print('Training Observations: %d' % (len(train)))
print('Testing Observations: %d' % (len(test)))


yhat = pd.DataFrame(y_train)
yhat['y'] = int(y_train[-1:])
yhat.min() == yhat.max()
plot_and_eval(target_vars='y', train = train, test = test)
