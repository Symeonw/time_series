import vega_datasets
import pandas as pd
import seaborn as sns
df = vega_datasets.data.sf_temps()
df.date = pd.to_datetime(df.date)
df = df.set_index("date")
df_2 = df.resample("D").mean()
x = df_2.index
y = df_2.temp
lineplot = sns.lineplot(x,y)
df_3 = df.resample("D").min()
x = df_3.index
y = df_3.temp
lineplot = sns.lineplot(x,y)
df_4 = df.resample("D").max()
x = df_4.index
y = df_4.temp
lineplot = sns.lineplot(x,y)
cold_df = df.resample("M").mean()
cold_num = df.resample("M").mean().min()
cold_num = cold_num.iloc[0]
coldest_month = cold_df[cold_df["temp"] <= cold_num]
hot_df = df.resample("M").mean()
hot_num = df.resample("M").mean().max()
hot_num = hot_num.iloc[0]
hottest_month = hot_df[hot_df["temp"] >= hot_num]
df_day = df.resample("D").agg(['min', 'max'])
big_diff = df_day.temp["max"] - df_day.temp["min"]
pd.DataFrame(big_diff)
