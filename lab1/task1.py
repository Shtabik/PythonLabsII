import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('space_titanic.csv')
pd.set_option('display.min_rows', 10)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(df)
print(df.isnull().sum())
df_final = df.copy()
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in num_cols:
    df_final[col]= df_final[col].fillna(df_final[col].median())
categ_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
for col in categ_cols:
    df_final[col]=df_final[col].fillna(df_final[col].mode()[0])
df_final.drop(['Name', 'Cabin'], axis='columns', inplace=True)
scaler = MinMaxScaler()
df_final[num_cols] = scaler.fit_transform(df_final[num_cols])
df_final = pd.get_dummies(df_final, columns=['HomePlanet', 'Destination', 'CryoSleep', 'VIP'], drop_first=True)
train_df, test_df = train_test_split(df_final, test_size=0.3, random_state=42)
print(df_final)
print(df_final.isnull().sum())
df_final.to_csv("newspace_titanic.csv", index=False)
# train_df.to_csv("train_data.csv", index=False)
# test_df.to_csv("test_data.csv", index=False)