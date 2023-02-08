import pandas as pd
from pprint import pprint
df = pd.read_csv("/work/kanakr/chat_persona/t5_grd_knw_grd_persona/predictions.csv")
df = df[['Generated Text', 'Actual Text']]
test_df = pd.read_csv("./data/focus_test_data.csv")
test_df['generated text'] = df['Generated Text']
test_df.to_csv("t5_generated.csv", index = False)

pprint(test_df.iloc[12])
# print(df.iloc[1])
# print(df.iloc[12])
