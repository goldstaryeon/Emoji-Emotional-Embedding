import json
import pymysql
import pandas as pd
from datasets import load_dataset
import emoji

#load raw data
dataset = load_dataset("go_emotions", "raw")
data = dataset['train'].to_pandas()
print(data['text'][:5], len(data))

data_with_emoji = data[data['text'].apply(lambda text: emoji.emoji_count(text) > 0)] 
print(data_with_emoji[:10], len(data_with_emoji))

col = ['emoji', 'context_emotion']
context_emotion = pd.DataFrame(columns=col)
ce_i = 0

for i in range(len(data_with_emoji)):
    row = data_with_emoji.iloc[i]
    text = row['text']
    emo_list = [item['emoji'] for item in emoji.emoji_list(text)]
    for col_name, value in row.items():
        if value == 1:
            emotion_name = col_name
    for e in emo_list:     
        context_emotion.loc[ce_i] = [e, emotion_name]
        ce_i += 1

print(context_emotion[:5], len(context_emotion))

context_emotion.to_csv('context_emotion.csv', index=False)