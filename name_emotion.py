from transformers import pipeline
import pymysql
import pandas as pd

#connect to database
password = '730203'
password_encoded = password.encode()

connection = pymysql.connect(
    host='localhost',
    port=3306,
    db='emoji',
    user='root',
    password=password_encoded,
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

try:
    with connection.cursor() as cursor:
        sql = "SELECT * FROM emoji_15_1;"
        cursor.execute(sql)

        result = cursor.fetchall()
        emoji = pd.DataFrame(result)

finally:
    connection.close()

#huggingface emotion analysis model
classifier = pipeline(task="text-classification",
                      model="SamLowe/roberta-base-go_emotions",
                      top_k=None)

#name emotion analysis function
def name_emotion(emoji_name):
    outputs = classifier(emoji_name)
    return outputs[0][0]['label']

# admiration(0), amusement(1), anger(2), annoyance(3), approval(4),
# caring(5), confusion(6), curiosity(7), desire(8), disappointment(9),
# disapproval(10), disgust(11), embarrassment(12), excitement(13), fear(14),
# gratitude(15), grief(16), joy(17), love(18), nervousness(19),
# optimism(20), pride(21), realization(22), relief(23), remorse(24),
# sadness(25), surprise(26), neutral(27)

def name_emotion_vec(emotions):
    vec = [0]*28
    for i in range(len(emotions[0])):
        label = emotions[0][i]['label']
        score = emotions[0][i]['score']

        if label == 'admiration':
            vec[0] = score
        elif label == 'amusement':
            vec[1] = score
        elif label == 'anger':
            vec[2] = score
        elif label == 'annoyance':
            vec[3] = score
        elif label == 'approval':
            vec[4] = score
        elif label == 'caring':
            vec[5] = score
        elif label == 'confusion':
            vec[6] = score
        elif label == 'curiosity':
            vec[7] = score
        elif label == 'desire':
            vec[8] = score
        elif label == 'disappointment':
            vec[9] = score
        elif label == 'disapproval':
            vec[10] = score
        elif label == 'disgust':
            vec[11] = score
        elif label == 'embarrassment':
            vec[12] = score
        elif label == 'excitement':
            vec[13] = score
        elif label == 'fear':
            vec[14] = score
        elif label == 'gratitude':
            vec[15] = score
        elif label == 'grief':
            vec[16] = score
        elif label == 'joy':
            vec[17] = score
        elif label == 'love':
            vec[18] = score
        elif label == 'nervousness':
            vec[19] = score
        elif label == 'optimism':
            vec[20] = score
        elif label == 'pride':
            vec[21] = score
        elif label == 'realization':
            vec[22] = score
        elif label == 'relief':
            vec[23] = score
        elif label == 'remorse':
            vec[24] = score
        elif label == 'sadness':
            vec[25] = score
        elif label == 'surprise':
            vec[26] = score
        elif label == 'neutral':
            vec[27] = score
    return vec

#name emotion analysis
emoji['name_emotion'] = emoji['name'].apply(classifier)
emoji['name_emotion_vec'] = emoji['name_emotion'].apply(name_emotion_vec)
emoji['name_emotion'] = emoji['name'].apply(name_emotion)
emoji.to_csv('name_emotion.csv', index=False)