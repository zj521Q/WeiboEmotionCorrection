import pandas as pd

df = pd.read_csv('weibo_predictions.csv', encoding='utf-8-sig')

user_sentiment_counts = df.groupby(['nickname', 'label']).size().unstack(fill_value=0)

negative_users = user_sentiment_counts[
    user_sentiment_counts[0] > (user_sentiment_counts[0] + user_sentiment_counts[1]) / 3
].index.tolist()

output_file = 'negative_users.txt'
with open(output_file, 'w', encoding='utf-8-sig') as f:
    for user in negative_users:
        f.write(user + '\n')
print(f"共有 {len(negative_users)} 个消极用户")
