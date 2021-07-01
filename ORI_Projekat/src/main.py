import pandas as pandas
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import string
from sklearn.metrics import mean_squared_error

N = 10


def clean_string(text):
    text = ''.join([word for word in text if word not in string.punctuation])
    text = text.lower()

    return text


if __name__ == '__main__':
    user_df = pandas.read_csv("data/steam-200k.csv")
    game_df = pandas.read_csv("data/steam_games.csv")
    game_df = pandas.DataFrame(game_df)

    # Remove bundle and sub types
    game_df = game_df[game_df['types'] == 'app']

    # Remove negatively reviewed games
    game_df = game_df[game_df['all_reviews'].str.contains('Negative') == False]
    game_df = game_df[game_df['all_reviews'].str.contains('Mixed') == False]

    # Set hours to 0 if action is purchase
    user_df.loc[(user_df['Action'] == 'purchase') & (user_df['Hours'] == 1.0), 'Hours'] = 0

    # Remove purchased row if game has been played
    user_df = user_df.sort_values(['UserID', 'Game', 'Action']) \
        .drop_duplicates(['UserID', 'Game'], keep='first').drop(['Action', 'Empty'], axis=1)

    # Remove unnecessary columns and duplicate games
    game_df = game_df.iloc[:, 2:-5]
    game_df = game_df.drop_duplicates(subset="name", keep='first')\
        .drop(['achievements', 'desc_snippet', 'recent_reviews', 'all_reviews', 'achievements', 'languages'], axis=1)

    # Features for game comparison
    features = ['release_date', 'developer', 'publisher', 'popular_tags', 'game_details', 'genre']
    for feature in features:
        game_df.loc[:, feature] = game_df[feature].fillna('')
        #game_df.loc[:, feature] = game_df[feature].apply(clean_string)

    game_df["release_date_developer_publisher_popular_tags_game_details_genre"] = game_df['release_date'] +\
        game_df['developer'] + game_df['publisher'] + game_df['popular_tags'] + game_df['game_details'] +\
        game_df['genre']

    # Compute the Cosine Similarity matrix using the column

    # CountVectorizer converts strings to numerical vectors
    # stop_words - most frequent words that give no meaning, we remove them

    count = CountVectorizer(stop_words='english', lowercase = True, token_pattern = '[a-zA-Z0-9]+')

    count_matrix = count.fit_transform(game_df['release_date_developer_publisher_popular_tags_game_details_genre'])
    cosine_similarity = cosine_similarity(count_matrix, count_matrix)

    unique_users = user_df.sort_values(['UserID']).drop_duplicates(['UserID'], keep='first')
    recommendations_df = pandas.DataFrame(columns=['UserID', 'Games', 'Recommended Games', 'MSE', 'RMSE', 'Accuracy'])

    user_games = list()
    recommendations = list()

    nesto = list()

    user_games_idx = list()
    recommendation_game_idx = list()

    top_n_games = list()
    for idx, row in unique_users.iterrows():
        user_games.append([game for game in user_df[user_df['UserID'] ==  row['UserID']]['Game']])

        print("========================")
        print(row['UserID'])
        for game in user_games[0]:
            if game in game_df['name'].unique():
                game_idx = game_df.index[game_df['name'] == game].tolist()[0]

                if game_idx >= len(cosine_similarity):
                    continue

                # (game, score)
                similarity = list(enumerate(cosine_similarity[game_idx]))

                user_games_idx.append(game_idx)
                # Sort by scores
                similarity = sorted(similarity, key=lambda x: x[1], reverse=True)
                similarity = similarity[1:2 + 1]

                # Get the games indices
                #top_n_games = [i[0] for i in similarity]
                for i in similarity:
                    top_n_games.append(i)

        top_n_games = sorted(top_n_games, key=lambda x: x[1], reverse=True)
        for recommended_game in top_n_games:
            if recommended_game[0] not in recommendation_game_idx:
                recommendation_game_idx.append(recommended_game[0])

        # Return the top most similar games

        recommendation_game_idx = recommendation_game_idx[:len(user_games_idx)]
        if len(recommendation_game_idx) == 0 or len(user_games_idx) == 0:
            continue

        mse = mean_squared_error(user_games_idx, recommendation_game_idx)
        print(mse)

        rmse = mean_squared_error(user_games_idx, recommendation_game_idx, squared=False)
        print(rmse)
        accuracy = len(set(recommendations) & set(user_games[0])) / float(len(user_games[0]))
        print(accuracy)
        nesto.append(accuracy)
        ug = ','.join(user_games[0])
        gg = ','.join(game_df['name'].iloc[recommendation_game_idx].tolist())

        recommendations_df = recommendations_df.append({'UserID': row['UserID'], 'Games': ug,
                                                        'Recommended Games': gg,
                                                        'MSE': mse,
                                                        'RMSE': rmse,
                                                        'Accuracy': accuracy}, ignore_index=True)
        print(recommendations_df)
        user_games = list()
        recommendations = list()

        user_games_idx = list()
        recommendation_game_idx = list()
        break
    print(max(nesto))
    recommendations_df.to_csv("data/recommendation.csv")