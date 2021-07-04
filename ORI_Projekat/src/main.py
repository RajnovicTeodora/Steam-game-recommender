import pandas as pandas
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

N = 20


def clean_string(text):
    return text.replace(" ", "").lower()


if __name__ == '__main__':
    user_df = pandas.read_csv("data/steam-200k.csv")
    game_df = pandas.read_csv("data/steam_games.csv")
    game_df = pandas.DataFrame(game_df)

    # Remove bundle and sub types
    game_df = game_df[game_df['types'] == 'app']

    # Remove negatively reviewed games
    game_df = game_df[game_df['all_reviews'].str.contains('Negative') == False]
    # game_df = game_df[game_df['all_reviews'].str.contains('Mixed') == False]

    # Set hours to 0.1 if action is purchase
    user_df.loc[(user_df['Action'] == 'purchase') & (user_df['Hours'] == 1.0), 'Hours'] = 0.1

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
        game_df.loc[:, feature] = game_df[feature].apply(clean_string)

    game_df["release_date_developer_publisher_popular_tags_game_details_genre"] = game_df['release_date'] +\
        game_df['developer'] + game_df['publisher'] + game_df['popular_tags'] + game_df['game_details'] +\
        game_df['genre']

    # Compute the Cosine Similarity matrix using the column

    # CountVectorizer converts strings to numerical vectors
    # stop_words - most frequent words that give no meaning
    count = CountVectorizer(stop_words='english')

    count_matrix = count.fit_transform(game_df['release_date_developer_publisher_popular_tags_game_details_genre'])
    cosine_similarity = cosine_similarity(count_matrix, count_matrix)

    # Unique users and games
    unique_users = user_df.sort_values(['UserID']).drop_duplicates(['UserID'], keep='first')
    unique_games = game_df['name'].unique()

    recommendations_df = pandas.DataFrame(columns=['UserID', 'Games', 'Playtime', 'Recommended Games', 'Predicted', 'MAE', 'RMSE', 'Accuracy'])

    # Make random test users
    random_users = unique_users.sample(150)

    for idx, row in random_users.iterrows():
        print(row['UserID'])
        user_games = list()
        recommendations = list()

        user_games_prediction = list()
        recommendation_predictions = list()

        user_games.append([game for game in user_df[user_df['UserID'] == row['UserID']]['Game'] if game in unique_games])
        if len(user_games[0]) <= 1:
            continue

        for game in user_games[0]:
            top_n_games = list()
            game_idx = game_df.index[game_df['name'] == game].tolist()[0]

            if game_idx >= len(cosine_similarity):
                continue

            # (game, score)
            similarity = list(enumerate(cosine_similarity[game_idx]))

            # Sort by scores
            similarity = sorted(similarity, key=lambda x: x[1], reverse=True)

            # Index of current game that we are predicting for
            user_games_prediction.append(user_df.loc[(user_df['UserID'] == row['UserID']) & (user_df['Game'] == game)]['Hours'].values[0])

            # Top N similar games
            similar_games = similarity[1:N + 1]

            # Make predictions
            for i in similar_games:

                denominator = 0.0
                num = 0.0
                # i - current game that we are calculating hours for

                # For each game that user has played
                for rated_game in user_games[0]:
                    sim_game_idx = game_df.index[game_df['name'] == rated_game].tolist()[0]

                    if sim_game_idx >= len(cosine_similarity) or rated_game == game_idx:
                        continue

                    # Sim of current game and previously rated game
                    sim = cosine_similarity[i[0]][sim_game_idx]

                    # Hours of game playtime
                    hours_played = user_df.loc[(user_df['UserID'] == row['UserID']) & (user_df['Game'] == rated_game)]['Hours'].values[0]

                    num += hours_played * sim
                    denominator += abs(sim)

                if denominator > 0.0:
                    result = num/denominator
                    top_n_games.append((i[0], result))
                else:
                    top_n_games.append((i[0], 0))

            # Pick the most similar game
            top_n_games = sorted(top_n_games, key=lambda x: x[1], reverse=True)

            for recommended_game in top_n_games:
                if recommended_game[0] not in recommendations:
                    recommendations.append(recommended_game[0])
                    recommendation_predictions.append(recommended_game[1])
                    break

        # If recommendations aren't found
        if len(recommendations) == 0 or len(user_games) == 0:
            continue

        mae = mean_absolute_error(user_games_prediction, recommendation_predictions)
        rmse = mean_squared_error(user_games_prediction, recommendation_predictions, squared=False)
        recommendations = game_df['name'].iloc[recommendations].tolist()
        accuracy = len(set(recommendations) & set(user_games[0])) / float(len(user_games[0]))

        user_games_prediction = [str(a) for a in user_games_prediction]
        recommendation_predictions = [str(a) for a in recommendation_predictions]
        user_games_print = ','.join(user_games[0])
        recommendation_print = ','.join(recommendations)
        playtime = ','.join(user_games_prediction)
        predicted = ','.join(recommendation_predictions)
        data = {'UserID': row['UserID'], 'Games': user_games_print,
                                                        'Playtime': playtime,
                                                        'Recommended Games': recommendation_print,
                                                        'Predicted': predicted,
                                                        'MAE': mae,
                                                        'RMSE': rmse,
                                                        'Accuracy': accuracy}
        recommendations_df = recommendations_df.append(data, ignore_index=True)
        print(data)

    recommendations_df.to_csv("data/recommendation.csv")
