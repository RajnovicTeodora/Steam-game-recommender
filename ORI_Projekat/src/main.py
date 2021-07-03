import pandas as pandas
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def clean_string(text):
    return text.replace(" ", "")


if __name__ == '__main__':
    user_df = pandas.read_csv("data/steam-200k.csv")
    game_df = pandas.read_csv("data/steam_games.csv")
    game_df = pandas.DataFrame(game_df)

    # Remove bundle and sub types
    game_df = game_df[game_df['types'] == 'app']

    # Remove negatively reviewed games
    game_df = game_df[game_df['all_reviews'].str.contains('Negative') == False]
    game_df = game_df[game_df['all_reviews'].str.contains('Mixed') == False]

    # Set hours to 1 if action is purchase
    user_df.loc[(user_df['Action'] == 'purchase') & (user_df['Hours'] == 1.0), 'Hours'] = 1

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
    # stop_words - most frequent words that give no meaning, we remove them
    count = CountVectorizer(stop_words='english')

    count_matrix = count.fit_transform(game_df['release_date_developer_publisher_popular_tags_game_details_genre'])
    cosine_similarity = cosine_similarity(count_matrix, count_matrix)

    unique_users = user_df.sort_values(['UserID']).drop_duplicates(['UserID'], keep='first')
    recommendations_df = pandas.DataFrame(columns=['UserID', 'Games', 'Recommended Games', 'MSE', 'RMSE', 'Accuracy'])
    print(len(unique_users))
    temp = 0
    for idx, row in unique_users.iterrows():
        user_games = list()
        recommendations = list()
        user_games_idx = list()
        top_n_games = list()

        user_games.append([game for game in user_df[user_df['UserID'] == row['UserID']]['Game']])

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

                # Set of similar games
                similarity = similarity[1:2 + 1]


                # Sum of all similar games to current game
                denominator = sum([abs(x[1]) for x in similarity])
                num = 0.0
                for i in similarity:
                    temp = game_df['name'].iloc[i[0]].tolist()
                    num += i[1] * user_df[user_df['UserID'] == row['UserID'] and user_df['Game'] == temp]

                if denominator > 0.0:
                    result = num/denominator\
                    top_n_games.append((i[0], reusult)
        top_n_games = sorted(top_n_games, key=lambda x: x[1], reverse=True)
        for recommended_game in top_n_games:
            if recommended_game[0] not in recommendations:
                recommendations.append(recommended_game[0])

        # Return the top most similar games
        recommendations = recommendations[:len(user_games_idx)]

        # If recommendations aren't found
        if len(recommendations) == 0 or len(user_games_idx) == 0:
            continue

        mse = mean_squared_error(user_games_idx, recommendations)
        rmse = mean_squared_error(user_games_idx, recommendations, squared=False)
        recommendations = game_df['name'].iloc[recommendations].tolist()
        accuracy = len(set(recommendations) & set(user_games[0])) / float(len(user_games[0]))

        temp += 1


        user_games_print = ','.join(user_games[0])
        recommendation_print = ','.join(recommendations)

        recommendations_df = recommendations_df.append({'UserID': row['UserID'], 'Games': user_games_print,
                                                        'Recommended Games': recommendation_print,
                                                        'MSE': mse,
                                                        'RMSE': rmse,
                                                        'Accuracy': accuracy}, ignore_index=True)

    print(temp)
    recommendations_df.to_csv("data/recommendation.csv")
