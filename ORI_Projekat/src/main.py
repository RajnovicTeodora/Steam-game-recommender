
import pandas as pandas
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import string

RECOMMENDATION_NUMBER = 10


def clean_string(text):
    text = ''.join([word for word in text if word not in string.punctuation])
    text = text.lower()

    return text


def get_recommendations(title, cosine_sim):

    if title not in game_df:
        return []

    # Get the index of the game that matches the name
    idx = game_df[title]

    # Get the pairwise similarity scores of all games with that game
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the games based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the most similar games
    # (not the first one because this games as a score of 1 (perfect score) similarity with itself)
    sim_scores = sim_scores[1:RECOMMENDATION_NUMBER + 1]

    # Get the games indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top most similar games
    return game_df['name'].iloc[movie_indices].tolist()


if __name__ == '__main__':
    
    user_df = pandas.read_csv("data/steam-200k.csv")
    game_df = pandas.read_csv("data/steam_games.csv")
    game_df = pandas.DataFrame(game_df)

    # Remove bundle and sub types
    game_df = game_df[game_df['types'] == 'app']

    # Remove negatively reviewed games
    game_df = game_df[game_df['all_reviews'].str.contains('Negative') == False]

    # Set hours to 0 if action is purchase
    user_df.loc[(user_df['Action'] == 'purchase') & (user_df['Hours'] == 1.0), 'Hours'] = 0

    # Remove purchased row if game has been played
    user_df = user_df.sort_values(['UserID', 'Game', 'Action']) \
        .drop_duplicates(['UserID', 'Game'], keep='first').drop(['Action', 'Empty'], axis=1)

    # Remove unnecessary columns and duplicate games
    game_df = game_df.iloc[:, 2:-5]

    game_df = game_df.drop_duplicates(subset="name", keep='first')\
        .drop(['achievements', 'desc_snippet', 'recent_reviews', 'achievements', 'languages'], axis=1)

    # relevant info for recommendation: genre game_details popular_tags publisher developer
    game_df.loc[:, 'genre'] = game_df['genre'].fillna('')
    game_df.loc[:, 'game_details'] = game_df['game_details'].fillna('')
    game_df.loc[:, 'popular_tags'] = game_df['popular_tags'].fillna('')
    game_df.loc[:, 'publisher'] = game_df['publisher'].fillna('')
    game_df.loc[:, 'developer'] = game_df['developer'].fillna('')

    game_df.loc[:, 'genre'] = game_df['genre'].apply(clean_string)
    game_df.loc[:, 'game_details'] = game_df['game_details'].apply(clean_string)
    game_df.loc[:, 'popular_tags'] = game_df['popular_tags'].apply(clean_string)
    game_df.loc[:, 'publisher'] = game_df['publisher'].apply(clean_string)
    game_df.loc[:, 'developer'] = game_df['developer'].apply(clean_string)

    game_df["genre_publisher_popular_tags_developer_game_details"] = game_df['genre'] + game_df['publisher'] + game_df[
        'popular_tags'] + game_df['developer'] + game_df['game_details']

    # Compute the Cosine Similarity matrix using the column

    # CountVectorizer converts strings to numerical vectors
    # stop_words - most frequent words that give no meaning, we remove them

    count = CountVectorizer(stop_words='english')

    count_matrix = count.fit_transform(game_df['genre_publisher_popular_tags_developer_game_details'])
    cosine_sim_matrix = cosine_similarity(count_matrix, count_matrix)

    #print(cosine_sim_matrix)

    #print(len(user_df))

    unique_users = user_df.sort_values(['UserID']).drop_duplicates(['UserID'], keep='first')
    recommendations = pandas.DataFrame(columns=['UserID', 'Games'])
    user_games = list()
    for idx, row in unique_users.iterrows():
        user_games.append([game for game in user_df[user_df['UserID'] == row['UserID']]['Game']])

        # games_for_recommendation = game_df.loc[~game_df['name'].isin(user_games)]
        # games_for_recommendation = games_for_recommendation.sort_values(by="all_reviews", ascending=False)
        print(user_games[0])
        for game in user_games[0]:

            if game in game_df['name'].unique():
                game_idx = game_df.index[game_df['name'] == game].tolist()[0]
