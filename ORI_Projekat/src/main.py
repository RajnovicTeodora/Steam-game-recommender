import numpy as numpy
import pandas as pandas


if __name__ == '__main__':
    user_df = pandas.read_csv("data/steam-200k.csv")
    game_df = pandas.read_csv("data/steam_games.csv")

    # Remove bundle and sub types
    game_df = game_df[game_df['types'] == 'app']

    # Set hours to 0 if action is purchase
    user_df.loc[(user_df['Action'] == 'purchase') & (user_df['Hours'] == 1.0), 'Hours'] = 0

    # Remove purchased row if game has been played
    user_df = user_df.sort_values(['UserID', 'Game', 'Action']) \
        .drop_duplicates(['UserID', 'Game'], keep='first').drop(['Action', 'Empty'], axis=1)

    # Remove unnecessary columns and duplicate games
    game_df = game_df.iloc[:, 2:-5]
    print(len(game_df))
    game_df.drop_duplicates(subset="name", keep='first', inplace=True)

    # print(game_df.head(5))
    # print(user_df)
    print(len(game_df))
