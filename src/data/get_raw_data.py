
# %%
import pandas as pd
import requests
from tqdm import tqdm

# %%
base_url = 'https://fantasy.premierleague.com/api/'

def load_current_player_data(player_df):

    """Function to load player histories for current players"""
    df_list =  []

    print("Loading current player data")
    for idx,meta in tqdm(player_df.iterrows(),total=player_df.shape[0]):
        id =   meta['id']
        player= f"https://fantasy.premierleague.com/api/element-summary/{id}/"
        r = requests.get(player).json()
        if len(r)>0:
            df = pd.json_normalize(r['history_past'])
            df['id'] = id
            df['web_name'] = meta['web_name']
            df['position'] = meta['position']
            df_list.append(df)

    return pd.concat(df_list,ignore_index=True)


# %%
cm_season = pd.read_csv('https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/cleaned_merged_seasons.csv',low_memory=False)

# %%
seasons = ['2020-21','2019-20','2018-19']
player_season_list = []
for season in seasons[::-1]:
  loc = f'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/players_raw.csv'
  player_list = pd.read_csv(loc,usecols=['code','id','element_type','first_name','second_name','web_name','now_cost'])
  player_list['season'] = season
  player_season_list.append(player_list)

# %%
merged_seasons = pd.concat(player_season_list)
merged_seasons['joint_name'] = merged_seasons['first_name'] +'_'+merged_seasons['second_name']
merged_seasons = merged_seasons.groupby(['id','joint_name']).tail(1)
merged_seasons['position']=merged_seasons.element_type.map(pos_dict).values

# %%
if __name__ == '__main__':

    # Set base URL
    base_url = 'https://fantasy.premierleague.com/api/'

    # get data from bootstrap-static endpoint
    r = requests.get(base_url+'bootstrap-static/').json()

    #Get position dict mapping id to position name
    pos_dict = pd.json_normalize(r['element_types'])[['id','plural_name_short']]\
    .set_index('id')\
        .to_dict()['plural_name_short']

    # Get the player names
    draft_players = pd.json_normalize(r['elements'])\
        .loc[:,['code','id','element_type','first_name','second_name','web_name','now_cost']]
    draft_players['position'] = draft_players.element_type.map(pos_dict).values

    present_player_histories = load_current_player_data(draft_players)
# %%
