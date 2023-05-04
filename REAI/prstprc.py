import os
import pandas as pd
import ast
# set the path to the top-level directory containing the multirun folders
top_dir = 'REAI/multirun/'

# define a lambda function to convert the rewards column to a nested list of integers
rewards_converter = lambda x: [[int(val) for val in row[1:-1].split()] for row in x.split('\n') if row]

# initialize an empty list to hold the dataframes
df_list = []
print(os.listdir())
# traverse the directory structure and load the rewards.csv files
for date_folder in os.listdir(top_dir):
    date_path = os.path.join(top_dir, date_folder)
    if not os.path.isdir(date_path):
        continue
    for time_folder in os.listdir(date_path):
        time_path = os.path.join(date_path, time_folder)
        if not os.path.isdir(time_path):
            continue
        for exp_folder in os.listdir(time_path):
            exp_path = os.path.join(time_path, exp_folder)
            if not os.path.isdir(exp_path):
                continue
            reward_file = os.path.join(exp_path, 'rewards.csv')
            if not os.path.isfile(reward_file):
                continue
            # load the rewards.csv file and add additional columns
            #converters = {'rewards': lambda x: ast.literal_eval(x) if isinstance(x, str) else None}
            try:
                df = pd.read_csv(reward_file)#, converters=converters)
            except Exception as e:
                print(f"Error loading {reward_file}: {e}")
                continue
            df['date'] = date_folder
            df['time'] = time_folder
            df['experiment'] = exp_folder
            df_list.append(df)

# concatenate the dataframes into one
result_df = pd.concat(df_list, ignore_index=True)