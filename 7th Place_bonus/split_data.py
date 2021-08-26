#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import os
import glob
import shutil
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
parser.add_argument('--fold_id', type=int, default=0, help='Fold id from 0 to 4')
args = parser.parse_args()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

train_df = pd.read_csv(os.path.join(args.data_dir, 'geopose_train.csv'))

# Create split stratified by city

train_df['city'] = train_df['agl'].apply(lambda x: x.split('_')[0])
# assert len(train_df['city'].unique()) == 4
train_df['fold_id'] = 0
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=33)
for fold_id, (train_index, val_index) in enumerate(kf.split(X=train_df, y=train_df['city'].values)):
        train_df.loc[train_df.index.isin(val_index), 'fold_id'] = fold_id
train_df = train_df.sample(frac=1.0, random_state=34)

# Create list of all validation files to move

val_df = train_df[train_df['fold_id'] == args.fold_id]
val_files = (list(val_df['agl'].values) + 
             list(val_df['json'].values) +
             list(val_df['rgb'].values))
# print(len(val_files))

# Create validation directory and move validation files

train_dir = os.path.join(args.data_dir, 'train_re_2')
val_dir = os.path.join(args.data_dir, 'val_f%d_re_2' % args.fold_id)

if not os.path.exists(val_dir):
    os.makedirs(val_dir)

for file in val_files:
    shutil.move(os.path.join(train_dir, file.replace('j2k', 'tif')), val_dir)

# Rename training directory to incorporate fold id
shutil.move(train_dir, os.path.join(args.data_dir, 'train_f%d_re_2' % args.fold_id))

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------