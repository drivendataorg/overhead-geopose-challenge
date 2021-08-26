#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import os
import glob
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
parser.add_argument('--out_dir', type=str, default='ckpt', help='Directory to save trained models')
args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

#------------------------------------------------------------------------------
# Create split stratified by city
#------------------------------------------------------------------------------

train_df = pd.read_csv(os.path.join(args.data_dir, 'geopose_train.csv'))
train_df['city'] = train_df['agl'].apply(lambda x: x.split('_')[0])
train_df['fold_id'] = 0

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=33)
for fold_id, (train_index, val_index) in enumerate(kf.split(X=train_df, y=train_df['city'].values)):
        train_df.loc[train_df.index.isin(val_index), 'fold_id'] = fold_id
train_df = train_df.sample(frac=1.0, random_state=34)

#------------------------------------------------------------------------------
# Collect training scale values (to use as target)
# Merge train and meta
#------------------------------------------------------------------------------

files_train = sorted(glob.glob(os.path.join(args.data_dir, 'train/*.json')))
# print(len(files_train)) # 5923

scales = []
ids = []
for file in files_train:
    idd = file.split('/')[-1].split('_')[1]
    city = file.split('/')[-1].split('_')[0]
    with open(file, 'r') as f:
        d = json.load(f)
    scale = d['scale']
    ids.append(idd)
    scales.append(scale)
    
scale_df = pd.DataFrame()
scale_df['id'] = ids
scale_df['scale'] = scales

meta_df = pd.read_csv(os.path.join(args.data_dir, 'metadata.csv'))
scale_df = pd.merge(scale_df, meta_df[['id', 'gsd']], on='id', how='left')
train_df = pd.merge(train_df[['id', 'city', 'fold_id']], scale_df, on='id', how='left')

# Label encode city
le = LabelEncoder()
train_df['city'] = le.fit_transform(train_df['city'])

#------------------------------------------------------------------------------
# Train
#------------------------------------------------------------------------------

scores_scale = []
for fold_id in range(5):
    tr_df = train_df[train_df['fold_id'] != fold_id].copy()
    val_df = train_df[train_df['fold_id'] == fold_id].copy()
    X_tr = tr_df[['city', 'gsd']].values
    y_tr_scale = tr_df['scale'].values
    X_val = val_df[['city', 'gsd']].values
    y_val_scale = val_df['scale'].values

    model = XGBRegressor(learning_rate=0.8,
                         max_depth=5,
                         n_estimators=200,
                         n_jobs=-1,
                         random_state=33)

    model = model.fit(X_tr, y_tr_scale)
    model.save_model(os.path.join(args.out_dir, 'xgb_model_f%d.json' % fold_id))
    y_pred_val_scale = model.predict(X_val)
    #
    score_scale = mean_squared_error(y_val_scale, y_pred_val_scale, squared=False)
    scores_scale.append(score_scale)
    print('fold %d: scale RMSE: %.8f' % (fold_id, score_scale))
print('--------')
print('MEAN scale RMSE:', np.mean(scores_scale)) # 0.00016579

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------



