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
parser.add_argument('--model_dir', type=str, default='models/run-20210720-0141-xgboost-scale', help='Directory where model files are stored')
parser.add_argument('--out_dir', type=str, default='output/ensemble_preds_converted_cm_compressed_uint16_xgb_scale', 
                    help='Directory with ensemble predictions in where we will replace scale values.')
args = parser.parse_args()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

test_df = pd.read_csv(os.path.join(args.data_dir, 'geopose_test.csv'))
meta_df = pd.read_csv(os.path.join(args.data_dir, 'metadata.csv'))
test_df['city'] = test_df['rgb'].apply(lambda x: x.split('_')[0])
test_df = pd.merge(test_df[['id', 'city']], meta_df[['id', 'gsd']], on='id', how='left')

le = LabelEncoder()
test_df['city'] = le.fit_transform(test_df['city'])

X_test = test_df[['city', 'gsd']].values

#------------------------------------------------------------------------------
# Predict
#------------------------------------------------------------------------------

test_preds_scale = []
for fold_id in range(5):
    model = XGBRegressor(learning_rate=0.8,
                         max_depth=5,
                         n_estimators=200,
                         n_jobs=-1,
                         random_state=33)
    model.load_model(os.path.join(args.model_dir, 'xgb_model_f%d.json' % fold_id))
    y_pred_test_scale = model.predict(X_test)
    test_preds_scale.append(y_pred_test_scale)

test_df['scale_xgb_ens'] = np.mean(test_preds_scale, axis=0)

#------------------------------------------------------------------------------
# Save predictions in .json
#------------------------------------------------------------------------------

files = sorted(glob.glob(os.path.join(args.out_dir, '*.json')))
print(len(files)) # 1025

for file in files:
    idd = file.split('/')[-1].split('_')[1]
    with open(file, 'r') as f:
        d = json.load(f)
    y_pred_angle = d['angle']
    y_pred_scale = float(test_df[test_df['id'] == idd]['scale_xgb_ens'].values[0])
    d = {'scale': y_pred_scale, 'angle': y_pred_angle}
    with open(file, 'w') as f:
        json.dump(d, f)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


