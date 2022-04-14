#https://www.kaggle.com/code/ottpocket/nn-starter-5
PROJECT_NAME =  #string
hyperparams = #dict
1/0 #delete this line once you have installed the kaggle secret code

#imports
import pandas as pd
import numpy as np
import gc


#Setting up WandB
import wandb
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("Api_Key")
!wandb login $secret_value_0

#See https://docs.wandb.com/library/init for more details
wandb.init(project= PROJECT_NAME, config=hyperparams)
config = wandb.config

#upload data
ss = pd.read_parquet('../input/hm-parquet/sample_submission')
articles = pd.read_parquet('../input/hm-parquet/articles')
train = pd.read_parquet('/kaggle/input/hm-parquet/transactions')
customers = pd.read_parquet('../input/hm-parquet/customers')
train['article_id'] = train.article_id.astype('int32')
train.t_dat = pd.to_datetime(train.t_dat)

#3 fold val dict with the test
val_dict = {'val_0': {'train': train.t_dat < pd.to_datetime('2020-09-16'),
                'val': train.t_dat >= pd.to_datetime('2020-09-16')},
            'val_1': {'train': train.t_dat < pd.to_datetime('2020-09-09'),
                'val': (train.t_dat >= pd.to_datetime('2020-09-09')) & (train.t_dat < pd.to_datetime('2020-09-16'))}, 
            'val_2': {'train': train.t_dat < pd.to_datetime('2020-09-02'),
                'val': (train.t_dat >= pd.to_datetime('2020-09-02')) & (train.t_dat < pd.to_datetime('2020-09-09'))}
            'test': {}}


#############################
#Val block
#############################
for key in val_dict.keys():
    print(f'Beginning {key}')
    if key != 'test':    
        train_df = train[val_dict[key]['train']].copy().reset_index(drop=True)
        val = train[val_dict[key]['val']].reset_index(drop=True)
        val = valid.rename({'article_id':'prediction'},axis=1)
        val['true'] =\
            val.prediction.apply(lambda x: ' '.join(['0'+str(k) for k in x]))
    else:
        train_df = train.copy()
        del train; gc.collect()
        
    #transformations DONE BY FUNCTIONS
    
    
    #get scores
    if key != 'test':
        score = mapk(
                        val['true'].map(lambda x: x.split()), 
                        train_df['prediction'].map(lambda x: x.split()), 
                        k=12
                    )
        wandb.run.summary[key] = score
    else:
        train_df.to_csv('sub.csv',index=False)
