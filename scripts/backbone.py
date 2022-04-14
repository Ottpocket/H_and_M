PROJECT_NAME =  #string
hyperparams = #dict
1/0 #delete this line once you have installed the kaggle secret code

#imports
import pandas as pd
import numpy as np


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

for i in range(3)
    #transformations DONE BY FUNCTIONS
    #get scores
    #save scores to WandB
    
#Get test values 
#save test score
