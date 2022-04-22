import pandas as pd

def add_most_popular_items(preds, df, prev_days=7)
    '''
    Adds the most popular 12 items over the last `prev_days` days
    to your predictions
    '''
    pass

def customer_purchases(df, prev_days=7)
    '''
    Takes in the train data and returns a dataframe listing the previous purchases 
    over the last `prev_days` days for each.  
    
    INPUTS
    ---------
    df: (pd.DataFrame) upload from hm-parquet/transactions
    
    OUTPUTS
    ----------
    purchases: (pd.DataFrame) dataframe containing customer_name and predictions columns.
                            Predictions column is as a string.
    '''
    weeks = 1
    temp_date = df.t_dat.max() - pd.Timedelta(days=prev_days)
    temp = df.loc[df.t_dat > temp_date, ['t_dat', 'customer_id', 'article_id']].reset_index(drop=True).copy()
    counts = temp.groupby(['customer_id','article_id']).agg({'t_dat':'count'}).reset_index().rename(columns={'t_dat':'count'})
    temp['count'] = temp.merge(counts, on=['customer_id','article_id'], how='left')['count']
    temp.drop_duplicates(subset=['customer_id','article_id'], keep='first', inplace=True)
    temp.sort_values(['count','t_dat'],ascending=False, inplace=True)
    temp['article_id'] = ' 0' + temp['article_id'].astype(str)
    return temp.groupby('customer_id').agg({'article_id':'sum'}).reset_index().rename(columns={'article_id':'prediction'})
