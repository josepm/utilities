import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar as ms


def hbat_process(a_dir):
    hbat = pd.read_csv(a_dir, names=['id_agent', 'ts_session_start', 'id_ticket', 'seconds_in_ticket', 'ds'],
                       dtype={'id_agent': str, 'ts_session_start': str, 'seconds_in_ticket': np.float, 'ds': str})

    print(a_dir + ' read')
    hbat.drop('ds', axis=1, inplace=True)
    hbat['ts_session_start'] = pd.to_datetime(hbat['ts_session_start'].values)
    hbat['id_ticket'] = hbat['id_ticket'].apply(lambda x: 'NULL' if (pd.isna(x) or x == '' or x == ' ') else str(int(x)))
    print(a_dir + ' processed')
    print(hbat.dtypes)
    print(hbat['id_ticket'].value_counts().head(10))
    print(hbat.head())

    # hbat summary
    r = len(hbat)
    a = hbat['id_agent'].nunique()
    tx = hbat['id_ticket'].nunique()
    t = hbat['seconds_in_ticket'].sum()
    null_hbat = hbat[hbat['id_ticket'] == 'NULL']
    null_r = len(null_hbat)
    null_t = null_hbat['seconds_in_ticket'].sum()

    print('dir: ' + str(a_dir))
    print('rows: ' + str(r))
    print('agents: ' + str(a))
    print('tickets: ' + str(tx))
    print('null rows: ' + str(null_r))
    print('null time: ' + str(null_t))
    print('Fraction of rows with NULL id_ticket: ' + str(100 * np.round(null_r / r, 2)) + '% ' +
          'Fraction of time on NULL id_ticket: ' + str(100 * np.round(null_t / t, 2)) + '%')
    return hbat


def hbat_group(hbat_df, idx):
    hbat_df.to_parquet('~/my_tmp/hbat' + str(idx) + '.par')
    hbat_df['cnt'] = 1
    gbat = hbat_df.groupby(['id_agent', 'id_ticket']).agg({'seconds_in_ticket': np.sum, 'ts_session_start': {'ts_start': np.min, 'ts_end': np.max}, 'cnt': np.sum}).reset_index()
    gbat.columns = ['id_agent', 'id_ticket', 'seconds_in_ticket', 'ts_start', 'ts_end', 'hbat_cnt']
    print(gbat.head())
    print('saving gbat' + str(idx) + '. rows: ' + str(len(gbat)))
    gbat.to_parquet('~/my_tmp/gbat' + str(idx) + '.par')
    return gbat


if __name__ == '__main__':

    print('starting hbat1')
    hbat1 = hbat_process('~/hbat1.txt.gz')
    print('starting gbat1')
    gbat1 = hbat_group(hbat1, 1)

    print('starting hbat2')
    hbat2 = hbat_process('~/hbat2.txt.gz')
    print('starting gbat2')
    gbat2 = hbat_group(hbat2, 2)

    gbat = gbat1.merge(gbat2, on=['id_agent', 'id_ticket'], how='inner', suffixes=('_1', '_2'))
    gbat['secs:hbat2/hbat1'] = gbat['seconds_in_ticket_2'] / gbat['seconds_in_ticket_1']
    gbat['cnt:hbat2/hbat1'] = gbat['hbat_cnt_2'] / gbat['hbat_cnt_1']
    gbat['secs:ts_start_2 - ts_start_1'] = (gbat['ts_start_2'] - gbat['ts_start_1']).dt.seconds
    gbat['secs:ts_end_2 - ts_end_1'] = (gbat['ts_end_2'] - gbat['ts_end_1']).dt.seconds
    print('saving gbat')
    gbat.to_parquet('~/my_tmp/gbat.par')
    dz = gbat[['secs:hbat2/hbat1', 'cnt:hbat2/hbat1', 'secs:ts_start_2 - ts_start_1', 'secs:ts_end_2 - ts_end_1']].describe()
    print(dz)
