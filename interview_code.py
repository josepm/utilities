import pandas as pd

df = pd.read_csv('~/my_repos/capacity_planning/data/config/interview_data.csv')
for c in ['ds_checkin', 'ds_checkout', 'ds_book']:
    df[c] = pd.to_datetime(df[c].values)


def user_trips(a_df_):
    # CI(i) <= CO(i) <= CI(j) <= CO(j)
    # same trip: CI(j) - CO(i) <= 3 AND CI(i) <= CI(j)
    a_df = a_df_.copy()
    a_df.sort_values(by='ds_checkin', inplace=True)
    a_df.reset_index(inplace=True, drop=True)
    dout = {'guest': [], 'first_home_ds_checkin': [], 'first_ds_book': [], 'dim_markets': []}
    guest = a_df.loc[a_df.index[0], 'id_guest']
    done_idx = list()
    for idx in a_df.index:
        if idx in done_idx:
            continue
        r = a_df.loc[idx, ]
        b_df = a_df[a_df.index > idx]
        co = r['ds_checkout']
        book = r['ds_book']
        markets = [r['dim_market']]
        for j in b_df.index:
            s = b_df.loc[j, ]
            if (s['ds_checkin'] - co).days <= 3:
                done_idx.append(j)
                co = max(co, s['ds_checkout'])  # concatenate trips
                book = min(book, s['ds_book'])
                markets.append(s['dim_market'])
            else:  # sorted checkin dates!: the next ones will not work
                break
        # log a new trip
        dout['first_ds_book'].append(book)
        dout['dim_markets'].append(list(set(markets)))
        dout['first_home_ds_checkin'].append(r['ds_checkin'])
        dout['guest'].append(guest)
    return pd.DataFrame(dout)


t_df = df.groupby('id_guest')[df.columns].apply(user_trips).reset_index(drop=True)
