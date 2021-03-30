import os
import wget
import zipfile
import numpy as np
import json


def download_raw_dataset(url, folder_path, is_zip):
    if os.path.exists(folder_path):
        print('Raw data already exists. Skip downloading')
        return
    print("Raw file doesn't exist. Downloading...")
    os.makedirs(folder_path)
    filename = wget.download(url, out=folder_path)
    if is_zip:
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(folder_path)


def make_implicit(df, min_rating):
    print('Turning into implicit ratings')
    df = df[df['rating'] >= min_rating]
    return df


def filter_triplets(df, min_user, min_item, USER_COL, ITEM_COL, RATING_COL, TIMESTAMP_COL):
    print('Filtering triplets')

    length_old = len(df)

    if min_user > 0:
        user_sizes = df.groupby(USER_COL).size()
        good_users = user_sizes.index[user_sizes >= min_user]
        df = df[df[USER_COL].isin(good_users)]

    if min_item > 0:
        item_sizes = df.groupby(ITEM_COL).size()
        good_items = item_sizes.index[item_sizes >= min_item]
        df = df[df[ITEM_COL].isin(good_items)]

    # while(len(df) != length_old):
    while (len(df) != length_old):
        length_old = len(df)

        if min_user > 0:
            user_sizes = df.groupby(USER_COL).size()
            good_users = user_sizes.index[user_sizes >= min_user]
            df = df[df[USER_COL].isin(good_users)]

        if min_item > 0:
            item_sizes = df.groupby(ITEM_COL).size()
            good_items = item_sizes.index[item_sizes >= min_item]
            df = df[df[ITEM_COL].isin(good_items)]

    return df


def densify_index(df, USER_COL, ITEM_COL, RATING_COL, TIMESTAMP_COL):
    print('Densifying index')
    umap = {u: i for i, u in enumerate(set(df[USER_COL]))}
    smap = {s: i for i, s in enumerate(set(df[ITEM_COL]))}
    df[USER_COL] = df[USER_COL].map(umap)
    df[ITEM_COL] = df[ITEM_COL].map(smap)
    return df, umap, smap


def split_df(df, user_count, split, USER_COL, ITEM_COL, RATING_COL, TIMESTAMP_COL):
    if split == 'leave_one_out':
        print('Splitting')
        user_group = df.groupby(USER_COL)
        user2items = user_group.apply(lambda d: list(d.sort_values(by=TIMESTAMP_COL)[ITEM_COL]))
        train, val, test = {}, {}, {}
        for user in range(user_count):
            items = user2items[user]
            if len(items) > 2:
                train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:] 
        return train, val, test


def store_data(train, val, test, umap, smap, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        flag = "y"
    else:
        print("%s is already exsiting, overwirite it? [Y/N]" % output_path)
        flag = input()
    if flag.lower() == "y":
        for name, data in zip(["train", "val", "test", "umap", "smap"], [train, val, test, umap, smap]):
            with open(os.path.join(output_path, "%s.json" % name), 'w') as f:
                json.dump(data, f)
            print("%s stored!" % ("%s.json" % name))