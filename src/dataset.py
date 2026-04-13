# src/dataset.py
import pandas as pd
import numpy as np
import os
import re
from collections import defaultdict, Counter


def load_all_data(data_dir, max_tags=50, test_ratio=0.2, random_seed=42):
    """
    读取所有4个CSV文件，构建混合特征，并进行 Train/Test 划分。
    """
    print(f">>> [Dataset] 正在读取数据: {data_dir} ...")

    # 1. 定义路径
    files = {
        'ratings': os.path.join(data_dir, 'ratings.csv'),
        'movies': os.path.join(data_dir, 'movies.csv'),
        'tags': os.path.join(data_dir, 'tags.csv'),
        'links': os.path.join(data_dir, 'links.csv')
    }

    # 2. 读取 CSV (容错处理)
    try:
        df_ratings = pd.read_csv(files['ratings'])
        df_movies = pd.read_csv(files['movies'])
        df_tags = pd.read_csv(files['tags'])
        df_links = pd.read_csv(files['links'])
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"❌ 数据文件缺失: {e}\n请确保 data 目录下包含 ratings.csv, movies.csv, tags.csv, links.csv")

    # ==========================
    # 3. ID 重映射 (Re-indexing)
    # ==========================
    # User ID 映射
    user_ids = df_ratings['userId'].unique()
    user_map = {old: new for new, old in enumerate(sorted(user_ids))}
    df_ratings['new_userId'] = df_ratings['userId'].map(user_map)

    # Movie ID 映射 (以 movies.csv 为准)
    movie_ids = df_movies['movieId'].unique()
    movie_map = {old: new for new, old in enumerate(sorted(movie_ids))}

    # 同步映射到所有表
    df_movies['new_movieId'] = df_movies['movieId'].map(movie_map)
    df_ratings['new_movieId'] = df_ratings['movieId'].map(movie_map)
    df_tags['new_movieId'] = df_tags['movieId'].map(movie_map)
    df_links['new_movieId'] = df_links['movieId'].map(movie_map)

    # ==========================
    # 4. 特征工程: Genres (类别)
    # ==========================
    all_genres = set()
    for genres in df_movies['genres']:
        if pd.isna(genres): continue
        for g in genres.split('|'):
            if g != '(no genres listed)':
                all_genres.add(g)
    genre_list = sorted(list(all_genres))
    genre_map = {g: i for i, g in enumerate(genre_list)}

    # ==========================
    # 5. 特征工程: Tags (标签)
    # ==========================
    # 统计 Top-K 热门标签
    df_tags['tag'] = df_tags['tag'].astype(str).str.lower()
    tag_counts = Counter(df_tags['tag'])
    top_tags = [t for t, c in tag_counts.most_common(max_tags)]
    tag_map = {t: i for i, t in enumerate(top_tags)}

    # 预处理：把每个电影的 tag 聚合起来 {movieId: [tag_index, ...]}
    movie_tags_dict = defaultdict(list)
    valid_tags = df_tags.dropna(subset=['new_movieId'])
    for _, row in valid_tags.iterrows():
        mid = int(row['new_movieId'])
        t = row['tag']
        if t in tag_map:
            movie_tags_dict[mid].append(tag_map[t])

    # ==========================
    # 6. 特征工程: Year (年份)
    # ==========================
    def extract_year(title):
        match = re.search(r'\((\d{4})\)', str(title))
        return int(match.group(1)) if match else 0

    df_movies['year'] = df_movies['title'].apply(extract_year)
    # 归一化年份 (Min-Max Scaling)
    years = df_movies['year'][df_movies['year'] > 1900]  # 过滤异常值
    if len(years) > 0:
        min_year, max_year = years.min(), years.max()
    else:
        min_year, max_year = 1990, 2020  # 默认值防止报错

    # ==========================
    # 7. 构建物品特征矩阵 (Item Feature Matrix)
    # ==========================
    # 为每个电影构建一个特征向量: [Genre(Multi-hot), Tag(Multi-hot), Year(Scalar)]
    item_features = {}

    for _, row in df_movies.iterrows():
        mid = row['new_movieId']
        if pd.isna(mid): continue
        mid = int(mid)

        # A. Genre
        g_vec = np.zeros(len(genre_list), dtype=np.float32)
        if not pd.isna(row['genres']):
            for g in row['genres'].split('|'):
                if g in genre_map:
                    g_vec[genre_map[g]] = 1.0

        # B. Tag
        t_vec = np.zeros(len(tag_map), dtype=np.float32)
        if mid in movie_tags_dict:
            for t_idx in movie_tags_dict[mid]:
                t_vec[t_idx] = 1.0

        # C. Year
        y_val = 0.0
        if row['year'] > 1900:
            y_val = (row['year'] - min_year) / (max_year - min_year + 1e-5)
        y_vec = np.array([y_val], dtype=np.float32)

        # 拼接所有特征
        item_features[mid] = np.concatenate([g_vec, t_vec, y_vec])

    # 计算特征总维度
    feature_dim = len(next(iter(item_features.values()))) if item_features else 0

    # ==========================
    # 8. 处理 Links (仅作映射表，用于可视化)
    # ==========================
    link_map = {}
    for _, row in df_links.iterrows():
        if not pd.isna(row['new_movieId']):
            link_map[int(row['new_movieId'])] = str(row['imdbId'])

    # ==========================
    # 9. 构建用户数据并划分 Train/Test
    # ==========================
    train_data = defaultdict(list)
    test_data = defaultdict(list)

    df_ratings = df_ratings.dropna(subset=['new_userId', 'new_movieId'])
    raw_data = df_ratings[['new_userId', 'new_movieId', 'rating']].values

    # 临时聚合
    temp_user_data = defaultdict(list)
    for uid, mid, rate in raw_data:
        uid, mid = int(uid), int(mid)
        if mid in item_features:
            vec = item_features[mid]
            temp_user_data[uid].append((mid, float(rate), vec))

    # 执行切分
    rng = np.random.default_rng(random_seed)
    total_samples = 0

    for uid, items in temp_user_data.items():
        if len(items) < 5:
            # 数据太少，不分测试集，全部用于训练
            train_data[uid] = items
        else:
            rng.shuffle(items)
            split_idx = int(len(items) * (1 - test_ratio))
            train_data[uid] = items[:split_idx]
            test_data[uid] = items[split_idx:]
        total_samples += len(items)

    stats = {
        'n_users': len(user_map),
        'n_items': len(movie_map),
        'feature_dim': feature_dim,
        'n_genres': len(genre_list),
        'n_tags': len(tag_map),
        'total_interactions': total_samples
    }

    print(f">>> [Dataset] 加载完成! 用户:{stats['n_users']}, 电影:{stats['n_items']}, 特征维数:{feature_dim}")
    print(
        f">>> [Dataset] 训练集样本: {sum(len(v) for v in train_data.values())}, 测试集样本: {sum(len(v) for v in test_data.values())}")

    return train_data, test_data, stats, link_map

