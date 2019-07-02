import pandas as pd

INPUT_PATH = 'Data/ratings.csv'
OUTPUT_PATH_TRAIN = 'Data/train.ratings'
OUTPUT_PATH_TEST = 'Data/test.ratings'

if __name__ == '__main__':
    df = pd.read_csv(INPUT_PATH, ',', names=['userId', 'docId', 'rating', 'timestamp'])
    df.sort_values(by=['timestamp'], inplace=True)
    mask = df.duplicated(subset={'userId'}, keep='last')
    train_df = df[mask]
    test_df = df[~mask]
    train_df.sort_values(by=['userId', 'timestamp'], inplace=True)
    test_df.sort_values(by=['userId', 'timestamp'], inplace=True)
    train_df.to_csv(OUTPUT_PATH_TRAIN, ',', header=False, index=False)
    test_df.to_csv(OUTPUT_PATH_TEST, ',', header=False, index=False)
