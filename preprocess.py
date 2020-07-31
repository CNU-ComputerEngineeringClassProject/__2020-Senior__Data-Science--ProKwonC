import pandas as pd


class Preprocess(object):

    def __init__(self, raw_data):
        self.genderless  = raw_data.drop(raw_data[raw_data['성별'].notnull()].index)
        self.gender = raw_data.dropna(subset=['성별'])

    def run(self):
        return self.gender, self.genderless , self.get_gender_preprocess_dataset(), self.get_genderless_preprocess_dataset()

    def get_gender_preprocess_dataset(self):
        dataset = self.parse_dataset(self.gender)
        dataset = self.normalize(dataset)
        return dataset

    def get_genderless_preprocess_dataset(self):
        dataset = self.parse_dataset(self.genderless)
        dataset = self.normalize(dataset)
        return dataset

    def parse_dataset(self, parsing_dataset):
        return parsing_dataset[['성별','이용건수', '운동량', '탄소량','이동거리','사용시간']]

    def normalize(self, dataset):

        dataset.loc[dataset['성별'] == 'f', '성별'] = 'F'
        dataset.loc[dataset['성별'] == 'm', '성별'] = 'M'

        idx_n = dataset[dataset['운동량'] == '\\N'].index
        dataset = dataset.drop(idx_n)
        dataset['운동량'] = pd.to_numeric(dataset['운동량'])

        idx_n = dataset[dataset['탄소량'] == '\\N'].index
        dataset = dataset.drop(idx_n)
        dataset['탄소량'] = pd.to_numeric(dataset['탄소량'])

        dataset['탄소량'] = pd.to_numeric(dataset['탄소량'] / dataset['이용건수'])
        dataset['운동량'] = pd.to_numeric(dataset['운동량'] / dataset['이용건수'])
        dataset['사용시간'] = pd.to_numeric(dataset['사용시간'] / dataset['이용건수'])
        dataset['이동거리'] = pd.to_numeric(dataset['이동거리'] / dataset['이용건수'])
        del dataset['이용건수']

        return dataset