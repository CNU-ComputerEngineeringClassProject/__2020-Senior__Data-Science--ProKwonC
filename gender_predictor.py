from os.path import join
import tensorflow as tf
import pandas as pd
import preprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


class GenderPredictor(object):

    def __init__(self,gender_raw, genderless_raw, gender_input, genderless_input):
        self.gender_raw = gender_raw
        self.genderless_raw = genderless_raw
        self.gender_input = gender_input
        self.genderless_input = genderless_input

    def run(self):
        pred_gender = self.model_genderpredictor()
        self.make_csv_file(self.gender_raw, self.genderless_raw, pred_gender)

    def model_genderpredictor(self):

        test = self.gender_input.iloc[-100:, :]
        train = self.gender_input.iloc[:-100, :]

        model = self._model_train(train)
        self._model_evaluate(model, test)

        return self._model_pred(model)

    def _model_train(self, train):
        trnx, tstx, trny, tsty = self._model_train_data(train)
        m = Sequential()

        m.add(Dense(10, kernel_initializer='uniform', input_shape=(4,), activation='relu'))
        m.add(Dense(20, kernel_initializer='uniform', activation='relu'))
        m.add(Dense(10, kernel_initializer='uniform', activation='relu'))
        m.add(Dense(2, kernel_initializer='uniform', activation='sigmoid'))

        m.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

        m.summary()

        m.fit(trnx, trny, validation_data=(tstx, tsty), batch_size=10, epochs=50)

        return m

    def _model_train_data(self, train):
        trnx, tstx, trny, tsty = train_test_split(train.iloc[:, 1:], train.iloc[:, 0], test_size=0.3, random_state=42)

        encoder = LabelEncoder()
        y1 = encoder.fit_transform(trny)
        trny = pd.get_dummies(y1).values

        encoder = LabelEncoder()
        y1 = encoder.fit_transform(tsty)
        tsty = pd.get_dummies(y1).values

        self.scaler = MinMaxScaler()
        trnx = self.scaler.fit_transform(trnx)
        tstx = self.scaler.transform(tstx)

        return trnx, tstx, trny, tsty

    def _model_evaluate(self, model, test):
        test_x = test.iloc[:, 1:]
        test_y = test.iloc[:, 0]

        encoder = LabelEncoder()
        y1 = encoder.fit_transform(test_y)
        test_y = pd.get_dummies(y1).values

        test_x = self.scaler.transform(test_x)

        loss, accuracy = model.evaluate(test_x, test_y)
        print("Evaluate_Accuracy = {:.2f}".format(accuracy))

    def _model_pred(self, model):
        pred_dataset = self.genderless_input.iloc[:, 1:]

        pred_out_data = model.predict(pred_dataset)

        pred_out_class = []
        for i, pred in zip(range(0, len(pred_out_data)), pred_out_data):
            pred_out_class.append(self.class_names[np.argmax(pred)])

        pred_out_class = np.array(pred_out_class)

        return pred_out_class

    def make_csv_file(self, gender, genderless, pred_gender):
        genderless['성별'] = pred_gender.T
        new_gender_dataset = np.concatenate((gender, genderless), axis=0)

        header = ['대여일자', '대여시간', '대여소번호', '대여소명', '대여구분코드', '성별', '연령대코드', '이용건수', '운동량', '탄소량', '이동거리', '사용시간']
        dataframe = pd.DataFrame(new_gender_dataset)
        dataframe.to_csv("./data/서울특별시_공공자전거_시간대별_대여정보_201812_201905(6)_new.csv", header=header, index=False,
                         encoding='utf-8-sig')

if __name__ == '__main__':

    def main(argv=None):

        raw_data = pd.read_csv(join('data', '서울특별시_공공자전거_시간대별_대여정보_201812_201905(6).csv'), encoding='CP949')
        gender_raw, genderless_raw, gender_input, genderless_input = preprocess.Preprocess(raw_data).run()
        GenderPredictor(gender_raw, genderless_raw, gender_input, genderless_input).run()

    tf.compat.v1.app.run()
