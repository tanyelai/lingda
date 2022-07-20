from typing import List
from types import FunctionType
import pandas as pd


class MyDataLoader:
    def __init__(self, train_filepath: str, test_filepath: str, text_col: str = 'text',
                 target_col: str = 'label', train_has_index: bool = False, train_index_col: int = 0):
        """inits a class of type MyDataLoader

        Args:
          train_filepath (str): The file location of the train csv dataset.
          test_filepath (str): The file location of the test csv dataset.
          text_col (str): The column name of the text data in the train file.
            (default is 'text')
          target_col (str): The column name of the target variable of the task
            (default is 'label')
          train_has_index (bool): Specifies whether the train data csv has an index column.
            (default is False)
          train_index_col (int): Specifies the number of the index column. Will be ignored if no index column exists.
            (default is 0)

        returns:
          an instance of the class
        """

        if train_has_index:
            original_train = pd.read_csv(
                train_filepath, index_col=train_index_col)
        else:
            original_train = pd.read_csv(train_filepath)

        test = pd.read_csv(test_filepath)

        original_train.dropna(subset=[text_col, target_col], inplace=True)

        self.X_train = original_train[text_col]
        self.y_train = original_train[target_col]

        self.X_test = test[text_col]
        self.y_test = test[target_col]

        self.train_embeddings = None
        self.test_embeddings = None

    def get_original_train(self):
        null_msk = self.X_train.isnull()
        return self.X_train[~null_msk], self.y_train[~null_msk]

    def get_test(self):
        return self.X_test, self.y_test

    def apply_functions(self, transforms: List[FunctionType]) -> bool:
        """
        Applies functions in the list `transforms` to the texts in the train and test datasets

        Args:
            transforms (list of funcitons): a list of functions that accept `str` and return `str` to be applied to the input text

        returns:
            a bool value indicating whether the functions where applied successfuly or not
        """
        def transform(text):
            for t in transforms:
                text = t(text)
            return text

        try:
            self.X_train = self.X_train.map(transform)
            self.X_test = self.X_test.map(transform)
            return True
        
        except Exception as e:
            print(e)
            return False
            