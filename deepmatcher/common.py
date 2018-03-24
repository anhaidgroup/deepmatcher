
"""
    Set random seed

    Args:
        seed (int): The random seed
"""
def set_seed(seed):
    ...

class BaseModel:
    """
    Base class for all models.
    """

    def train(train_dataset, validation_dataset=None):
        """
        Trains the model using training data from train_dataset.

        Args:
            train_dataset(Dataset): The training dataset object. At each
                                    iteration the get_next() method is called
                                    which returns the next set of training
                                    Examples.
            validation_dataset (Dataset): The validation dataset objectself.
                                          (Defaults to None)
        Returns:
            self, for chaining.
        """

    def evaluate(dataset):
        """
        Evluates the model using data from the dataset.

        Args:
            dataset (Dataset): The dataset to evaluate the model on.
        Returns:
            A dictionary of metrics with their corresponding values.
        """

    def predict(dataset):
        """
        Makes predictions given unlabeled data.

        Args:
            dataset (Dataset): The unlabeled dataset.
        Returns:
            A dictionary of IDs mapped to their predicted labels.
        """

class HybridModel(BaseModel):
    def __init__(pos_neg_weight_ratio, hidden_size=None, lr=0.001, lr_decay_factor=None, fine_config=None):
        """
        Create a Hybrid Entity Matching Model (see arxiv.org/...)

        Args:
            pos_neg_weight_ratio (float): Relative cost of misclassfying a
                                          true match compared to a non-match.
            hidden_size (int): Hidden size of the RNN used in the model.
                               Hidden size of other layers will be scaled
                               according to this unless overriden using
                               `fine_config`. If None, will be inferred using
                               heuristics. (Defaults to None)
            lr (float): Learning rate used for SGD. (Defaults to 0.001)
            lr_decay_factor (float): Factor by which learning rate should be
                                     multiplied after each epoch. If None,
                                     will be inferred using heuristics.
                                     (Defaults to None)
            fine_config (dict): Dictionary of fine grained configuration
                                parameters.
        """

    Note: This class will contain a field called `estimator` that will
          allow advanced users to directly make calls to tensorflow estimator
          API.

    ...

import deepmatcher as dm
import pandas as pd

table_i = pd.read_csv(...)
table_i = custom_prepreprocess_function(table_i)
table_i = dm.validate(table_i)

train, valid, test = dm.split_table(table_i, [0.6, 0.2, 0.2])

train_dataset = dm.preprocess(train, dataset_path='~/em_data/train.npy')
valid_dataset = dm.preprocess(valid, dataset_path='~/em_data/valid.npy')
test_dataset = dm.preprocess(test, dataset_path='~/em_data/test.npy')

em_model = dm.models.HybridModel(pos_neg_weight_ratio=1.2)

==========================
Workflow 1: Simple use case
--------------------------
print('Simple Training...')
em_model.train(train_dataset, valid_dataset)

print('Simple Testing...')
scores = em_model.evaluate(test_dataset)
print(scores)

==========================
Workflow 2: Advanced use case
--------------------------
print('Custom Training directly using TF estimator...')
em_model.estimator.train(..., hooks=CustomLoggingHook)
em_model.estimator.export_savedmodel(...)

print('Custom Training directly using TF estimator......')
scores = em_model.estimator.evaluate(..., hooks=CustomLoggingHook)
print(scores)

==========================
Workflow 3: Prediction
--------------------------
print('Simple Training...')
em_model.train(train_dataset, valid_dataset)

table_j = pd.read_csv(...)
table_j = dm.validate(table_j)
table_j = custom_prepreprocess_function(table_j)
unlabeled_dataset = dm.preprocess(table_j)

print('Prediction...')
predictions = em_model.predict(unlabeled_dataset)

preprocessed_j = unlabeled_dataset.get_raw_table()
for row in preprocessed_j.rows:
    print(row, 'Match prob = ', predictions[row['__id']])

==========================
Workflow B: Later iteration
--------------------------

import deepmatcher as dm
import pandas as pd

train_dataset = dm.preprocess(dataset_path='~/em_data/train.npy')
valid_dataset = dm.preprocess(dataset_path='~/em_data/valid.npy')
test_dataset = dm.preprocess(dataset_path='~/em_data/test.npy')

em_model = dm.models.HybridModel(pos_neg_weight_ratio=1.2)

print('Simple Training...')
em_model.train(train_dataset, valid_dataset)

print('Simple Testing...')
scores = em_model.evaluate(test_dataset)
print(scores)
