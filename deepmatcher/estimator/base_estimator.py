
class BaseEstimator(tf.Estimator):
     def train(self,
            train_dataset,
            valid_dataset,
            hooks=None,
            steps=None,
            max_steps=None,
            saving_listeners=None):

       def evaluate(self,
          dataset):

       def predict(self,
          dataset):
