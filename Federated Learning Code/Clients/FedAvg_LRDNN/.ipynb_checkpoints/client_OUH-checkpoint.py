import warnings
import flwr as fl
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

import utils

LOCAL_DATA_DIR = '../raw/'

if __name__ == "__main__":
    
    print (LOCAL_DATA_DIR);
    
    #Import data
    
    #Perform test-train split
    
    #Perform normalisation
    
    #Perform imputation on test set (based on training set)
        
    #Call federated learning
    (X_train, y_train), (X_test, y_test) = utils.load_OUH()



    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty="l2",
        max_iter=1000,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model)

    # Define Flower client
    class LRClient(fl.client.NumPyClient):
        def get_parameters(self, config):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            #A,B,C=self.evaluate(parameters, config)
            print(f"Round {config['server_round']}")


            return utils.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            preds=model.predict_proba(X_test)[:,1]
            loss = log_loss(y_test,preds)
            accuracy = roc_auc_score(y_test,preds)
            return loss, len(X_test), {"AUROC": accuracy}

    # Start Flower client -- UNCOMMENT WHEN READY TO TEST
    #fl.client.start_numpy_client(server_address="localhost:8008", client=LRClient())
