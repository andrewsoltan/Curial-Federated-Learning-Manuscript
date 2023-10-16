import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict
from sklearn.metrics import roc_auc_score

UHB=0
OUH=0

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    (X_test2, y_test2) = utils.load_test_UHB()
    (X_test1, y_test1) = utils.load_test_OUH()

    (X_test3, y_test3) = utils.load_test_Bed()

    (X_test4, y_test4) = utils.load_test_Port()

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)

        preds=model.predict_proba(X_test1)[:,1]
        loss = log_loss(y_test1,preds)
        auc1 = roc_auc_score(y_test1,preds)
        auc1_95CI=utils.AUC_CIs (preds, y_test1, alpha = 0.95)

        preds=model.predict_proba(X_test2)[:,1]
        loss = log_loss(y_test2,preds)
        auc2 = roc_auc_score(y_test2,preds)
        auc2_95CI=utils.AUC_CIs (preds, y_test2, alpha = 0.95)

        preds=model.predict_proba(X_test3)[:,1]
        loss = log_loss(y_test3,preds)
        auc3 = roc_auc_score(y_test3,preds)
        auc3_95CI=utils.AUC_CIs (preds, y_test3, alpha = 0.95)

        preds=model.predict_proba(X_test4)[:,1]
        loss = log_loss(y_test4,preds)
        auc4 = roc_auc_score(y_test4,preds)
        auc4_95CI=utils.AUC_CIs (preds, y_test4, alpha = 0.95)


        return loss, {"OUH": auc1_95CI, "UHB": auc2_95CI, "Bed":auc3_95CI, "Port":auc4_95CI}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression()
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=3,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    Z=fl.server.start_server(
        server_address="localhost:8008",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=20),
    )
    print('==========================')
    # print(Z)
