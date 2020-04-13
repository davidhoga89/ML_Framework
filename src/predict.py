import pandas as pd
import os
import joblib

from . import dispatcher
from sklearn import preprocessing
from sklearn import metrics

MODEL = os.environ.get("MODEL")
TEST_DATA = os.environ.get("TEST_DATA")

if __name__ == "__main__":
    df = pd.read_csv(TEST_DATA)

    for FOLD in range(5):
        encoders = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_label_encoder.pkl"))

        for c in train_df.columns:
            lbl = encoders[c]
            df.loc[:, c] = lbl.transform(df[c].values.tolist())

            # data is ready to train
            clf = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}.pkl"))
            cols = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_columns.pkl"))
            preds = clf.predict_proba(valid_df)[:, 1]
            # print(preds)
            print(metrics.roc_auc_score(yvalid, preds))
