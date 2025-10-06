"""
Analyze and summarize metrics
"""

from collections import Counter
import glob
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


# noinspection PyPep8Naming
def regression_on_mod_depth_params(data_file="./Data/Sessions/attrib_params_pos_ppi.txt",
                                   n_model_runs=100,
                                   test_size=0.25):
    """
    Regression on link-depth attributes to see if they effect accuracy of LLM response.

    :param data_file: as produced by `drugmechcf.llmx.test_editlink.TestEditLink.write_all_sessions_sample_params`
    :param n_model_runs:
    :param test_size:
    """
    df = pd.read_csv(data_file)
    X = df.drop('response_is_correct', axis=1)
    y = df['response_is_correct']

    n_correct = df['response_is_correct'].sum()
    n_samples = df.shape[0]
    expected_accuracy_min = n_correct / n_samples
    if expected_accuracy_min < 0.5:
        expected_accuracy_min = 1.0 - expected_accuracy_min

    print("Columns:", ", ".join(df.columns.values.tolist()))
    print(f"nbr Samples read = {n_samples:,d}")
    print(f"nbr Correct response = {n_correct}, {n_correct/n_samples:.1%}")
    print()
    print(f"Desired min Accuracy of LR = {expected_accuracy_min:.3f}")
    print()

    model_runs_data = []
    for _ in range(n_model_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        run_data = [accuracy_score(y_test, y_pred), *model.coef_[0].tolist(), model.intercept_[0]]
        model_runs_data.append(run_data)

    model_runs_data = np.asarray(model_runs_data)

    var_names = ['accuracy'] + df.columns.values.tolist()[1:] + ['intercept']
    for i, vnm in enumerate(var_names):
        data = model_runs_data[:, i]
        print(f'{vnm}: mean = {np.mean(data):.5f}, s.d. = {np.std(data, ddof=0):.5f}')
        print()

    return


def get_llm_opt_freq(files_patt: str):
    files = glob.glob(files_patt)
    print()
    print("nbr Files =", len(files))

    opt_cnt = Counter()
    for file in files:
        with open(file) as jf:
            jdict = json.load(jf)
            opt_cnt.update([s['opt_match_metrics']["option_key"] for s in jdict['session']])

    n_samples = sum(opt_cnt.values())
    print("nbr samples read =", n_samples)
    print()

    maxw = max(len(k) for k in opt_cnt)

    for k, v in opt_cnt.most_common():
        print(f"{k:{maxw}s} = {v:5,d} ... {v/n_samples:6.1%}")

    print()
    return
