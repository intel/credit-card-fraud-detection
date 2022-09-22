# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
This code has the functions needed for batch/streaming prediction
'''

import time
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, plot_confusion_matrix, f1_score

def lgbm_model_predict(flag, lgb_classifier, df_for_prediction, class_for_prediction, filename):
    lst_prediction_time = []
    for _i in [1, 2, 3, 4, 5]:
        if flag:
            import daal4py as d4p  # pylint: disable=C0415,E0401
            daal_model = d4p.get_gbt_model_from_lightgbm(lgb_classifier.booster_)
            start_time = time.time()
            y_pred = d4p.gbt_classification_prediction(nClasses=2).compute(df_for_prediction, daal_model).prediction  # noqa: F841
            end_time = time.time()

        else:
            start_time = time.time()
            y_pred = lgb_classifier.predict(df_for_prediction)  # noqa: F841
            end_time = time.time()
        lst_prediction_time.append(end_time-start_time)
    pred_time = min(lst_prediction_time)
    rec_score = recall_score(class_for_prediction, y_pred.reshape(-1))
    macrof1_score = f1_score(class_for_prediction, y_pred.reshape(-1), average='macro')
    plot_confusion_matrix(lgb_classifier, df_for_prediction, class_for_prediction)
    plt.savefig(filename)
    plt.show()
    return rec_score, macrof1_score, sum(y_pred.reshape(-1)), pred_time
    
def lgbm_model_predict_streaming(flag, lgb_classifier, df_for_prediction):
    if flag:
        import daal4py as d4p  # pylint: disable=C0415,E0401
        daal_model = d4p.get_gbt_model_from_lightgbm(lgb_classifier.booster_)

    lst_for_avg_stream_time = []
    test_df = df_for_prediction.sample(n=1000)
    for _i in [1, 2, 3, 4, 5]:
        lst_of_stream_times = []
        for _counter in range(len(test_df)):
            sample_df = test_df.sample(n=1)
            if flag:
                start_time = time.time()
                y_pred = d4p.gbt_classification_prediction(nClasses=2).compute(sample_df, daal_model).prediction
                end_time = time.time()
            else:
                start_time = time.time()
                y_pred = lgb_classifier.predict(sample_df)
                end_time = time.time()
            lst_of_stream_times.append(end_time-start_time)
        lst_for_avg_stream_time.append(sum(lst_of_stream_times) / len(lst_of_stream_times))
    pred_time = min(lst_for_avg_stream_time)
    return y_pred, pred_time
