# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
Master code for executing prediction benchmarks
'''

if __name__ == "__main__":

    import argparse
    import logging
    logging.getLogger('matplotlib.font_manager').disabled = True
    from utils.prediction import lgbm_model_predict, lgbm_model_predict_streaming
    from utils.data_processing import read_data
    import joblib
    import pathlib
    import pandas as pd

    parser = argparse.ArgumentParser()

    parser.add_argument('-l',
                        '--logfile',
                        type=str,
                        default="",
                        help="log file to output benchmarking results to")

    parser.add_argument('-i',
                        '--intel',
                        default=False,
                        action="store_true",
                        help="use intel accelerated technologies where available")
    parser.add_argument('-mc',
                        '--clusteredmodel',
                        type=str,
                        default="",
                        help="provide clustered model")
    parser.add_argument('-mf',
                        '--fullmodel',
                        type=str,
                        default="",
                        help="provide full model")
    parser.add_argument('-s',
                        '--streaming',
                        default=False,
                        action="store_true",
                        help="run streaming inference if true")

    FLAGS = parser.parse_args()

    if FLAGS.logfile == "":
        logging.basicConfig(level=logging.DEBUG)
    else:
        path = pathlib.Path(FLAGS.logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=FLAGS.logfile, level=logging.DEBUG)
        
    logger = logging.getLogger()
    intel_flag = FLAGS.intel

    model_clustered = joblib.load(FLAGS.clusteredmodel)
    model_full = joblib.load(FLAGS.fullmodel)

    filepath = "./data/creditcard_test.csv"
    
    logger.info("Reading Data...")
    test_data = read_data(filepath)
    if FLAGS.streaming:
        X_test = test_data.drop(columns='Class')
        logger.info("=======> Testing model_cluster on test data")
        stream_pred_time = lgbm_model_predict_streaming(intel_flag, model_clustered, X_test)
        logger.info("=======> LGBM prediction time for clustered data = %s", str(stream_pred_time))

        logger.info("\n")

        logger.info("=======> Testing model_full on test data")
        stream_pred_time = lgbm_model_predict_streaming(intel_flag, model_clustered, X_test)
        logger.info("=======> LGBM prediction time for full data = %s", str(stream_pred_time))
        
        logger.info("\n")
    else:
        for multiplier in [0.5, 1, 2, 5, 10]:
            if multiplier == 0.5:
                test_data_large = test_data.sample(42500)
            else:
                test_data_large = pd.concat([test_data]*multiplier)
            test_data_large = test_data_large.sample(frac=1).reset_index(drop=True)
            X_test = test_data_large.drop(columns='Class')
            y_test = test_data_large['Class']
            
            logger.info("=======> Running the model on dataframe length of = %s", str(len(X_test)))

            logger.info("=======> Testing model_cluster on test data")
            rec_score, macro_f1score, positives, pred_time = lgbm_model_predict(intel_flag, model_clustered, X_test, y_test, 'clustered.png')
            logger.info("=======> Recall of model on test data = %s", str(rec_score))
            logger.info("=======> Macro f1 score of model on test data = %s", str(macro_f1score))
            logger.info("=======> Total Positives Predicted = %s", str(positives))
            logger.info("=======> LGBM prediction time for clustered data = %s", str(pred_time))

            logger.info("\n")

            logger.info("=======> Testing model_full on test data")
            rec_score, macro_f1score, positives, pred_time = lgbm_model_predict(intel_flag, model_full, X_test, y_test, 'full.png')
            logger.info("=======> Recall of model on full data = %s", str(rec_score))
            logger.info("=======> Macro f1 score of model on full data = %s", str(macro_f1score))
            logger.info("=======> Total Positives Predicted = %s", str(positives))
            logger.info("=======> LGBM prediction time for full data = %s", str(pred_time))
            
            logger.info("\n")
