import pandas as pd
from fbprophet import Prophet


def preprocess(df):
    """This function takes a dataframe and preprocesses it so it is
    ready for the training stage.

    The DataFrame contains the time axis and the target column.

    It also contains some rows for which the target column is unknown.
    Those are the observations you will need to predict for KATE
    to evaluate the performance of your model.

    Here you will need to return the training time serie: ts together
    with the preprocessed evaluation time serie: ts_eval.

    Make sure you return ts_eval separately! It needs to contain
    all the rows for evaluation -- they are marked with the column
    evaluation_set. You can easily select them with pandas:

         - df.loc[df.evaluation_set]


    :param df: the dataset
    :type df: pd.DataFrame
    :return: ts, ts_eval
    """
    flag = df['evaluation_set']
    ts = df[~flag][['day', 'consumption']]
    ts_eval = df[flag][['day', 'consumption']]
    # transform consumption so that diff normal distribution
    ts['transf_consumption'] = ts['consumption']
    return ts, ts_eval


def train(ts):
    """Trains a new model on ts and returns it.

    :param ts: your processed training time serie
    :type ts: pd.DataFrame
    :return: a trained model
    """
    dat = pd.DataFrame()
    dat['ds'] = ts['day'].values
    dat['y'] = ts['transf_consumption'].values
    model = Prophet(daily_seasonality=0)
    model.fit(dat)
    return model


def predict(model, ts_test):
    """This functions takes your trained model as well
    as a processed test time serie and returns predictions.

    On KATE, the processed testt time serie will be the ts_eval you built
    in the "preprocess" function. If you're testing your functions locally,
    you can try to generate predictions using a sample test set of your
    choice.

    This should return your predictions either as a pd.DataFrame with one column
    or a pd.SeriesFalse

    :param model: your trained model
    :param ts_test: a processed test time serie (on KATE it will be ts_eval)
    :return: y_pred, your predictions
    """
    n_periods = ts_test.shape[0]
    df_dates = model.make_future_dataframe(periods=n_periods, include_history=False)
    model_prediction = model.predict(df_dates)
    y_pred = model_prediction[['ds', 'yhat']]
    y_pred = y_pred.set_index('ds')
    y_pred['yhat'] = y_pred['yhat']
    return y_pred['yhat']
