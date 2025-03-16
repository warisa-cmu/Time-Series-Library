import marimo

__generated_with = "0.11.20"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import marimo as mo
    os.getcwd()
    return mo, os


@app.cell
def _():
    import copy
    from pathlib import Path
    import warnings

    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
    from lightning.pytorch.loggers import TensorBoardLogger
    import numpy as np
    import pandas as pd
    import torch

    from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
    from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
    return (
        Baseline,
        EarlyStopping,
        GroupNormalizer,
        LearningRateMonitor,
        MAE,
        Path,
        PoissonLoss,
        QuantileLoss,
        SMAPE,
        TemporalFusionTransformer,
        TensorBoardLogger,
        TimeSeriesDataSet,
        copy,
        np,
        optimize_hyperparameters,
        pd,
        pl,
        torch,
        warnings,
    )


@app.cell
def _():
    from pytorch_forecasting.data.examples import get_stallion_data

    data = get_stallion_data()
    return data, get_stallion_data


@app.cell
def _(data, mo):
    mo.ui.table(data)
    return


@app.cell
def _(data, np):
    # add time index
    data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
    data["time_idx"] -= data["time_idx"].min()

    # add additional features
    data["month"] = data.date.dt.month.astype(str).astype("category")  # categories have be strings
    data["log_volume"] = np.log(data.volume + 1e-8)
    data["avg_volume_by_sku"] = data.groupby(["time_idx", "sku"], observed=True).volume.transform("mean")
    data["avg_volume_by_agency"] = data.groupby(["time_idx", "agency"], observed=True).volume.transform("mean")
    return


@app.cell
def _(data):
    data["date"].dt.year * 12 + data["date"].dt.month
    return


@app.cell
def _(data):
    # we want to encode special days as one variable and thus need to first reverse one-hot encoding
    special_days = [
        "easter_day",
        "good_friday",
        "new_year",
        "christmas",
        "labor_day",
        "independence_day",
        "revolution_day_memorial",
        "regional_games",
        "fifa_u_17_world_cup",
        "football_gold_cup",
        "beer_capital",
        "music_fest",
    ]
    data[special_days] = data[special_days].apply(lambda x: x.map({0: "-", 1: x.name})).astype("category")
    return (special_days,)


@app.cell
def _(GroupNormalizer, TimeSeriesDataSet, data, special_days):
    max_prediction_length = 6
    max_encoder_length = 24
    training_cutoff = data["time_idx"].max() - max_prediction_length

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="volume",
        group_ids=["agency", "sku"],
        min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["agency", "sku"],
        static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
        time_varying_known_categoricals=["special_days", "month"],
        variable_groups={"special_days": special_days},  # group of categorical variables can be treated as one variable
        time_varying_known_reals=["time_idx", "price_regular", "discount_in_percent"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "volume",
            "log_volume",
            "industry_volume",
            "soda_volume",
            "avg_max_temp",
            "avg_volume_by_agency",
            "avg_volume_by_sku",
        ],
        target_normalizer=GroupNormalizer(
            groups=["agency", "sku"], transformation="softplus"
        ),  # use softplus and normalize by group
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # create validation set (predict=True) which means to predict the last max_prediction_length points in time
    # for each series
    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

    # create dataloaders for model
    batch_size = 128  # set this between 32 to 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
    return (
        batch_size,
        max_encoder_length,
        max_prediction_length,
        train_dataloader,
        training,
        training_cutoff,
        val_dataloader,
        validation,
    )


@app.cell
def _(Baseline, MAE, val_dataloader):
    # calculate baseline mean absolute error, i.e. predict next value as the last available value from the history
    baseline_predictions = Baseline().predict(val_dataloader, return_y=True)
    MAE()(baseline_predictions.output, baseline_predictions.y)
    return (baseline_predictions,)


@app.cell
def _(QuantileLoss, TemporalFusionTransformer, pl, training):
    # configure network and trainer
    pl.seed_everything(42)
    trainer = pl.Trainer(
        accelerator="cpu",
        # clipping gradients is a hyperparameter and important to prevent divergance
        # of the gradient for recurrent neural networks
        gradient_clip_val=0.1,
    )


    tft = TemporalFusionTransformer.from_dataset(
        training,
        # not meaningful for finding the learning rate but otherwise very important
        learning_rate=0.03,
        hidden_size=8,  # most important hyperparameter apart from learning rate
        # number of attention heads. Set to up to 4 for large datasets
        attention_head_size=1,
        dropout=0.1,  # between 0.1 and 0.3 are good values
        hidden_continuous_size=8,  # set to <= hidden_size
        loss=QuantileLoss(),
        optimizer="Ranger",
        # reduce learning rate if no improvement in validation loss after x epochs
        # reduce_on_plateau_patience=1000,
    )
    print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")
    return tft, trainer


@app.cell
def _(tft, train_dataloader, trainer, val_dataloader):
    # find optimal learning rate
    from lightning.pytorch.tuner import Tuner

    res = Tuner(trainer).lr_find(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        max_lr=10.0,
        min_lr=1e-6,
    )

    print(f"suggested learning rate: {res.suggestion()}")
    fig = res.plot(show=True, suggest=True)
    fig.show()
    return Tuner, fig, res


@app.cell
def _(
    EarlyStopping,
    LearningRateMonitor,
    QuantileLoss,
    TemporalFusionTransformer,
    TensorBoardLogger,
    pl,
    tft,
    training,
):
    # configure network and trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

    trainer2 = pl.Trainer(
        max_epochs=50,
        accelerator="cpu",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        limit_train_batches=50,  # coment in for training, running valiation every 30 batches
        # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )

    tft2 = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(),
        log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        optimizer="Ranger",
        reduce_on_plateau_patience=4,
    )
    print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")
    return early_stop_callback, logger, lr_logger, tft2, trainer2


if __name__ == "__main__":
    app.run()
