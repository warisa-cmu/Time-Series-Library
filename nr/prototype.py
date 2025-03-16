from models.TimesNet import Model
from dataclasses import make_dataclass


configsDict = dict(
    task_name="short_term_forecast",
    root_path="./dataset/m4",
    seasonal_patterns="Yearly",
    model_id="m4_Yearly",
    model="TimesNet",
    data="m4",
    features="M",
    seq_len=16,
    label_len=8,
    pred_len=16,
    num_kernels=8,
    embed="timeF",
    freq="h",
    dropout=0.2,
    e_layers=2,
    d_layers=1,
    factor=3,
    enc_in=1,
    dec_in=1,
    c_out=1,
    batch_size=16,
    d_model=32,
    d_ff=32,
    top_k=5,
    des="Exp",
    itr=1,
    learning_rate=0.001,
    loss="SMAPE",
)

configs = make_dataclass(
    "ConfigsDataClass", ((k, type(v)) for k, v in configsDict.items())
)(**configsDict)


model = Model(configs=configs)

pass
