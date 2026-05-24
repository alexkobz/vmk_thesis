import yaml

from config import CONFIG_DIR

with open(CONFIG_DIR / "config.yml", "r") as file:
    cfg = yaml.safe_load(file)

y_name = cfg['y_name']
y_log = cfg['y_log']
secid = cfg["secid"]
tradedate = cfg["tradedate"]
year = cfg["year"]
boardid = cfg["boardid"]
inn = cfg["inn"]
share_type = cfg["type"]
is_vacation = cfg["is_vacation"]
issue_cumsum = cfg["issue_cumsum"]
close = cfg["close"]
type = cfg["type"]
sector = cfg["sector"]
x_lr_30 = cfg["x_lr_30"]
index_cols = cfg["index_cols"]
dp = cfg["data_processing"]
fe = cfg["feature_engineering"]
tr = cfg["training"]
mults = cfg["mults"]
lines = cfg["lines"]
macro = cfg["macro"]
dtype_dict_raw = cfg["dtype_dict_raw"]
dtype_dict = cfg["dtype_dict"]
raw_dataset_name = dp["raw_dataset_name"]
processed_dataset_name = fe["processed_dataset_name"]
