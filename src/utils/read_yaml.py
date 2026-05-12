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
close = cfg["close"]
type = cfg["type"]
sector = cfg["sector"]
x_lr_30 = cfg["x_lr_30"]
index_cols = cfg["index_cols"]
dp = cfg["data_processing"]
fe = cfg["feature_engineering"]
tr = cfg["training"]
ev = cfg["evaluate"]
mults = cfg["mults"]
lines = cfg["lines"]
macro = cfg["macro"]
dtype_dict_raw = cfg["dtype_dict_raw"]
