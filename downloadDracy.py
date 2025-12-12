# 作为示例脚本运行
from pathlib import Path
from neuralop.data.datasets.web_utils import download_from_zenodo_record

root = Path("/home/leeshu/wmm/neuraloperator-main/neuralop/data/darcy").expanduser()  # 你希望存放数据的目录
all_res = [16, 32, 64, 128, 421]
files = [f"darcy_{res}.tgz" for res in all_res]

download_from_zenodo_record(
    record_id="12784353",
    root=root,
    files_to_download=files
)
print("All Darcy resolutions downloaded to:", root)