from huggingface_hub import snapshot_download
from pathlib import Path
repo_path = "qmlp_dmlm_msmarco_distil_kl_l1_0.0001"
if not Path(repo_path).is_dir():
    snapshot_download(repo_id="lsr42/qmlp_dmlm_msmarco_distil_kl_l1_0.0001",
                      local_dir="qmlp_dmlm_msmarco_distil_kl_l1_0.0001")
