# Kaggle runner (copy each block into its own notebook cell)
Before starting: Settings -> Accelerator GPU, Internet On; Add Input -> dataset `yrk2005/spoofdet-data-v1`
(its owner must share it with your account). If `features/` is not found, run `!find /kaggle/input -name feats.npy` and point
SPOOFDET_DATA at the folder above `features/`.

## Cell 1 - code
```python
import os
from kaggle_secrets import UserSecretsClient           # Add-ons -> Secrets -> add GITHUB_TOKEN
tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
BRANCH = "main"                                        # the branch that holds the merged code of A, B and C
!git clone https://{tok}@github.com/aashrithreddyjakku-cmd/cross-method-spoof-detection.git /kaggle/working/repo
%cd /kaggle/working/repo
!git checkout {BRANCH} && git log -1 --oneline
!pip install -q -e . librosa soundfile pyyaml tqdm scikit-learn tabulate
```
## Cell 2 - environment
```python
import os
os.environ["SPOOFDET_DATA"]    = "/kaggle/input/spoofdet-data-v1"        # contains features/ and manifests/
os.environ["SPOOFDET_CKPT"]    = "/kaggle/working/checkpoints"
os.environ["SPOOFDET_RESULTS"] = "/kaggle/working/results"
import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
```
## Cell 3 - run a queue (re-run the same cell after a timeout; finished runs are skipped)
```python
!python scripts/make_matrix.py --kind p0 --models smallcnn lcnn resnet18 vggish_cbam --seeds 1
!python scripts/run_queue.py "configs/experiments/p0_*_s1.yaml"
```
## Cell 4 - package results for the team
```python
!cd /kaggle/working && zip -rq B_batch1.zip results/scores checkpoints/*/best.pt checkpoints/*/train_log.csv checkpoints/*/config.yaml
!ls -lh /kaggle/working/B_batch1.zip
```
