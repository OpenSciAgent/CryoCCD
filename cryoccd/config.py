import logging 
import os
import sys

logger = logging.getLogger(__name__)

def get_model_from_args():
    if "--model" in sys.argv:
        index = sys.argv.index("--model")
        if index < len(sys.argv) - 1:
            return sys.argv[index + 1]
    return os.environ.get("CRYOGEM_MODEL", "cryogem")

_select_model = get_model_from_args()
_select_dataset = "cryogem"  # 数据集保持不变

logger.info(f"Selected model: {_select_model}")