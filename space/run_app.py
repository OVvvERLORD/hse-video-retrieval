import os
import sys

# Корень репозитория — там лежат baseline.py и data_saver.py (общие модули),
# а также metadata.parquet, vectors.usearch, EmoVid_Data/ и stash/.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Переходим в корень, чтобы DataStorage нашёл metadata.parquet и vectors.usearch
# (они грузятся относительно рабочей папки). Благодаря этому приложение можно
# запускать как кнопкой ▶️ в VSCode, так и из любой папки.
os.chdir(_ROOT)

from gradio_app import VideoSearchApp

TARGET_ROOT_DIR = os.path.join(_ROOT, 'EmoVid_Data')
MODEL_PATH = os.path.join(_ROOT, 'stash', 'model_surely_not_overfitted.joblib')

app = VideoSearchApp(root_dir=TARGET_ROOT_DIR, model_path=MODEL_PATH)
app.launch(share=True)
