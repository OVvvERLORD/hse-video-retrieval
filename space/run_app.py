import os
import sys

# Корень репозитория — там лежат baseline.py и data_saver.py (общие модули),
# а также metadata.parquet, vectors.usearch, EmoVid_Data/ и stash/.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Переходим в корень, чтобы DataStorage нашёл metadata.parquet и vectors.usearch
os.chdir(_ROOT)

from gradio_app import VideoSearchApp

TARGET_ROOT_DIR = os.path.join(_ROOT, 'EmoVid_Data')
EMBEDDER_PATH = os.path.join(_ROOT, 'stash', 'emotion-embedder')
CLASSIFIER_PATH = os.path.join(_ROOT, 'stash', 'emotion_classifier_head_best.pt')

app = VideoSearchApp(
    root_dir=TARGET_ROOT_DIR,
    backend="bert",
    embedder_path=EMBEDDER_PATH,
    classifier_path=CLASSIFIER_PATH,
)
app.launch(share=True)
