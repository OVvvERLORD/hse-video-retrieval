"""
Загрузка Gradio-приложения в Hugging Face Space.
"""

import os

from huggingface_hub import HfApi

SPACE_REPO_ID = "Ylko/emovid-search"     # имя будущего Space
DATASET_REPO_ID = "Ylko/emovid"          # приватный датасет с видео
PRIVATE = True                           # Space приватный

TOKEN = os.environ.get("HF_TOKEN")       
if not TOKEN:
    raise SystemExit("Задайте write-токен: export HF_TOKEN='hf_...'")

_HERE = os.path.dirname(os.path.abspath(__file__))  
_ROOT = os.path.dirname(_HERE)                       

FILES_TO_UPLOAD = {
    os.path.join(_HERE, "app.py"): "app.py",
    os.path.join(_HERE, "gradio_app.py"): "gradio_app.py",
    os.path.join(_HERE, "requirements.txt"): "requirements.txt",
    os.path.join(_ROOT, "baseline.py"): "baseline.py",
    os.path.join(_ROOT, "data_saver.py"): "data_saver.py",
    os.path.join(_ROOT, "metadata.parquet"): "metadata.parquet",
    os.path.join(_ROOT, "vectors.usearch"): "vectors.usearch",
    os.path.join(_ROOT, "stash", "model_surely_not_overfitted.joblib"): "model_surely_not_overfitted.joblib",
}

SPACE_README = f"""---
title: EmoVid Search
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.16.0
app_file: app.py
pinned: false
---

# 🎬 Поиск видео по текстовому описанию

Подбор эмоционально подходящих видео из библиотеки EmoVid по текстовому
промпту. Видео подтягиваются по запросу из приватного датасета
[`{DATASET_REPO_ID}`](https://huggingface.co/datasets/{DATASET_REPO_ID}).

"""


def main() -> None:
    api = HfApi(token=TOKEN)
    who = api.whoami()["name"]
    print(f"Авторизованы как: {who}")

    api.create_repo(
        SPACE_REPO_ID,
        repo_type="space",
        space_sdk="gradio",
        private=PRIVATE,
        exist_ok=True,
    )
    print(f"Space готов: https://huggingface.co/spaces/{SPACE_REPO_ID}")

    api.upload_file(
        path_or_fileobj=SPACE_README.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=SPACE_REPO_ID,
        repo_type="space",
    )

    for local_path, repo_path in FILES_TO_UPLOAD.items():
        if not os.path.exists(local_path):
            print(f"  ! пропуск (нет файла): {local_path}")
            continue
        print(f"  загрузка {local_path} -> {repo_path}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=SPACE_REPO_ID,
            repo_type="space",
        )

    print("\nГотово. Не забудьте добавить секрет HF_TOKEN в настройках Space:")
    print(f"  https://huggingface.co/spaces/{SPACE_REPO_ID}/settings")


if __name__ == "__main__":
    main()
