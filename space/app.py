"""
Точка входа для Hugging Face Space (SDK: gradio).

Особенность развёртывания: сам датасет EmoVid_Data весит ~21 ГБ и лежит в
приватном датасет-репозитории `Ylko/emovid`. Класть его целиком в Space или
скачивать при старте — дорого и медленно, поэтому здесь видео и аудио
подтягиваются ПО ОДНОМУ, по мере выдачи результатов поиска, через
huggingface_hub.hf_hub_download (с кешем).

В Space-репозитории физически лежат только лёгкие артефакты:
  - model_surely_not_overfitted.joblib  (SVC-классификатор эмоций)
  - metadata.parquet                    (таблица метаданных)
  - vectors.usearch                     (готовый векторный индекс)

Для доступа к приватному датасету в настройках Space нужно добавить
секрет HF_TOKEN (read-токен). См. README этого Space.
"""

import os
import shutil
import sys
import uuid

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE

import gradio_app


# Датасет-репозиторий, в который загрузили EmoVid_Data.
DATASET_REPO_ID = os.environ.get("EMOVID_DATASET_REPO", "Ylko/emovid")

# Read-токен HF (берётся из секрета Space с именем HF_TOKEN).
HF_TOKEN = os.environ.get("HF_TOKEN")

# Префикс, с которым абсолютные пути записаны в metadata.parquet.
# Всё, что после него, является путём ОТНОСИТЕЛЬНО корня датасета.
LOCAL_DATA_PREFIX = "EmoVid_Data/"

MODEL_PATH = "model_surely_not_overfitted.joblib"


def _to_repo_relpath(local_path: str) -> str:
    """Превращает абсолютный путь из metadata.parquet в путь внутри датасета.

    '/home/.../EmoVid_Data/sticker/annotation/excitement/04545.json'
        -> 'sticker/annotation/excitement/04545.json'
    """
    norm = local_path.replace("\\", "/")
    idx = norm.find(LOCAL_DATA_PREFIX)
    if idx != -1:
        return norm[idx + len(LOCAL_DATA_PREFIX):]
    return norm.lstrip("/")


class SpaceVideoSearchApp(gradio_app.VideoSearchApp):
    """Версия приложения для Space: файлы тянутся из приватного датасета на лету."""

    def _download_from_dataset(self, repo_relpath: str) -> str | None:
        """Скачивает (с кешем) один файл из датасета. None, если файла нет."""
        try:
            return hf_hub_download(
                repo_id=DATASET_REPO_ID,
                filename=repo_relpath,
                repo_type="dataset",
                token=HF_TOKEN,
            )
        except Exception as e:  
            print(f"[hub] не удалось скачать {repo_relpath}: {e}")
            return None

    def _ensure_servable(self, path: str) -> str:
        """Гарантирует, что файл лежит в temp_dir (внутри рабочей папки /app).

        Gradio отдаёт в браузер только файлы из рабочей директории/temp, но не из
        кеша huggingface_hub. Файлы из кеша (стикеры без аудио, либо неудавшаяся
        склейка) копируем в temp_dir, чтобы избежать InvalidPathError.
        """
        abs_path = os.path.abspath(path)
        if abs_path.startswith(self.temp_dir + os.sep):
            return abs_path
        dst = os.path.join(self.temp_dir, f"{uuid.uuid4().hex}{os.path.splitext(abs_path)[1]}")
        shutil.copyfile(abs_path, dst)
        return dst

    def process_prompt_and_get_videos(self, prompt: str, num_videos: int):
        """Та же логика, но видео/аудио подтягиваются из датасета на Hub."""
        self.cleanup_old_temp_files(max_age_minutes=5)

        matches = self.storage.search(prompt, num_videos).keys
        df = self.storage.df.set_index("usearch_uid")
        found_annotation_paths = list(df.loc[matches]["file_path"])

        muxed_video_paths = []
        for ann_path in found_annotation_paths:
            ann_rel = _to_repo_relpath(ann_path)

            video_rel = ann_rel.replace("/annotation/", "/video/", 1).rsplit(".", 1)[0] + ".mp4"
            local_video = self._download_from_dataset(video_rel)
            if local_video is None:
                continue

            audio_base = ann_rel.replace("/annotation/", "/audio/", 1).rsplit(".", 1)[0]
            local_audio = None
            for ext in gradio_app.AUDIO_EXTENSIONS:
                local_audio = self._download_from_dataset(audio_base + ext)
                if local_audio is not None:
                    break

            muxed = self.mux_video_audio(local_video, local_audio)
            muxed_video_paths.append(self._ensure_servable(muxed))

        return muxed_video_paths


if __name__ == "__main__":
    app = SpaceVideoSearchApp(root_dir="./EmoVid_Data", model_path=MODEL_PATH)
    # Видео для стикеров (без аудио) отдаются прямо из кеша huggingface_hub,
    # поэтому добавляем этот каталог в allowed_paths — иначе Gradio блокирует
    # отдачу файлов вне рабочей папки/temp (InvalidPathError).
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[HF_HUB_CACHE],
    )
