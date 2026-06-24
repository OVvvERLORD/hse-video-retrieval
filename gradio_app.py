import gradio as gr
import os
import shutil
import data_saver as ds
import baseline 
import joblib
import subprocess
import uuid
import time
import imageio_ffmpeg

# Расширения, в которых может лежать аудио в датасете (animation -> .wav, movie -> .mp3)
AUDIO_EXTENSIONS = ('.wav', '.mp3')


class VideoSearchApp:
    def __init__(self, root_dir: str = "./EmoVid_Data", model_path: str = 'model_surely_not_overfitted.joblib'):
        """
        Инициализация приложения поиска видео.
        :param root_dir: Корневая директория датасета (например, './EmoVid_Data')
        :param model_path: Путь к файлу модели joblib
        """
        self.root_dir = os.path.abspath(os.path.expanduser(root_dir))
        
        self.temp_dir = os.path.abspath(os.path.join(os.getcwd(), 'temp_muxed_videos'))
        os.makedirs(self.temp_dir, exist_ok=True)

        # Системный ffmpeg, если есть; иначе бинарник из пакета imageio-ffmpeg
        self.ffmpeg_exe = shutil.which('ffmpeg') or imageio_ffmpeg.get_ffmpeg_exe()
        print(f"Используется ffmpeg: {self.ffmpeg_exe}")

        print(f"Загрузка модели из: {model_path}")
        self.model = joblib.load(model_path)
        self.support_model = baseline.SupportModel(svc_model=self.model)
        print("Создание объекта класса DataStorage")
        self.storage = ds.DataStorage(root_dir=self.root_dir, support_model=self.support_model, embedder=self.support_model.emb)

        if len(self.storage.index) == 0:
            print("Индекс пустой, создаём базу данных с нуля...")
            self.storage.scan_new()
            self.storage.embed_pending()

        print(f"Инициализация завершена. Временные файлы будут в: {self.temp_dir}")

    def cleanup_old_temp_files(self, max_age_minutes: int = 5):
        """Удаляет файлы из временной папки, которые старше заданного времени."""
        current_time = time.time()
        deleted_count = 0
        for filename in os.listdir(self.temp_dir):
            filepath = os.path.join(self.temp_dir, filename)
            if os.path.isfile(filepath):
                file_age_seconds = current_time - os.path.getmtime(filepath)
                if file_age_seconds > (max_age_minutes * 60):
                    try:
                        os.remove(filepath)
                        deleted_count += 1
                    except Exception as e:
                        print(f"Не удалось удалить {filepath}: {e}")
        
        if deleted_count > 0:
            print(f"[Cleanup] Удалено {deleted_count} устаревших временных файлов.")

    def find_audio_path(self, annotation_path: str) -> str | None:
        """Подбирает существующий аудиофайл для аннотации, перебирая возможные расширения.
        Возвращает None, если аудио нет (например, у стикеров)."""
        base = annotation_path.replace('/annotation/', '/audio/', 1).rsplit('.', 1)[0]
        for ext in AUDIO_EXTENSIONS:
            candidate = base + ext
            if os.path.exists(candidate):
                return candidate
        return None

    def mux_video_audio(self, video_path: str, audio_path: str | None) -> str:
        """Объединяет видео и аудио в один mp4 файл с помощью FFmpeg.
        Если аудио нет или склейка не удалась — возвращает исходное видео без звука."""
        if not os.path.exists(video_path):
            print(f"Warning: видео не найдено: {video_path}")
            return video_path
        if audio_path is None:
            # У клипа нет аудиодорожки (стикеры) — показываем видео как есть.
            return video_path

        output_filename = f"{uuid.uuid4().hex}.mp4"
        output_path = os.path.join(self.temp_dir, output_filename)

        cmd = [
            self.ffmpeg_exe, '-y',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',      # Копируем видео без перекодирования
            '-c:a', 'aac',       # Кодируем аудио в AAC
            '-shortest',         # Обрезаем по самому короткому потоку
            output_path
        ]

        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return output_path
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
            print(f"FFmpeg error for {video_path}: {e}")
            return video_path

    def process_prompt_and_get_videos(self, prompt: str, num_videos: int):
        """Основная логика обработки запроса пользователя."""
        self.cleanup_old_temp_files(max_age_minutes=5)

        matches = self.storage.search(prompt, num_videos).keys
        df = self.storage.df.set_index('usearch_uid')

        found_annotation_paths = list(df.loc[matches]['file_path'])

        video_paths = [
            p.replace('/annotation/', '/video/', 1).rsplit('.', 1)[0] + '.mp4'
            for p in found_annotation_paths
        ]
        audio_paths = [self.find_audio_path(p) for p in found_annotation_paths]

        muxed_video_paths = [
            self.mux_video_audio(v_path, a_path) 
            for v_path, a_path in zip(video_paths, audio_paths)
        ]
        
        return muxed_video_paths

    def build_interface(self) -> gr.Blocks:
        """Построение Gradio интерфейса."""
        with gr.Blocks(title="Поиск видео по текстовому промпту", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🎬 Поиск видео по текстовому описанию")
            gr.Markdown("Введите описание, и модель подберет подходящие видео из библиотеки.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_input = gr.Textbox(
                        label="Текстовый промпт на английском",
                        placeholder="Например: 'A cute kitten playing with a ball'",
                        lines=3,
                        value="Sweety kitten"
                    )
                    num_videos_input = gr.Slider(
                        minimum=1, maximum=10, step=1, value=3, 
                        label="Количество видео для отображения"
                    )
                    submit_btn = gr.Button("Найти видео", variant="primary")
                    
                with gr.Column(scale=2):
                    video_output = gr.Gallery(
                        label="Результат поиска",
                        columns=2, rows=2, height="auto",
                        object_fit="cover", preview=True
                    )

            submit_btn.click(
                fn=self.process_prompt_and_get_videos,
                inputs=[prompt_input, num_videos_input],
                outputs=[video_output]
            )
        return demo

    def launch(self, **kwargs):
        """
        Запуск приложения.
        :param kwargs: Дополнительные аргументы для demo.launch() (например, share=True, server_port=7860)
        """
        # Полная очистка при старте
        self.cleanup_old_temp_files(max_age_minutes=0)
        
        demo = self.build_interface()
        demo.launch(**kwargs)
