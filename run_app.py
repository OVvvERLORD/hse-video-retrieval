from gradio_app import VideoSearchApp

TARGET_ROOT_DIR = './EmoVid_Data/'
MODEL_PATH = 'model_surely_not_overfitted.joblib'

app = VideoSearchApp(root_dir=TARGET_ROOT_DIR, model_path=MODEL_PATH)
app.launch(share=True)
