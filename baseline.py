import os
import re
from typing import Union, Optional, List

import numpy as np

# Тяжёлые зависимости импортируются лениво внутри классов, чтобы можно было
# пользоваться только одним из бэкендов (w2v ИЛИ bert), не устанавливая второй.

# Классы эмоций в том же порядке, в котором обучалась голова-классификатор:
# sorted(set(labels)). Индекс выхода головы (логит i) соответствует позиции
# эмоции в этом списке. Совпадает с полем `emotion` в metadata.parquet.
EMOTION_CLASSES = [
    "amusement", "anger", "awe", "contentment",
    "disgust", "excitement", "fear", "sadness",
]


class W2VSentenceEmbedder:
    """
    Эмбеддер предложений на базе Word2Vec.
    Инкапсулирует препроцессинг, усреднение векторов слов и нормализацию.
    Совместим с DataStorage через протокол Callable + атрибут .vector_size.
    """
    def __init__(self, model: str = "word2vec-google-news-300"):
        import gensim.downloader
        from gensim.models import KeyedVectors

        if os.path.exists(model):
            self.w2v = KeyedVectors.load_word2vec_format(model, binary=True)
        else:
            self.w2v = gensim.downloader.load(model)

        self.vector_size = self.w2v.vector_size

    def __call__(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Позволяет вызывать объект как функцию: embedder("text") или embedder(["text1", "text2"])
        """
        if isinstance(text, str):
            return self.embed_single(text)
        elif isinstance(text, list):
            return self.embed_batch(text)
        else:
            raise TypeError("Ожидается str или List[str]")

    def _preprocess(self, sentences: List[str]) -> List[List[np.ndarray]]:
        '''
        Не очень оопшно, но у нас есть вектор из промптов,
        из которых мы разбивааем на вектор из слов, затем каждое корректное
        слово с точки зрения w2v мы закидываем в сам w2v, то есть для каждого
        промпта ("пользовательского запроса") мы получаем вектор из эмбеддингов каждого допустимого
        слова предложения.
        '''
        new_x = []
        for sentence in sentences:
            words = re.findall(r'\w+', sentence.lower())
            vectors = [self.w2v[word] for word in words if word in self.w2v]
            new_x.append(vectors)
        return new_x

    def _merge(self, vectors_list: List[List[np.ndarray]]) -> List[np.ndarray]:
        '''
        Нам приходит вектор, каждый элемент которого является массивом
        эмбеддингов для какого-то предложения. Мы складываем все вектора-
        эмбеддинги одного предложения и нормализуем их, надеясь, что таким образом сохраним
        смысл всего предложения. Нормализация нужна для того, чтобы мы
        не зависили от количества слов в предложении.
        '''
        merged = []
        for vecs in vectors_list:
            if not vecs:
                merged.append(np.zeros(self.vector_size, dtype=np.float32))
                continue

            shared = np.sum(vecs, axis=0)
            norm = np.linalg.norm(shared)

            # Нормализуем и гарантируем float32 (требование usearch)
            merged.append((shared / norm if norm > 0 else shared).astype(np.float32))
        return merged

    def embed_single(self, text: str) -> np.ndarray:
        return self._merge(self._preprocess([text]))[0]

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        return self._merge(self._preprocess(texts))


class SBERTEmbedder:
    """
    Эмбеддер предложений на базе fine-tuned SentenceTransformer
    (sentence-transformers/all-mpnet-base-v2, дообученный на эмоциях).
    Возвращает L2-нормированные векторы размерности 768.
    Совместим с DataStorage через протокол Callable + атрибут .vector_size.
    """

    def __init__(self, model_path: str, device: Optional[str] = None, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_path, device=device)
        self.vector_size = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size

    def __call__(self, text: Union[str, List[str]]) -> np.ndarray:
        """embedder("text") -> 1D ndarray; embedder(["t1", "t2"]) -> 2D ndarray."""
        if isinstance(text, str):
            return self.embed_single(text)
        elif isinstance(text, list):
            return self.embed_batch(text)
        raise TypeError("Ожидается str или List[str]")

    def embed_single(self, text: str) -> np.ndarray:
        vec = self.model.encode(
            text, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec.astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts, batch_size=self.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        return vecs.astype(np.float32)


class SupportModel:
    """
    Связка W2V-эмбеддер + SVC-классификатор эмоций (исходный бэкенд).
    DataStorage использует у неё: .emb, .classes_, .predict_proba(...).
    """
    emb: 'W2VSentenceEmbedder'
    vector_size: int

    def __init__(self, svc_model=None, emb_model: 'str | W2VSentenceEmbedder' = 'word2vec-google-news-300'):
        from sklearn import svm

        if isinstance(emb_model, str):
            self.emb = W2VSentenceEmbedder(emb_model)
        else:
            self.emb = emb_model

        self._is_fitted = False
        if svc_model is not None:
            self.svc = svc_model
            self._is_fitted = True
        else:
            self.svc = svm.SVC(probability=True)

    @property
    def classes_(self) -> np.ndarray:
        return self.svc.classes_

    def fit(self, X: Union[str, List[str]], y: List, **kwargs):
        X_embedded = self.emb(X)
        self.svc.fit(X_embedded, y, **kwargs)
        self._is_fitted = True
        return self

    def predict(self, sentence: List[str]):
        if not self._is_fitted:
            raise Exception('Model is not fitted yet!')

        if not isinstance(sentence, list):
            raise Exception("You have to provide a list even if you want to " \
            "predict only one sentence. Uncomfortable, but it is what it is.")

        sentence_embedded = self.emb(sentence)
        return self.svc.predict(sentence_embedded)

    def predict_proba(self, sentence: List[str]):
        if not self._is_fitted:
            raise Exception('Model is not fitted yet!')

        if not isinstance(sentence, list):
            raise Exception("You have to provide a list even if you want to " \
            "predict only one sentence. Uncomfortable, but it is what it is.")

        sentence_embedded = self.emb(sentence)
        if isinstance(sentence_embedded, np.ndarray) and sentence_embedded.ndim == 1:
            sentence_embedded = sentence_embedded.reshape(1, -1)
        return self.svc.predict_proba(sentence_embedded)


class LinearClassifier:
    """
    Голова-классификатор эмоций над эмбеддингами SBERT (768 -> 8 классов).
    Архитектура и имена слоёв ДОЛЖНЫ совпадать с обучающим ноутбуком, иначе
    state_dict из emotion_classifier_head_*.pt не загрузится.

    Объявлена как фабрика, чтобы не тянуть torch на уровне модуля.
    """
    def __new__(cls, *args, **kwargs):
        import torch.nn as nn

        class _LinearClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin1 = nn.Linear(768, 256)
                self.ln1 = nn.LayerNorm(256)
                self.act1 = nn.ReLU()
                self.drop1 = nn.Dropout(0.3)

                self.lin2 = nn.Linear(256, 48)
                self.ln2 = nn.LayerNorm(48)
                self.act2 = nn.ReLU()
                self.drop2 = nn.Dropout(0.2)

                self.lin3 = nn.Linear(48, 8)

            def forward(self, x):
                x = self.drop1(self.act1(self.ln1(self.lin1(x))))
                x = self.drop2(self.act2(self.ln2(self.lin2(x))))
                return self.lin3(x)

        return _LinearClassifier()


class SBERTSupportModel:
    """
    Связка SBERT-эмбеддер + обученная torch-голова-классификатор эмоций.
    Альтернатива SupportModel (W2V+SVC). Предоставляет тот же интерфейс,
    который использует DataStorage: .emb, .classes_, .predict_proba(...).
    """
    emb: 'SBERTEmbedder'
    classes_: np.ndarray

    def __init__(
            self,
            classifier_path: str,
            embedder: Union[str, SBERTEmbedder],
            classes: Optional[List[str]] = None,
            device: Optional[str] = None,
    ):
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(embedder, str):
            self.emb = SBERTEmbedder(embedder, device=self.device)
        else:
            self.emb = embedder

        self.classes_ = np.array(classes if classes is not None else EMOTION_CLASSES)

        self.head = LinearClassifier().to(self.device)
        state = torch.load(classifier_path, map_location=self.device)
        self.head.load_state_dict(state)
        self.head.eval()

    def predict_proba(self, sentences: List[str]) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        if not isinstance(sentences, list):
            raise TypeError(
                "Передайте список строк, даже если предсказываете одно предложение."
            )
        emb = np.asarray(self.emb(sentences), dtype=np.float32)
        if emb.ndim == 1:
            emb = emb[None, :]
        with torch.no_grad():
            logits = self.head(torch.from_numpy(emb).to(self.device))
            return F.softmax(logits, dim=1).cpu().numpy()

    def predict(self, sentences: List[str]) -> np.ndarray:
        proba = self.predict_proba(sentences)
        return self.classes_[np.argmax(proba, axis=1)]
