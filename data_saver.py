import numpy as np
import pandas as pd
from usearch.index import Index
from typing import Callable, Union, Literal, List, Any
from tqdm import tqdm
from baseline import SupportModel, W2VSentenceEmbedder
import os
import json
import pathlib
import pyarrow as pa
import pyarrow.parquet as pq

_META_SCHEMA = pa.schema([
    ("usearch_uid", pa.int64()),
    ("uid",         pa.string()),
    ("file_path",   pa.string()),
    ("style",       pa.string()),
    ("emotion",     pa.string()),
    ("caption",     pa.string()),
    ("brightness",  pa.float64()),
    ("colorfulness",pa.float64()),
    ("hue",         pa.float64()),
    ("duration",    pa.float64()),
    ("status",      pa.string()),
])

def _infer_embedder_dim(embedder: Any) -> int:
    """
    Универсальный способ узнать размерность вектора.
    Проверяет стандартные атрибуты -> если нет, запускает тестовый инференс.
    """
    if hasattr(embedder, 'vector_size'):
        return int(embedder.vector_size)          # gensim
    if hasattr(embedder, 'get_sentence_embedding_dimension'):
        return int(embedder.get_sentence_embedding_dimension())  # sentence-transformers
    if hasattr(embedder, 'dim'):
        return int(embedder.dim)                  # кастомные классы

    try:
        dummy_vec = embedder("test_inference")
        if isinstance(dummy_vec, np.ndarray) and dummy_vec.ndim == 1:
            return int(dummy_vec.shape[0])
    except Exception as e:
        raise RuntimeError(f"Не удалось вывести размерность эмбеддера. Убедитесь, что он возвращает 1D numpy.ndarray. Ошибка: {e}")

    raise ValueError("Эмбеддер должен возвращать 1D numpy.ndarray или иметь атрибут .dim/.vector_size")
    
class DataStorage:
    def __init__(
            self,
            root_dir: str,
            meta_path: str = "metadata.parquet",
            index_path: str = "vectors.usearch",
            metric: Union[Literal["cos", "l2", "ip"], Callable] = "cos",
            embedder: Callable = None,
            batch_size: int = 64,
            connectivity: int = 16,
            expansion_add: int = 128,
            expansion_search: int = 64,
            support_model : SupportModel = None
            ):
        """
        Инициализирует хранилище данных.
        Загружает существующую таблицу метаданных и векторный индекс с диска
        или создаёт их с нуля. Автоматически определяет размерность эмбеддинга
        и настраивает параметры графа Usearch.
        """
        self.root_dir = root_dir
        self.meta_path = meta_path
        self.index_path = index_path
        self.batch_size = batch_size

        if embedder is None:
            self.embedder = W2VSentenceEmbedder()
        else:
            self.embedder = embedder

        self.dim = _infer_embedder_dim(self.embedder)

        if os.path.exists(self.meta_path):
            self.df = pd.read_parquet(self.meta_path)
        else:
            self.df = pd.DataFrame(columns=[
                "usearch_uid", "uid", "file_path", "style", "emotion", "caption",
                "brightness", "colorfulness", "hue", "duration",
                "status"
            ])

        valid_ids = self.df["usearch_uid"].dropna()
        self._next_id = int(valid_ids.max()) + 1 if len(valid_ids) > 0 else 0

        def _new_index() -> Index:
            return Index(
                ndim=self.dim,
                metric=metric,
                connectivity=connectivity,
                expansion_add=expansion_add,
                expansion_search=expansion_search
            )

        if os.path.exists(self.index_path):
            self.index = Index()
            self.index.load(self.index_path)
            if self.index.ndim != self.dim:
                # Размерность на диске не совпадает с текущим эмбеддером
                # (например, переключились с W2V (300) на SBERT (768)).
                # Пересоздаём индекс с нуля и помечаем все записи как pending,
                # чтобы embed_pending() пересчитал векторы новым эмбеддером.
                print(
                    f"Индекс на диске имеет dim={self.index.ndim}, а эмбеддер выдаёт "
                    f"dim={self.dim}. Пересоздаю векторный индекс с нуля..."
                )
                os.remove(self.index_path)
                self.index = _new_index()
                if "status" in self.df.columns and len(self.df) > 0:
                    self.df["status"] = "pending"
                    self._save_meta()
        else:
            self.index = _new_index()

        if support_model is None:
            self.support_model = SupportModel()
        else:
            self.support_model = support_model 

    def scan_new(self, batch_size: int = 5000):
        """
        Рекурсивно сканирует целевую директорию на наличие JSON-файлов.
        Новые записи (по uid) потоково дописываются на диск через ParquetWriter,
        полная таблица в память во время сканирования НЕ загружается.
        Возвращает: int — количество успешно добавленных новых записей.
        """
    
        if os.path.exists(self.meta_path):
            existing_uids = set(
                pq.read_table(self.meta_path, columns=["uid"])
                  .column("uid").to_pylist()
            )
        else:
            existing_uids = set()

        def _to_float(v):
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        def _to_str(v):
            return v if (v is None or isinstance(v, str)) else str(v)

        def _write_chunked(writer, table, chunk):
            # пишем таблицу row-group'ами фиксированного размера,
            # чтобы будущие сканы тоже стримились дёшево
            for off in range(0, table.num_rows, chunk):
                writer.write_table(table.slice(off, chunk))

        total_added = 0
        batch = []
        tmp_path = self.meta_path + ".tmp"
        writer = pq.ParquetWriter(tmp_path, _META_SCHEMA, compression="zstd")

        def _flush(rows):
            writer.write_table(pa.Table.from_pylist(rows, schema=_META_SCHEMA))

        try:
            if os.path.exists(self.meta_path):
                pf = pq.ParquetFile(self.meta_path)
                for i in range(pf.num_row_groups):
                    rg = pf.read_row_group(i)
                    try:
                        rg = rg.cast(_META_SCHEMA)
                    except Exception:
                        d = rg.to_pandas().reindex(
                            columns=[f.name for f in _META_SCHEMA])
                        for c in ("brightness", "colorfulness", "hue", "duration"):
                            d[c] = pd.to_numeric(d[c], errors="coerce")
                        d["usearch_uid"] = pd.to_numeric(
                            d["usearch_uid"], errors="coerce").astype("Int64")
                        rg = pa.Table.from_pandas(
                            d, schema=_META_SCHEMA, preserve_index=False)
                    _write_chunked(writer, rg, batch_size)

            for p in tqdm(pathlib.Path(self.root_dir).rglob("*.json"),
                          desc="Сканирование"):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        raw = json.load(f)

                    data = {k.strip(): v for k, v in raw.items()}
                    uid = str(data.get("video_id", p.stem)).strip()

                    if uid in existing_uids:
                        continue

                    batch.append({
                        "usearch_uid": self._next_id,
                        "uid": uid,
                        "file_path": str(p),
                        "style": _to_str(data.get("style")),
                        "emotion": _to_str(data.get("emotion")),
                        "caption": _to_str(data.get("caption")),
                        "brightness": _to_float(data.get("brightness")),
                        "colorfulness": _to_float(data.get("colorfulness")),
                        "hue": _to_float(data.get("hue")),
                        "duration": _to_float(data.get("duration")),
                        "status": "pending",
                    })
                    self._next_id += 1
                    existing_uids.add(uid)

                    if len(batch) >= batch_size:
                        _flush(batch)
                        total_added += len(batch)
                        batch.clear()

                except Exception as e:
                    print(f"Ошибка {p}: {e}")

            if batch:
                _flush(batch)
                total_added += len(batch)
                batch.clear()
        finally:
            writer.close()

        os.replace(tmp_path, self.meta_path)
        self.df = pd.read_parquet(self.meta_path)

        if total_added:
            print(f"+{total_added} новых записей в метаданных")
        return total_added


    def embed_pending(self):
        """
        Генерирует векторные представления для всех записей со статусом 'pending'.
        Добавляет полученные векторы в индекс Usearch батчами и обновляет статус
        записей на 'ready'. Сохраняет изменения на диск.
        Возвращает: int — количество обработанных записей.
        """
        pending_mask = self.df["status"] == "pending"

        if not pending_mask.any():
            return 0
        
        pending_df = self.df[pending_mask].copy()
        uids = pending_df["usearch_uid"].tolist()
        texts = pending_df["caption"].tolist()

        for i in tqdm(range(0, len(texts), self.batch_size), desc="Эмбеддинг"):
            batch_uids = uids[i:i+self.batch_size]
            batch_texts = texts[i:i+self.batch_size]
            
            # Эмбеддим батч одним вызовом: SBERT кодирует список эффективнее,
            # чем по одной строке; W2V-эмбеддер тоже принимает список.
            vectors = np.asarray(self.embedder(batch_texts), dtype=np.float32)

            self.index.add(batch_uids, vectors)
            self.df.loc[self.df["usearch_uid"].isin(batch_uids), "status"] = "ready"

        self._save_all()
        return len(pending_df)


    def search(self, prompt: str, K : int):
        """
        Выполняет поиск похожих записей по текстовому запросу.
        Преобразует запрос в вектор, ищет ближайшие соседи в индексе
        и возвращает результат в виде отфильтрованного DataFrame.
        K - количество возвращаемых результатов.
        """
        from types import SimpleNamespace

        X = self.embedder(prompt)

        probs = self.support_model.predict_proba([prompt])[0]
        top_3_probs = np.argsort(probs)[-3:][::-1]
        expected_classes = self.support_model.classes_[top_3_probs]

        valid_uids = set(
            self.df[self.df['emotion'].isin(expected_classes)]['usearch_uid']
            .dropna().astype(np.uint64).tolist()
        )

        search_count = min(len(self.index), max(K * 5, K))
        if search_count == 0:
            return SimpleNamespace(keys=np.array([], dtype=np.uint64))

        results = self.index.search(X, count=search_count)
        filtered = np.array([k for k in results.keys if k in valid_uids][:K], dtype=np.uint64)
        return SimpleNamespace(keys=filtered)


    def _save_meta(self):
        """
        Сохраняет DataFrame с метаданными в Parquet-файл с ZSTD-сжатием.
        """
        self.df.to_parquet(self.meta_path, index=False, compression="zstd")

    def _save_all(self):
        """
        Атомарно сохраняет векторный индекс Usearch и таблицу метаданных на диск.
        """
        self.index.save(self.index_path)
        self._save_meta()

    def run_pipeline(self):
        """
        Запускает полный цикл обработки данных:
        1. Сканирование директории на новые файлы.
        2. Генерация эмбеддингов для новых записей.
        3. Выполнение тестового поиска по демонстрационному запросу.
        """
        self.scan_new()
        self.embed_pending()
        print("\nДемо-поиск: 'happy man in blue shirt'")
        print(self.search("happy man in blue shirt", top_k=3))