import json
import hashlib
import uuid
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, PayloadSchemaType
from openai import OpenAI
from tqdm import tqdm

import os
from dotenv import load_dotenv

load_dotenv(interpolate=False)

class QdrantCubeIndexer:
    def __init__(self, 
                 qdrant_url: str = "http://localhost:6333",
                 model_name: str = "text-embedding-3-small",
                 api_key: str = ""): 
        
        self.client = QdrantClient(url=qdrant_url)
        self.model_name = model_name
        self.openai_client = OpenAI(
            api_key=api_key,
            base_url="https://api.aitunnel.ru/v1/"
        )
        self.vector_size = 1536

    def setup_collections(self):
        """Создает коллекции для мер и для столбцов"""
        collections = ["genbi_measures", "genbi_columns"]
        for col in collections:
            try:
                self.client.get_collection(col)
                print(f"Коллекция {col} уже существует.")
            except Exception:
                self.client.create_collection(
                    collection_name=col,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
                )
                # Индекс по имени таблицы для быстрой фильтрации/группировки
                self.client.create_payload_index(col, "table_name", PayloadSchemaType.KEYWORD)
                print(f"Коллекция {col} создана.")

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts: 
            return []
        response = self.openai_client.embeddings.create(model=self.model_name, input=texts)
        return [item.embedding for item in response.data]

    def _generate_uuid(self, text: str) -> str:
        return str(uuid.UUID(hashlib.md5(text.encode('utf-8')).hexdigest()))

    def parse_bim(self, file_path: str):
        """Извлекает меры и столбцы из .bim файла"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        measures_to_index = []
        columns_to_index = []
        
        tables = data.get('model', {}).get('tables', [])
        for table in tables:
            t_name = table.get('name')
            if table.get('isHidden'): 
                continue
            
            # Извлекаем меры
            for m in table.get('measures', []):
                if m.get('isHidden'): continue
                measures_to_index.append({
                    "table_name": t_name,
                    "name": m.get('name'),
                    "folder": m.get('displayFolder', 'Базовые'),
                    # Текст для векторизации: объединяем имя и папку для контекста
                    "search_text": f"{m.get('name')}. Категория: {m.get('displayFolder', 'Общее')}"
                })
            
            # Извлекаем столбцы
            for c in table.get('columns', []):
                if c.get('isHidden'): 
                    continue
                columns_to_index.append({
                    "table_name": t_name,
                    "name": c.get('name'),
                    "search_text": f"{c.get('name')}. Таблица: {t_name}"
                })
        
        return measures_to_index, columns_to_index

    def upload_to_qdrant(self, collection_name: str, data: List[Dict], batch_size: int = 50):
        """Загружает данные в Qdrant батчами"""
        print(f"Загрузка {len(data)} объектов в {collection_name}...")
        
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i : i + batch_size]
            texts = [item['search_text'] for item in batch]
            vectors = self._get_embeddings(texts)
            
            points = []
            for j, item in enumerate(batch):
                p_id = self._generate_uuid(f"{collection_name}_{item['table_name']}_{item['name']}")
                points.append(PointStruct(
                    id=p_id,
                    vector=vectors[j],
                    payload=item
                ))
            
            self.client.upsert(collection_name=collection_name, points=points)

    def run(self, bim_path: str):
        self.setup_collections()
        measures, columns = self.parse_bim(bim_path)
        
        if measures:
            self.upload_to_qdrant("genbi_measures", measures)
        if columns:
            self.upload_to_qdrant("genbi_columns", columns)
        
        print("Индексация куба завершена успешно!")

if __name__ == "__main__":
    BIM_FILE = "OLAP_MK.bim"
    indexer = QdrantCubeIndexer(
        api_key=str(os.getenv("API_KEY") or os.getenv("SECRET_KEY") or "")
    )
    indexer.run(BIM_FILE)