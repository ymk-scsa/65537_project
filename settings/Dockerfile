FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依存パッケージのインストール
COPY requirements.txt /app/settings/requirements.txt
RUN pip install --no-cache-dir -r /app/settings/requirements.txt

# ソースコードのコピー
COPY . /app

# エントリーポイント
ENTRYPOINT ["python", "main.py"]
