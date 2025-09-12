FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN ln -sf python3 /usr/bin/python
RUN ln -sf pip3 /usr/bin/pip
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121
RUN pip install torch==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121
COPY . .
CMD ["python", "main.py"]
