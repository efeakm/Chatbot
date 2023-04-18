FROM python:3.9.16-slim
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip3 install torch==1.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 80
CMD ["python","app.py"]
