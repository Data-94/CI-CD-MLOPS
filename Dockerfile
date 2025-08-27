FROM python:3.9.18-slim-bullseye

#app adında bir klasör oluşturduk
RUN mkdir /app  

#app klasörünü çalışma dizini olarak belirledik
WORKDIR /app 

#Gerekli dosyaları kopyaladık
COPY requirements.txt main.py scaler.pkl IsolationForest_model.pkl /app/

# Gerekli kütüphaneleri yükledik
RUN pip install update pip && pip install -r requirements.txt 

EXPOSE 8000

# Uygulamayı çalıştırmak için gerekli komut
CMD [ "uvicorn", "main:app", "--host=0.0.0.0", "--port=8000" ]