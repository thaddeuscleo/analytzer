FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

EXPOSE 7860

WORKDIR /workspace

COPY . .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install -r requirements.txt

RUN python download.py

ENTRYPOINT [ "/bin/sh", "./scripts/start.sh" ]