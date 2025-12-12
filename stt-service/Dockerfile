FROM landsoftdocker/nuclearchat:latest

# Whisper만 추가 설치 (1분)
RUN pip3 install --no-cache-dir faster-whisper==1.0.3

WORKDIR /app/stt
COPY app.py entrypoint.sh /app/stt/
RUN chmod +x /app/stt/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/app/stt/entrypoint.sh"]