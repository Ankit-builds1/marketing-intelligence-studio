FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user app app
COPY --chown=user src src
COPY --chown=user data data
COPY --chown=user .streamlit .streamlit
COPY --chown=user README.md README.md

EXPOSE 7860

CMD ["python", "-m", "streamlit", "run", "app/budget_optimizer_app.py", "--server.address=0.0.0.0", "--server.port=7860"]
