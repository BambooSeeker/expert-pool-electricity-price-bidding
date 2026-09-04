FROM python:3.11-slim

WORKDIR /workspace
COPY environment/requirements-lock.txt /workspace/environment/requirements-lock.txt
RUN python -m pip install --no-cache-dir -r environment/requirements-lock.txt
COPY . /workspace
CMD ["python", "run_demo.py"]
