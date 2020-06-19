FROM python:3.6-slim-stretch

RUN apt-get update
RUN apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

COPY . /CRAFT
RUN cd CRAFT && \
    pip3 install -r requirements.txt

WORKDIR /CRAFT

# CMD ["python", "manage.py", "db", "init"] 
# CMD ["python", "manage.py", "db", "migrate"] 
# CMD ["python", "manage.py", "db", "upgrade"]

#CMD ["gunicorn", "-b", "0.0.0.0:5555", "--workers=5", "--threads=2", "wsgi:app"]
# CMD ["python", "run_app.py"]