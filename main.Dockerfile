FROM python:slim-buster

# Set environment variables
# ENV PIP_DISABLE_PIP_VERSION_CHECK 1
# ENV PYTHONDONTWRITEBYTECODE 1
# ENV PYTHONUNBUFFERED 1
# ENV PYTHONBUFFERED 1
WORKDIR /app

RUN apt-get update \
  # # dependencies for building Python packages
  # && apt-get install -y build-essential netcat \
  # # psycopg2 dependencies
  # && apt-get install -y libpq-dev \
  # && apt-get install -y gcc \
  # && apt-get install -y default-libmysqlclient-dev \
  # && apt install -y libmariadb-dev-compat libmariadb-dev \
  # # Translations dependencies
  # && apt-get install -y gettext \
  # cleaning up unused files
  && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

COPY . .

# RUN python manage.py collectstatic --noinput --clear
# RUN python manage.py collectstatic --noinput
# RUN python manage.py makemigrations app
# RUN python manage.py makemigrations
# RUN python manage.py migrate

# RUN chmod +x ./start
