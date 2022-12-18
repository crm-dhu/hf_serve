FROM git.infra.igodigital.com:4567/pi/cobaltstarfish/ubuntu:20.04
LABEL description="Marketing Cloud Einstein Kernel hugging face example run"

RUN apt-get update && apt-get -y install sudo python3.8
RUN ln -s /usr/bin/python3.8 /usr/bin/python3
RUN ln -s /usr/bin/python3.8 /usr/bin/python
RUN apt-get -y install python3-pip

# copy files
ENV APP_DIR='/build'
WORKDIR $APP_DIR
COPY . $APP_DIR

RUN groupadd -r -g 2058 app_user && useradd -m -u 2058 -g 2058 app_user -s /bin/bash
RUN chown -R app_user:app_user $APP_DIR

USER app_user
ENV PATH /home/app_user/.local/bin:$PATH

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip --no-cache-dir install -r requirements.txt

ENV PYTHONUNBUFFERED=TRUE

CMD ./start.sh
