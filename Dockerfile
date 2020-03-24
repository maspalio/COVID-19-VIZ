FROM centos:centos8

RUN yum install --assumeyes python3 python3-virtualenv

RUN virtualenv /opt/ve

WORKDIR /root/covid

ADD requirements.txt .

RUN /opt/ve/bin/pip install --requirement requirements.txt

ADD covid.py .
ADD COVID.ipynb .

RUN /opt/ve/bin/jupyter trust COVID.ipynb

CMD ["/opt/ve/bin/jupyter", "notebook", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
