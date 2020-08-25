FROM nvcr.io/nvidia/pytorch:20.07-py3

RUN pip3 install -r requirements.txt

CMD [ "/bin/bash" ]