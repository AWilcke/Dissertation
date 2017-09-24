FROM pytorch/pytorch

LABEL maintainer "Arthur Wilcke <arthur.wilcke@gmail.com>"

RUN pip install scikit-image

WORKDIR /home
