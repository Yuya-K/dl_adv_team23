FROM nvidia/cuda:8.0-cudnn6-runtime

RUN apt-get update

#utils
RUN apt-get install -y curl wget git unzip imagemagick bzip2 

#build tools
RUN apt-get install -y build-essential cmake libgtk2.0-dev pkg-config
#libraries
RUN apt-get install -y python-dev libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev libsm6 libxext6 libx11-dev

#BLAS
RUN apt install -y libatlas-base-dev libatlas-doc libopenblas-base libopenblas-dev

#####################################################
#these packages should be installed via conda
#####################################################

#openCV
#RUN apt-get install python-dev libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
#RUN git clone https://github.com/opencv/opencv.git
#RUN git clone https://github.com/opencv/opencv_contrib.git
#RUN cd ~/opencv &&\
#RUN mkdir build &&\
#RUN cd build &&\
#cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..

##dlib
#boost.python
#RUN wget https://dl.bintray.com/boostorg/release/1.66.0/source/boost_1_66_0.tar.bz2
#RUN tar --bzip2 -xf boost_1_66_0.tar.bz2
#RUN cd ./boost_1_66_0 &&\
#	./bootstrap.sh --with-libraries=python
#RUN ./b2

#dlib
#RUN apt-get install libx11-dev libatlas-base-dev libgtk-3-dev libboost-python-dev libpython-all-dev
#RUN wget http://dlib.net/files/dlib-19.8.tar.bz2
#RUN tar xvf dlib-19.8.tar.bz2
#RUN cd dlib-19.6/ &&\
#	mkdir build &&\
#	cd build &&\
#	cmake .. &&\
#	cmake --build . --config Release &&\
#	make install &&\
#	ldconfig &&\

####################################################


#prepare pyenv
RUN git clone git://github.com/yyuu/pyenv.git .pyenv

WORKDIR /
ENV HOME  /
ENV PYENV_ROOT /.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN pyenv install anaconda3-5.0.1
RUN pyenv global anaconda3-5.0.1
RUN pyenv rehash

#tensorflow gpu: for cuda8
RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp36-cp36m-linux_x86_64.whl
RUN pip install tensorflow-gpu

#chainer, cv, keras ...
RUN pip install opencv-python tqdm h5py keras kaggle-cli gym
RUN pip install chainer chainercv

#pytorch
RUN pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl
RUN pip install torchvision

#boost, dlib: these commands should be modified depends on version or repository change
RUN conda install -c conda-forge boost
RUN conda install -c conda-forge dlib

RUN pip install imutils

RUN pip --no-cache-dir install --upgrade ipython && \
	pip --no-cache-dir install \
		Cython \
		ipykernel \
		jupyter \
		path.py \
		Pillow \
		pygments \
		six \
		sphinx \
		wheel \
		zmq \
		&& \
	python -m ipykernel.kernelspec

#setup jupyter
#from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.gpu
#launch with -p 8888:8888

COPY jupyter_notebook_config.py /root/.jupyter/
COPY run_jupyter.sh /home/

EXPOSE 8888

WORKDIR "/home"
CMD ["/bin/bash"]
