**HUMAN DETECTION ON RAILWAY TRACKS MODEL**

**AIM AND OBJECTIVES**

**Aim**

To create Human Detection on Railway Tracks model which will detect
Humans crossing on Railway Tracks and then convey the message on the
viewfinder of camera in real time when humans are crossing the Tracks.

**Objectives**

  - The main objective of the project is to create a program which can
    be either run on Jetson nano or any pc with YOLOv5 installed and
    start detecting using the camera module on the device.

  - Using appropriate data sets for recognizing and interpreting data
    using machine learning.

  - To show on the optical viewfinder of the camera module when Humans
    cross Railway Tracks.

**ABSTRACT**

  - Humans are detected when they are crossing the railway tracks and
    then shown on the viewfinder of the camera.

  - Our Team in SparrowAi have completed this project on jetson nano
    which is a very small computational device.

  - A lot of research is being conducted in the field of Computer Vision
    and Machine Learning (ML), where machines are trained to identify
    various objects from one another. Machine Learning provides various
    techniques through which various objects can be detected.

  - One such technique is to use YOLOv5 with Roboflow model, which
    generates a small size trained model and makes ML integration
    easier.

  - More than 13,000 train accidents across the country have killed
    nearly 12,000 railway passengers in 2020, according to a recent
    report by the National Crime Records Bureau (NCRB).

  - The NCRB report also shows that 8,400 or about 70 per cent of these
    railway accidents last year took place as passengers either fell off
    the train or came in the way while crossing the railway track.

**INTRODUCTION**

  - This project is based on Human detection model. We are going to
    implement this project with Machine Learning and this project can be
    even run on jetson nano which our team at SparrowAi has already
    done.

  - This project can also be used to gather information about how many
    people are crossing the Railway Tracks in a given time.

  - Humans can be classified into whether they are standing close to
    railway tracks or are crossing them based on the image annotation we
    give in roboflow.

  - Human detection becomes difficult sometimes on account of various
    conditions like rain, fog, night time making Human Detection harder
    for model to detect. However, training in Roboflow has allowed us to
    crop images and change the contrast of certain images to match the
    time of day, lighting for better recognition by the model.

  - Neural networks and machine learning have been used for these tasks
    and have obtained good results.

  - Machine learning algorithms have proven to be very useful in pattern
    recognition and classification, and hence can be used for Human
    detection on Railway Tracks as well.

**LITERATURE REVIEW**

  - The NCRB report on railway accidents across the country showed that
    Maharashtra ranks first and Uttar Pradesh second, both in terms of
    the number of accidents as well as deaths. Uttar Pradesh topped the
    list ranking the number of train collisions at railway crossings,
    while Bihar and Madhya Pradesh came second and third in this list,
    respectively.

  - The report further detailed that as many as 11,986 railway
    passengers were killed and 11,127 were injured in these accidents
    during the past year. The highest number of train accidents or 20
    per cent of the total were found to be in Maharashtra, while Uttar
    Pradesh followed up at number two with 12 per cent of the total
    number of accidents.

  - Out of the 13,018 recorded train accidents in 2020, as many as 9,117
    of them (70 per cent of the total) took place due to passengers
    falling off the train or coming on its way while crossing the track.
    A total of 8,400 people died in this manner, which is again 70 per
    cent of the total death toll of 11,987. Additionally, Uttar Pradesh
    topped the list in terms of train collisions at railway crossings,
    recording as many as 380 of the 1,014 total accidents reported in
    this manner. Bihar, with 191 train accidents and 144 crossing
    collisions, came second.

  - In Madhya Pradesh, there were a total of 191 train accidents and 144
    collision crossings; the state came third in the list of similar
    accidents. In all, a total of 1,185 people died in train collisions,
    with 561 deaths (47 per cent), in UP, 142 deaths (16 per cent) in
    Bihar, and 191 deaths in Madhya Pradesh.

  - While many fatalities on tracks last year could not be registered
    for some reason or the other, the deaths of 16 migrant workers after
    being run over by a freight train in Maharashtra's Aurangabad last
    May rattled many as they were killed while resting on the tracks
    thinking no train would be coming due to the Covid suspension.

  - The Railways has carried out massive campaigns to reduce such deaths
    and also paid compensation to the family members of the victim on
    sympathetic grounds in some cases. The measures taken by the
    Railways include elimination of unmanned-level crossings over the
    broad gauge network, signal modernization, the use of modern
    machines in maintenance, among others.

  - Our model detects humans crossing Railway tracks and then can inform
    the respective officials about it and then the officials can take
    actions against the trespassers.

**JETSON NANO COMPATIBILITY**

  - The power of modern AI is now available for makers, learners, and
    embedded developers everywhere.

  - NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer
    that lets you run multiple neural networks in parallel for
    applications like image classification, object detection,
    segmentation, and speech processing. All in an easy-to-use platform
    that runs in as little as 5 watts.

  - Hence due to ease of process as well as reduced cost of
    implementation we have used Jetson nano for model detection and
    training.

  - NVIDIA JetPack SDK is the most comprehensive solution for building
    end-to-end accelerated AI applications. All Jetson modules and
    developer kits are supported by JetPack SDK.

  - In our model we have used JetPack version 4.6 which is the latest
    production release and supports all Jetson modules.

**PROPOSED SYSTEM**

1.  Study basics of machine learning and image recognition.

2.  Start with implementation

<!-- end list -->

  - Front-end development

  - Back-end development

<!-- end list -->

3.  Testing, analyzing and improvising the model. An application using
    python and Roboflow and its machine learning libraries will be using
    machine learning to identify a person when he or she is crossing
    Railway tracks illegally.

4.  Use data sets to interpret humans and convey it when they are
    crossing tracks in the viewfinder.

**METHODOLOGY**

The human detection model on Railway tracks is a program that focuses on
implementing real time human detection on Railway tracks.

It is a prototype of a new product that comprises of the main module:

Human detection and then showing on viewfinder when one is crossing
tracks according to data fed.

Human Detection Module

This Module is divided into two parts:

**1. Human Detection**

  - Ability to detect the location of a person in any input image or
    frame. The output is the bounding box coordinates on the detected
    person.

  - For this task, initially the Data set library Kaggle was considered.
    But integrating it was a complex task so then we just downloaded the
    images from google images and made our own data set.

  - This Data set identifies human in a Bitmap graphic object and
    returns the bounding box image with annotation of name present.

**2. Classification Detection**

  - Classification of the human based on when they are crossing Railway
    tracks on the viewfinder.

  - Hence YOLOv5 which is a model library from roboflow for image
    classification and vision was used.

  - There are other models as well but YOLOv5 is smaller and generally
    easier to use in production. Given it is natively implemented in
    PyTorch (rather than Darknet), modifying the architecture and
    exporting and deployment to many environments is straightforward.

**INSTALLATION**

> sudo apt-get remove --purge libreoffice\*
> 
> sudo apt-get remove --purge thunderbird\*
> 
> sudo fallocate -l 10.0G /swapfile1
> 
> sudo chmod 600 /swapfile1
> 
> sudo mkswap /swapfile1
> 
> sudo vim /etc/fstab
> 
> \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#add line\#\#\#\#\#\#\#\#\#\#\#
> 
> /swapfile1 swap defaults 0 0
> 
> vim \~/.bashrc
> 
> \#\#\#\#\#\#\#\#\#\#\#\#\#add line \#\#\#\#\#\#\#\#\#\#\#\#\#
> 
> exportPATH=/usr/local/cuda/bin${PATH:+:${PATH}}
> 
> exportLD\_LIBRARY\_PATh=/usr/local/cuda/lib64${LD\_LIBRARY\_PATH:+:${LD\_LIBRARY\_PATH}}
> 
> exportLD\_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
> 
> sudo apt-get update
> 
> sudo apt-get upgrade
> 
> \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#pip-21.3.1 setuptools-59.6.0
> wheel-0.37.1\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
> 
> sudo apt install curl
> 
> curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
> 
> sudo python3 get-pip.py
> 
> sudo apt-get install libopenblas-base libopenmpi-dev
> 
> sudo apt-get install python3-dev build-essential autoconf libtool
> pkg-config python-opengl python-pil python-pyrex
> python-pyside.qtopengl idle-python2.7 qt4-dev-tools qt4-designer
> libqtgui4 libqtcore4 libqt4-xml libqt4-test libqt4-script
> libqt4-network libqt4-dbus python-qt4 python-qt4-gl libgle3 python-dev
> libssl-dev libpq-dev python-dev libxml2-dev libxslt1-dev libldap2-dev
> libsasl2-dev libffi-dev libfreetype6-dev python3-dev
> 
> vim \~/.bashrc
> 
> \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\# add line
> \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
> 
> exportOPENBLAS\_CORETYPE=ARMV8
> 
> source\~/.bashrc
> 
> sudo pip3 install pillow
> 
> curl -LO
> https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
> 
> mv p57jwntv436lfrd78inwl7iml6p13fzh.whl
> torch-1.8.0-cp36-cp36m-linux\_aarch64.whl
> 
> sudo pip3 install torch-1.8.0-cp36-cp36m-linux\_aarch64.whl
> 
> sudo python3 -c "import torch; print(torch.cuda.is\_available())"
> 
> git clone --branch v0.9.1 https://github.com/pytorch/vision
> torchvision
> 
> cdtorchvision/
> 
> sudo python3 setup.py install
> 
> cd
> 
> git clone https://github.com/ultralytics/yolov5.git
> 
> cdyolov5/
> 
> sudo pip3 install numpy==1.19.4
> 
> history
> 
> \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#comment torch,PyYAML and
> torchvision in
> requirement.txt\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
> 
> sudo pip3 install --ignore-installed PyYAML\>=5.3.1
> 
> sudo pip3 install -r requirements.txt
> 
> sudo python3 detect.py
> 
> sudo python3 detect.py --weights yolov5s.pt --source 0
> 
> \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#Tensorflow\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
> 
> sudo apt-get install python3.6-dev libmysqlclient-dev
> 
> sudo apt install -y python3-pip libjpeg-dev libcanberra-gtk-module
> libcanberra-gtk3-module
> 
> pip3 install tqdm cython pycocotools
> 
> \#\#\#\#\#\#\#\#\#\#\#\#\#
> https://developer.download.nvidia.com/compute/redist/jp/v46/tensorflow/tensorflow-2.5.0%2Bnv21.8-cp36-cp36m-linux\_aarch64.whl
> \#\#\#\#\#\#
> 
> sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev
> zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
> 
> sudo apt-get install python3-pip
> 
> sudo pip3 install -U pip testresources setuptools==49.6.0
> 
> sudo pip3 install -U --no-deps numpy==1.19.4 future==0.18.2
> mock==3.0.5 keras\_preprocessing==1.1.2 keras\_applications==1.0.8
> gast==0.4.0 protobuf pybind11 cython pkgconfig
> 
> sudo env H5PY\_SETUP\_REQUIRES=0 pip3 install -U h5py==3.1.0
> 
> sudo pip3 install -U cython
> 
> sudo apt install python3-h5py
> 
> sudo pip3 install \#install downloaded tensorflow(sudo pip3 install
> --pre --extra-index-url
> https://developer.download.nvidia.com/compute/redist/jp/v46
> tensorflow)
> 
> python3
> 
> import tensorflow as tf
> 
> tf.config.list\_physical\_devices("GPU")
> 
> print(tf.reduce\_sum(tf.random.normal(\[1000,1000\])))
> 
> \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#mediapipe\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
> 
> git clone https://github.com/PINTO0309/mediapipe-bin
> 
> ls
> 
> cdmediapipe-bin/
> 
> ls
> 
> ./v0.8.5/numpy119x/mediapipe-0.8.5\_cuda102-cp36-cp36m-linux\_aarch64\_numpy119x\_jetsonnano\_L4T32.5.1\_download.sh
> 
> ls
> 
> sudo pip3 install
> mediapipe-0.8.5\_cuda102-cp36-none-linux\_aarch64.whl


 
**ADVANTAGES**

  - Deaths due to illegal crossing is major cause of concern in India
    and around the world our model can be used to mitigate this problem
    by keeping a watchful eye on the Railway tracks.

  - Human detection system shows humans crossing Railway tracks in
    viewfinder of camera module with good accuracy.

  - Our model can be used in places where there is less workforce with
    respect to overall population and hence makes the process of
    recognizing humans on tracks more efficient.

  - Human detection on Railway tracks model works completely automated
    and no user input is required.

  - It can work around the clock and therefore becomes more cost
    efficient.

**APPLICATION**

  - Detects humans and then checks whether they are crossing the Railway
    tracks in each image frame or viewfinder using a camera module.

  - Can be used anywhere Railway tracks are laid and also places where
    illegal human Railway tracks crossing is regularly observed.

  - Can be used as a reference for other ai models based on human
    detection on Railway tracks.

**FUTURE SCOPE**

  - As we know technology is marching towards automation, so this
    project is one of the step towards automation.

  - Thus, for more accurate results it needs to be trained for more
    images, and for a greater number of epochs.

  - Human detection on Railway tracks model will become a necessity in
    the future due to rise in population and hence our model will be of
    great help to tackle the situation in an efficient way. As
    population increase means more human crowd and hence increase in
    chance of Railway tracks crossing.

**CONCLUSION**

  - In this project our model is trying to detect a Human and then
    showing it on viewfinder, live as to whether they are crossing a
    Railway track as we have specified in Roboflow.

  - This model tries to solve the problem of people crossing Railway
    tracks illegally and thus reduce the chances of their deaths and
    also any other consequent accident related to Railway tracks
    crossing.

  - The model is efficient and highly accurate and hence works without
    any lag and also if the data is downloaded can be made to work
    offline.

**REFERENCE**

1.  Roboflow:-[<span class="underline">https://roboflow.com/</span>](https://roboflow.com/)

2.  Datasets or images used :- Google images

**ARTICLES**

1.  https://www.hindustantimes.com/india-news/over-13-000-train-accidents-in-2020-32-lives-lost-daily-on-average-ncrb-report-101635646106710.html

2.  https://www.ndtv.com/india-news/railways-8-700-people-died-on-tracks-in-2020-despite-reduced-passenger-train-services-2454739
