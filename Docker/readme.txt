プロジェクトの開発環境をまとめたdocker fileで、
FaderNetworksや顔画像の切り出しプログラム(openCV, dlib)の動作を目的としたものです。

NVIDIA dockerのバージョン1.xを使っています
ビルドは
sudo docker build -t <my_image_name> .
runは
sudo nvidia-docker run -it <my_image_name> /bin/bash
です

顔画像切り出しのみ使いたい場合はGPU関連のパッケージを除き、ビルドして下さい。

run_jupyter.shを起動するとdocker内でjupyterが起動し、ホストの8888番ポートからのアクセスが可能です。