�v���W�F�N�g�̊J�������܂Ƃ߂�docker file�ŁA
FaderNetworks���摜�̐؂�o���v���O����(openCV, dlib)�̓����ړI�Ƃ������̂ł��B

NVIDIA docker�̃o�[�W����1.x���g���Ă��܂�
�r���h��
sudo docker build -t <my_image_name> .
run��
sudo nvidia-docker run -it <my_image_name> /bin/bash
�ł�

��摜�؂�o���̂ݎg�������ꍇ��GPU�֘A�̃p�b�P�[�W�������A�r���h���ĉ������B

run_jupyter.sh���N�������docker����jupyter���N�����A�z�X�g��8888�ԃ|�[�g����̃A�N�Z�X���\�ł��B