import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.image

from src.logger import create_logger
from src.loader import load_images, DataSampler
from src.utils import bool_flag
import pdb

# parse parameters
parser = argparse.ArgumentParser(description='Attributes swapping')
parser.add_argument("--model_path", type=str, default="",
                    help="Trained model path")
parser.add_argument("--n_images", type=int, default=1,
                    help="Number of images to modify")
parser.add_argument("--offset", type=int, default=0,
                    help="First image index")
parser.add_argument("--n_interpolations", type=int, default=8,
                    help="Number of interpolations per image")
parser.add_argument("--alpha_min_1", type=float, default=1,
                    help="Min interpolation value for the 1st attribute")
parser.add_argument("--alpha_max_1", type=float, default=1,
                    help="Max interpolation value for the 1st attribute")
parser.add_argument("--alpha_min_2", type=float, default=1,
                    help="Min interpolation value for the 2nd attribute")
parser.add_argument("--alpha_max_2", type=float, default=1,
                    help="Max interpolation value for the 2nd attribute")
parser.add_argument("--plot_size", type=int, default=5,
                    help="Size of images in the grid")
parser.add_argument("--merge_ratio", type=float, default=0.5,
                    help="merge ratio (of the 1st image) [0,1]")

# これいらない
parser.add_argument("--row_wise", type=bool_flag, default=True,
                    help="Represent image interpolations horizontally")

parser.add_argument("--output_path", type=str, default="output.png",
                    help="Output path")
params = parser.parse_args()

# check parameters
assert os.path.isfile(params.model_path)
assert params.n_images >= 1 and params.n_interpolations >= 2

# create logger / load trained model
logger = create_logger(None)
ae = torch.load(params.model_path).eval()

# restore main parameters
params.debug = True
params.batch_size = 32
params.v_flip = False
params.h_flip = False
params.img_sz = ae.img_sz
params.attr = ae.attr
params.n_attr = ae.n_attr

# only even number is accepted
assert params.n_images % 2 == 0

# load dataset
data, attributes = load_images(params)
test_data = DataSampler(data[2], attributes[2], params)

def synthesize(enc_outputs, params):
    """
    obtain the weighted averages of enc_outputs
    """
    assert enc_outputs[-1].size(0) == params.n_images
    synthesized_outputs = [torch.FloatTensor(torch.Size([int(params.n_images/2)]) + \
                                                    enc_outputs[i].size()[1:]) for i in range(len(enc_outputs))]

    
    for i in range(len(synthesized_outputs)):
        for j in range(int(params.n_images/2)):
            # synthesized_outputs[i][j] = (enc_outputs[i][2*j].data * merge_ratio + \
            #                                                     enc_outputs[i][2*j+1].data) / (merge_ratio + 1)
            synthesized_outputs[i][j] = (enc_outputs[i][2*j].data * params.merge_ratio + \
                                                                enc_outputs[i][2*j+1].data * (1 - params.merge_ratio))
    synthesized_outputs = [Variable(tensor).cuda() for tensor in synthesized_outputs]
    return synthesized_outputs


def get_interpolations_2dim(ae, images, attributes, params):
    """
    Reconstruct images / create interpolations two dimensionally
    """
    assert len(images) == len(attributes)
    enc_outputs = ae.encode(images)
    synthesized_outputs = synthesize(enc_outputs, params)
    assert synthesized_outputs[-1].size(0) == params.n_images / 2

    # interpolation values
    alphas_1 = np.linspace(1 - params.alpha_min_1, params.alpha_max_1, params.n_interpolations)
    alphas_2 = np.linspace(1 - params.alpha_min_2, params.alpha_max_2, params.n_interpolations)

    # original image1 / original image2 / interpolations
    outputs = []
    #pdb.set_trace()
    images = images.unsqueeze(1).unsqueeze(2)
    first_images = images[::2]
    second_images = images[1::2]
    outputs.append(torch.cat([first_images] + [Variable(torch.ones(first_images.size()) * 255).cuda()] * (params.n_interpolations - 1), 1))
    #recons = ae.decode(enc_outputs, attributes)[-1].unsqueeze(1).unsqueeze(2)
    outputs.append(torch.cat([second_images] + [Variable(torch.ones(second_images.size()) * 255).cuda()] * (params.n_interpolations - 1), 1))
    #outputs.append(ae.decode(enc_outputs, attributes)[-1])
    #pdb.set_trace()
    for alpha_1 in alphas_1:
        stack = []
        for alpha_2 in alphas_2:
            alpha = torch.FloatTensor([1 - alpha_1, alpha_1, 1 - alpha_2, alpha_2])
            alpha = Variable(alpha.unsqueeze(0).expand((len(images)//2, int(params.n_attr))).cuda())
            stack.append(ae.decode(synthesized_outputs, alpha)[-1])
        #pdb.set_trace()
        outputs.append(torch.cat([x.unsqueeze(1).unsqueeze(1) for x in stack], 1))

    return torch.cat(outputs, 2).data.cpu()

interpolations = []

for k in range(0, params.n_images, 100):
    i = params.offset + k
    j = params.offset + min(params.n_images, k + 100)
    images, attributes = test_data.eval_batch(i, j)
    interpolations.append(get_interpolations_2dim(ae, images, attributes, params))

#interpolations = torch.cat(interpolations, 0)
assert interpolations[0].size() == (params.n_images//2, params.n_interpolations, params.n_interpolations + 2,
                                 3, params.img_sz, params.img_sz)

def get_grid(images, row_wise, plot_size=5):
    """
    Create a grid with all images.
    """
    n_images, n_columns, img_fm, img_sz, _ = images.size()
    if not row_wise:
        images = images.transpose(0, 1).contiguous()
    images = images.view(n_images * n_columns, img_fm, img_sz, img_sz)
    images.add_(1).div_(2.0)
    return make_grid(images, nrow=(n_columns if row_wise else n_images))


# generate the grid / save it to a PNG file
imname, extension = os.path.splitext(params.output_path)
for idx_interpolations in range(len(interpolations)):
    for idx in range(params.n_images//2):
        grid = get_grid(interpolations[idx_interpolations][idx], params.row_wise, params.plot_size)
        matplotlib.image.imsave(imname + '_' + str(params.offset + 2*idx)\
                                + 'and' + str(params.offset + 2*idx + 1) +  extension, grid.numpy().transpose((1, 2, 0)))