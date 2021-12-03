from torch import nn

import torch.nn.functional as F
import torch

from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from natsort import natsorted

def get_file_list(data_dir, suffix=""):
    file_list = []

    for dirpath, _, filenames in os.walk(data_dir):
        for filename in filenames:
            if suffix in filename:
                file_list.append(os.path.join(dirpath, filename))

    file_list = natsorted(file_list)

    return file_list

KEY_VARIANCE = {17: 18.965496063232422, 84: 18.238582611083984, 16: 17.62625503540039, 314: 17.342655181884766, 85: 17.177982330322266, 315: 16.50968360900879, 15: 16.047866821289062, 181: 15.401814460754395, 86: 15.401474952697754, 14: 15.25156021118164, 316: 14.916162490844727, 87: 14.875166893005371, 180: 14.537986755371094, 317: 14.533039093017578, 405: 13.683375358581543, 179: 13.18199348449707, 404: 13.172924995422363, 178: 13.024672508239746, 18: 12.389984130859375, 402: 12.287979125976562, 403: 12.233527183532715, 83: 11.809043884277344, 313: 11.569454193115234, 91: 11.24007511138916, 90: 10.997507095336914, 200: 10.511945724487305, 88: 10.414156913757324, 89: 10.384500503540039, 201: 9.809059143066406, 199: 9.693489074707031, 421: 9.59658145904541, 320: 9.55312728881836, 318: 9.529691696166992, 321: 9.519864082336426, 152: 9.518776893615723, 175: 9.441099166870117, 319: 9.315605163574219, 182: 9.280145645141602, 406: 8.966537475585938, 377: 8.871063232421875, 148: 8.866243362426758, 208: 8.824270248413086, 428: 8.686577796936035, 171: 8.640405654907227, 396: 8.587196350097656, 95: 7.947456359863281, 96: 7.790765762329102, 77: 7.757366180419922, 400: 7.5607757568359375, 146: 7.51237154006958, 176: 7.385627746582031, 194: 7.3249640464782715, 418: 7.113580703735352, 324: 7.106655120849609, 369: 7.013779640197754, 140: 6.886379718780518, 325: 6.788877010345459, 32: 6.746204376220703, 262: 6.677685737609863, 307: 6.592177867889404, 106: 6.362393379211426, 375: 6.215060234069824, 335: 6.144772529602051, 378: 6.065235137939453, 149: 5.690601348876953, 395: 5.456660270690918, 204: 5.203995704650879, 170: 5.194314956665039, 62: 5.111909866333008, 78: 5.110922813415527, 76: 5.086575508117676, 424: 5.080007553100586, 61: 5.026874542236328, 431: 4.981131076812744, 211: 4.964018821716309, 379: 4.698431968688965, 308: 4.418610572814941, 292: 4.409973621368408, 306: 4.324930191040039, 150: 4.229571342468262, 43: 4.225931167602539, 291: 4.222723960876465, 394: 4.145513534545898, 273: 4.071269989013672, 169: 3.8565611839294434, 183: 3.514094352722168, 184: 3.5088438987731934, 430: 3.4743614196777344, 365: 3.446742534637451, 185: 3.4464504718780518, 202: 3.4442028999328613, 210: 3.3955860137939453, 191: 3.3844614028930664, 422: 3.373734474182129, 136: 2.963453769683838, 364: 2.9446029663085938, 407: 2.8978986740112305, 408: 2.84171199798584, 415: 2.8094229698181152, 409: 2.7516589164733887, 287: 2.6652674674987793, 57: 2.6370229721069336, 397: 2.6083054542541504, 135: 2.5998363494873047, 40: 2.3424015045166016, 74: 2.305135726928711, 42: 2.2818500995635986, 288: 2.213219165802002, 80: 2.2062857151031494, 172: 2.1851091384887695, 367: 2.1739962100982666, 212: 2.151231050491333, 432: 2.1447606086730957, 434: 2.1008245944976807, 214: 1.9824650287628174, 270: 1.9380097389221191, 361: 1.9151620864868164, 310: 1.9064078330993652, 272: 1.890311598777771, 304: 1.8824903964996338, 410: 1.8465478420257568, 138: 1.8052912950515747, 58: 1.7760241031646729, 186: 1.7218245267868042, 81: 1.6777923107147217, 435: 1.6668847799301147, 41: 1.661131501197815, 13: 1.654787302017212, 73: 1.641161561012268, 39: 1.6315116882324219, 323: 1.629990577697754, 82: 1.6037687063217163, 312: 1.5759391784667969, 416: 1.5604535341262817, 12: 1.5513306856155396, 311: 1.528236746788025, 38: 1.514960527420044, 11: 1.508102536201477, 269: 1.4972316026687622, 268: 1.4851301908493042, 271: 1.4836475849151611, 72: 1.4730554819107056, 303: 1.4667319059371948, 302: 1.4620106220245361, 132: 1.4454379081726074, 401: 1.413730263710022, 0: 1.4104417562484741, 386: 1.4076260328292847, 267: 1.384002447128296, 454: 1.3788859844207764, 37: 1.3706142902374268, 192: 1.3439428806304932, 433: 1.3050451278686523, 436: 1.2732532024383545, 215: 1.2578562498092651, 216: 1.249117374420166, 387: 1.245216965675354, 93: 1.1597611904144287, 322: 1.1582136154174805, 366: 1.1265974044799805, 385: 1.1165144443511963, 356: 1.1135244369506836, 427: 1.0405446290969849, 213: 1.003720760345459}
_KEY_IDX = [17, 84, 16, 314, 85, 315, 15, 181, 86, 14, 316, 87, 180, 317, 405, 179, 404, 178, 18, 402, 403, 83, 313, 91, 90, 200, 88, 89, 201, 199, 421, 320, 318, 321, 152, 175, 319, 182, 406, 377, 148, 208, 428, 171, 396, 95, 96, 77, 400, 146, 176, 194, 418, 324, 369, 140, 325, 32, 262, 307, 106, 375, 335, 378, 149, 395, 204, 170, 62, 78, 76, 424, 61, 431, 211, 379, 308, 292, 306, 150, 43, 291, 394, 273, 169, 183, 184, 430, 365, 185, 202, 210, 191, 422, 136, 364, 407, 408, 415, 409, 287, 57, 397, 135, 40, 74, 42, 288, 80, 172, 367, 212, 432, 434, 214, 270, 361, 310, 272, 304, 410, 138, 58, 186, 81, 435, 41, 13, 73, 39, 323, 82, 312, 416, 12, 311, 38, 11, 269, 268, 271, 72, 303, 302, 132, 401, 0, 386, 267, 454, 37, 192, 433, 436, 215, 216, 387, 93, 322, 366, 385, 356, 427, 213]
KEY_IDX = _KEY_IDX[:]
STABLE_IDX = [(196, 0.041146062314510345), (419, 0.041638121008872986), (174, 0.04262330010533333), (122, 0.045432206243276596), (188, 0.04641878977417946), (197, 0.048861853778362274), (399, 0.048883505165576935), (168, 0.04902322590351105), (236, 0.04929608106613159), (6, 0.0516219325363636), (3, 0.05169089883565903), (351, 0.05251950025558472), (456, 0.05543915927410126), (248, 0.056157588958740234), (412, 0.05668545514345169), (114, 0.05828186497092247), (195, 0.05984281748533249), (217, 0.061995554715394974), (343, 0.07085694372653961), (51, 0.07101800292730331)]

def mix_mesh_tensor(target, source):
    res = torch.tensor(source).to(source.device)
    res[KEY_IDX] = target[KEY_IDX]
    return res
 
def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['value']

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm2d(in_features, affine=True)
        self.norm2 = BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        return out


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out


import sys

import numpy as np
import math
import torch
import os
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
from tqdm import tqdm
import cv2
from scipy.interpolate import LinearNDInterpolator


# Input :
#       reference(dictionary from vertex idx to normalized landmark, dict[idx] = [x, y, z]) : landmark of reference frame.
#       target(dictionary from vertex idx to normalized landmark, dict[idx] = [x, y, z]) : landmark of target frame.
# Output : 
#       R : 3x3 Rotation matrix(np.array)
#       c : scale value(float)
#       t : 3x1 translation matrix(np.array)

KEY_IDX = [17, 84, 16, 314, 85, 315, 15, 181, 86, 14, 316, 87, 180, 317, 405, 179, 404, 178, 18, 402, 403, 83, 313, 91, 90, 200, 88, 89, 201, 199, 421, 320, 318, 321, 152, 175, 319, 182, 406, 377, 148, 208, 428, 171, 396, 95, 96, 77, 400, 146, 176, 194, 418, 324, 369, 140, 325, 32, 262, 307, 106, 375, 335, 378, 149, 395, 204, 170, 62, 78, 76, 424, 61, 431, 211, 379, 308, 292, 306, 150, 43, 291, 394, 273, 169, 183, 184, 430, 365, 185, 202, 210, 191, 422, 136, 364, 407, 408, 415, 409, 287, 57, 397, 135, 40, 74, 42, 288, 80, 172, 367, 212, 432, 434, 214, 270, 361, 310, 272, 304, 410, 138, 58, 186, 81, 435, 41, 13, 73, 39, 323, 82, 312, 416, 12, 311, 38, 11, 269, 268, 271, 72, 303, 302, 132, 401, 0, 386, 267, 454, 37, 192, 433, 436, 215, 216, 387, 93, 322, 366, 385, 356, 427, 213]

def Umeyama_algorithm(reference, target):
    # idx 2 -> nose, 130 -> left eye, 359 -> right eye
    idx_list = [2, 94, 19, 1, 4, 5, 195, 197, 6, 168, 8, 9, 151, 10, 109, 108, 67, 69, 103, 104, 54, 68, 338, 337, 297, 299, 332, 333, 284, 298, 130, 243, 244, 359, 362, 463,
                21, 71, 162, 139, 156, 70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 301, 251, 55, 285, 193, 417, 122, 351, 196, 419, 3, 248, 51, 281,
                45, 275, 44, 274, 220, 440, 134, 363, 236, 456]
    # idx_list = [19, 243, 463]
    ref_points = []
    tgt_points = []

    for idx in idx_list:
        ref_points.append(reference[idx])
        tgt_points.append(target[idx])

    ref_points = np.array(ref_points)
    tgt_points = np.array(tgt_points)

    ref_mu = ref_points.mean(axis=0)
    tgt_mu = tgt_points.mean(axis=0)
    ref_var = ref_points.var(axis=0).sum()
    tgt_var = tgt_points.var(axis=0).sum()
    n, m = ref_points.shape
    covar = np.matmul((ref_points - ref_mu).T, tgt_points - tgt_mu) / n
    det_covar = np.linalg.det(covar)
    u, d, vh = np.linalg.svd(covar)
    detuv = np.linalg.det(u) * np.linalg.det(vh.T)
    cov_rank = np.linalg.matrix_rank(covar)
    S = np.identity(m)

    if cov_rank > m - 1:
        if det_covar < 0:
            S[m - 1, m - 1] = -1
    else: 
        if detuv < 0:
            S[m - 1, m - 1] = -1

    R = np.matmul(np.matmul(u, S), vh)
    c = (1 / tgt_var) * np.trace(np.matmul(np.diag(d), S))
    t = ref_mu.reshape(3, 1) - c * np.matmul(R, tgt_mu.reshape(3, 1))

    return R, t, c


def landmark_to_dict(landmark_list):
    landmark_dict = {}
    for idx, landmark in enumerate(landmark_list):
        landmark_dict[idx] = [landmark.x, landmark.y, landmark.z]

    return landmark_dict

def landmarkdict_to_normalized_mesh_tensor(landmark_dict):
    vertex_list = []
    for idx, coord in landmark_dict.items():
        if (idx == 'R') or (idx == 't') or (idx == 'c'):
            continue
        vertex_list.append(coord)
    
    if not ('R' in landmark_dict):
        return torch.tensor(vertex_list)
    
    R = torch.from_numpy(landmark_dict['R']).float()
    t = torch.from_numpy(landmark_dict['t']).float()
    c = float(landmark_dict['c'])
    vertices = torch.tensor(vertex_list).transpose(0, 1)
    norm_vertices = (c * torch.matmul(R, vertices) + t).transpose(0, 1)
    return norm_vertices


def landmarkdict_to_mesh_tensor(landmark_dict):
    vertex_list = []
    for idx, coord in landmark_dict.items():
        if (idx == 'R') or (idx == 't') or (idx == 'c'):
            continue
        vertex_list.append(coord)

    vertices = torch.tensor(vertex_list)
    return vertices

def mesh_tensor_to_landmarkdict(mesh_tensor):
    landmark_dict = {}
    for i in range(mesh_tensor.shape[0]):
        landmark_dict[i] = mesh_tensor[i].tolist()
    
    return landmark_dict

def edge2color(edge_index):
    total = len(mp_face_mesh.FACEMESH_TESSELATION)
    id = (edge_index + 1) / total
    c = 127 + int(id * 128)
    return (c, c, c)

def draw_lips(keypoints, new_img, c = (255, 255, 255), th=1):
    keypoints = keypoints.astype(np.int32)
    for i in range(48, 59):
	    cv2.line(new_img, tuple(keypoints[i]), tuple(keypoints[i+1]), color=c, thickness=th)
    cv2.line(new_img, tuple(keypoints[48]), tuple(keypoints[59]), color=c, thickness=th)
    cv2.line(new_img, tuple(keypoints[48]), tuple(keypoints[60]), color=c, thickness=th)
    cv2.line(new_img, tuple(keypoints[54]), tuple(keypoints[64]), color=c, thickness=th)
    cv2.line(new_img, tuple(keypoints[67]), tuple(keypoints[60]), color=c, thickness=th)
    for i in range(60, 67):
        cv2.line(new_img, tuple(keypoints[i]), tuple(keypoints[i+1]), color=c, thickness=th)


def draw_mask(maskKp, shape, c=(255, 255, 255)):
  mask = np.zeros(shape, dtype=np.int32)
  center = np.mean(maskKp, axis=0)
  delta = maskKp - center
#   delta[1:8, 1] *= 2
  maskKp = center + delta
  _ = cv2.fillPoly(mask, [maskKp.astype(np.int32)], c)
  mask = mask.astype(np.float32) / 255.0
  return mask

def get_seg(mesh_dict, shape):
    keypoints = np.array(list(mesh_dict.values())[:478])[:, :2]
    keypoints = keypoints.astype(np.int32)
    oval_idx = [356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389]
    out_lip_idx = [181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61, 146, 91]
    in_lip_idx = [178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78, 95, 88]
    maskKp = keypoints[oval_idx]
    seg = draw_mask(maskKp, shape)
    cv2.fillPoly(seg, [keypoints[out_lip_idx]], color=(2, 2, 2))
    cv2.fillPoly(seg, [keypoints[in_lip_idx]], color=(3, 3, 3))
    return seg

def draw_mesh_image(mesh_dict, save_path, image_rows, image_cols):
    drawing_spec = mp_drawing.DrawingSpec(color= mp_drawing.BLACK_COLOR, thickness=1, circle_radius=1)
    idx_to_coordinates = {}
    for idx, coord in mesh_dict.items():
        if (idx == 'R') or (idx == 't') or (idx == 'c'):
            continue
        x_px = min(math.floor(coord[0]), image_cols - 1)
        y_px = min(math.floor(coord[1]), image_rows - 1)
        landmark_px = (x_px, y_px)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    
    # get segment map
    segmap = get_seg(mesh_dict, (image_cols, image_rows, 3)) * 32

    # draw mesh
    connections = mp_face_mesh.FACEMESH_TESSELATION
    for edge_index, connection in enumerate(connections):
        start_idx = connection[0]
        end_idx = connection[1]
        color = edge2color(edge_index)
        if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
            cv2.line(segmap, 
                idx_to_coordinates[start_idx],
                idx_to_coordinates[end_idx], 
                color,
                1
            )
    cv2.imwrite(save_path, segmap)

def normalize_mesh(mesh_dict):
    mesh = landmarkdict_to_normalized_mesh_tensor(mesh_dict)
    return mesh_tensor_to_landmarkdict(mesh)

def normalize_meshes(mesh_dir, save_dir):
    mesh_filename_list = get_file_list(mesh_dir)
    os.makedirs(save_dir, exist_ok=True)
    for mesh_filename in tqdm(mesh_filename_list):
        mesh_dict = torch.load(mesh_filename)
        mesh = landmarkdict_to_normalized_mesh_tensor(mesh_dict)
        save_path = os.path.join(save_dir, os.path.basename(mesh_filename)[:-3] + '.pt')
        torch.save(mesh_tensor_to_landmarkdict(mesh), save_path)

def interpolate_zs_dir(data_dir, image_rows=256, image_cols=256):
    for mesh_dir in os.listdir(data_dir):
        if not mesh_dir.startswith('mesh_dict'):
            continue
        print(f"working on {mesh_dir}...")
        post_fix = mesh_dir[len('mesh_dict'):]
        save_dir = 'z' + post_fix
        interpolate_zs(os.path.join(data_dir, mesh_dir), os.path.join(data_dir, save_dir), image_rows, image_cols)


def interpolate_zs(mesh_dir, save_dir, image_rows, image_cols):
    mesh_filename_list = get_file_list(mesh_dir)
    os.makedirs(save_dir, exist_ok=True)
    for mesh_filename in tqdm(mesh_filename_list):
        mesh_dict = torch.load(mesh_filename)
        mesh = landmarkdict_to_mesh_tensor(mesh_dict)
        save_path = os.path.join(save_dir, os.path.basename(mesh_filename)[:-3] + '.pt')
        z = interpolate_z(mesh, image_rows, image_cols) # H x W x 1
        torch.save(z, save_path)

def interpolate_z(mesh, image_rows=256, image_cols=256):
  points, values = mesh[:, :2].numpy(), mesh[:, [2]].numpy()
  interp = LinearNDInterpolator(points, values, fill_value=0)
  X, Y = np.meshgrid(range(0, image_rows), range(0, image_cols))
  Z = interp(X, Y)
  return torch.from_numpy(Z) # image_rows x image_cols x 1

def draw_mesh_images(mesh_dir, save_dir, image_rows, image_cols):
    mesh_filename_list = get_file_list(mesh_dir)
    os.makedirs(save_dir, exist_ok=True)
    for mesh_filename in tqdm(mesh_filename_list):
        mesh_dict = torch.load(mesh_filename)
        save_path = os.path.join(save_dir, os.path.basename(mesh_filename)[:-3] + '.png')
        draw_mesh_image(mesh_dict, save_path, image_rows, image_cols)
    

def draw_mesh_images_dir(data_dir, image_rows=256, image_cols=256):
    for mesh_dir in os.listdir(data_dir):
        if not mesh_dir.startswith('mesh_dict'):
            continue
        print(f"working on {mesh_dir}...")
        post_fix = mesh_dir[len('mesh_dict'):]
        save_dir = 'mesh_image' + post_fix
        draw_mesh_images(os.path.join(data_dir, mesh_dir), os.path.join(data_dir, save_dir), image_rows, image_cols)

