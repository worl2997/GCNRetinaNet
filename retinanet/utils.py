import torch
import torch.nn as nn
import numpy as np
from dgl.nn.pytorch import GraphConv
import dgl




def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class conv1x1(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(conv1x1,self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.conv1 =  nn.Conv2d(self.in_feat, self.out_feat, kernel_size=1, stride=1, padding=0)
    def forward(self,x):
        return self.conv1(x)


# GCN 기반으로 이미지 feature map을 업데이트 하는 부분
# node_feauter은 forward의 input으로 들어감
class FUB(nn.Module):
    def __init__(self, feat_size, activation, dropout, fmap_s, node_size):
        super(FUB, self).__init__()
        # 직접 해당 클래스 안에서 input_feature를 기반으로 그래프를 구현해야 함
        self.dropout = dropout
        self.in_feats = feat_size
        self.fmap_size = fmap_s[0] * fmap_s[1] * fmap_s[2] * fmap_s[3]
        self.out_feats = feat_size  # 사실상 in_feat_size와 동일하게 만들어 주어야함
        self.layer = GraphConv(self.fmap_size,self.fmap_size)
        self.dropout = nn.Dropout(p=0.2) #
        # self.EdgeWeight = nn.Parameter(torch.Tensor(node_size**2, 1))

    # edge 계산 메소드
    def make_distance(self, x1, x2, in_feats):
        x_add = x1 + x2  # elementwise add
        c_cat = torch.cat([x2, x_add], dim=1)  # 이거는 C x H x W 차원일 기준으로 하것
        convolution = conv1x1(2 * in_feats, in_feats)
        target = convolution(c_cat)
        distance = ((x2 - target).abs()).sum()
        # distance가 이게 맞나?
        # 현재 이 distance가 과연 스칼라 값일까
        return distance

    # 이부분도 개선의 여지가 있음
    # 다만 일단 가장 베이스 부분만 구현을 진행하고 추후 추가 구현을 진행하자
    def make_edge_matirx(self, node_feats, in_feats):
        # 입력 받은 feature node  리스트를 기반으로 make_distance로 edge를 계산하고
        # pruning 기능 추가
        # edata에 넣어줄 형식으로 변환해야함, size -> (36, 1)
        Node_feats = node_feats
        edge_list = []
        edge_feature = []
        for i, node_i in enumerate(Node_feats):
            for j, node_j in enumerate(Node_feats):
                if i == j:
                    edge_list.append(1)  # 수정 가능성 존재
                else:
                    edge_list.append(self.make_distance(node_i, node_j, in_feats))
                    # edge_feature.append(self.make_distance(node_i, node_j, in_feats))
        # edge_feature = torch.tensor(edge_feature)
        # mean = edge_feature.mean()
        edge_list = torch.tensor(edge_list)

        return edge_list

    # graph 와 node feature matrix 반환
    def make_dgl_graph(self, node_feats, edge_feats):
        # 여기에 그래프 구성 코드를 집어넣으면 됨
        ns = node_feats[0].size()
        len_node = len(node_feats)
        src_node = []
        dst_node = []
        for i in range(len_node):
            src_node += ([i] * len_node)
            for j in range(len_node):
                dst_node.append(j)
        g = dgl.graph(data=(src_node, dst_node), num_nodes=len_node, device='cpu')
        node_feats_matrix = torch.Tensor(len_node, ns[0], ns[1], ns[2], ns[3])
        for i, node in enumerate(node_feats):
            node_feats_matrix[i] = node
        h = node_feats_matrix.view(len_node, -1)  # 6 x (NxCxHxW) , node feature  -> 이부분은 외부에서 입력으로 넣어주어야 하는 값인듯
        g.edata['e'] = edge_feats  # 6 x 6 ..? , edge feature

        return g, h

    def forward(self, x):
        # GCN을 구현해서 input으로 넣어주어야함
        # 노드랑 edge feats
        cuda = False
        node_feats = x  # list form으로 구성되어있음 [re_c1,.., re_c2]
        edge_feats = self.make_edge_matirx(node_feats, self.in_feats)
        size = node_feats[0].size()
        g, h = self.make_dgl_graph(x, edge_feats)  # 여기서 그래프를 초기화 해줌 ㅇㅇ

        h = self.dropout(h)
        h = self.layer(g, h)

        # h 차원 -> [node_num, N, C, H,W]
        out = [x.reshape(size[0], size[1], size[2], size[3]) for x in h]

        return out  # original feature 리스트와  업데이트 된 feature list 반환

class GFPN_conv(nn.Module):
    def __init__(self, feature_size):
      super(GFPN_conv,self).__init__()
      self.in_feat = feature_size*2
      self.out_feat = feature_size
      self.activation = nn.LeakyReLU()
      self.conv_1 = nn.Conv2d(self.in_feat, self.out_feat, kernel_size=1, stride=1, padding=0)
      self.conv = conv3x3(feature_size, feature_size, stride=1)
      self.bn = nn.BatchNorm2d(self.out_feat)
    def forward(self, origin, h):
      result_feat = []
      for origin_feat, updated_feat in zip(origin, h):
        feat = torch.cat([origin_feat, updated_feat],dim=1)
        out = self.conv_1(feat)
        out = self.activation(out)
        out = self.conv(out)
        out = self.bn(out)
        result_feat.append(out)
        # 최종적으로 prediction head로 넘겨줄 output_head들
        # 다만 사이즈를 원래의 original feature사이즈로 변환해주어야함
      return result_feat

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            if torch.cuda.is_available():
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
            else:
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))

        else:
            self.mean = mean
        if std is None:
            if torch.cuda.is_available():
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
            else:
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        else:
            self.std = std

    def forward(self, boxes, deltas):

        widths  = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x   = boxes[:, :, 0] + 0.5 * widths
        ctr_y   = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w     = torch.exp(dw) * widths
        pred_h     = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):

        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)
      
        return boxes
