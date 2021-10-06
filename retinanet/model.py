import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms
from retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors
from retinanet import losses
import torch.nn.functional as F

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# Node feature create la
class Nodefeats(nn.Module):
    def __init__(self, num_GCN, fpn_channels):
        super(Nodefeats, self).__init__()
        self.num_GCN = num_GCN
        self.fpn_cahnnels = fpn_channels # [256, 512, 1024, 2048] # 레벨별 채널 수
        self.num_backbone_feats = len(fpn_channels)
        self.target_size = fpn_channels[(self.num_backbone_feats+2)/2-1]  # 채널수가 너무많음.. 1024

        self.make_C5_ = self.make_C5(self.fpn_channels[-1],self.target_size) # C4 -> C5
        self.make_C6_ =  nn.Conv2d(self.target_size, self.target_size,kernel_size=3, stride=2, padding=1) # C5 -> C6

        # Target node = C3 -> 256 channel로 통일하자
        #  1x1 conv (channel resize) -> 2-stride max_pooing -> 3x3 2 stride conv (down sample)
        self.resize_C1 =  self.resize_node_feature(fpn_channels[0],self.target_size,1)
        # 1 1x1 conv (channel resize) -> 3x3 2stride conv
        self.resize_C2 = self.resize_node_feature(fpn_channels[1], self.target_size,2)
        self.resize_C3 = self.resize_node_feature(fpn_channels[2], self.target_size,3) # channel size = 1024
        self.resize_C4 = nn.Upsample(scale_factor=2, mode='nearest') #-> upsample x2
        self.resize_C5 = nn.Upsample(scale_factor=4, mode='nearest') # upsample X4
        self.resize_C6 = nn.Upsample(scale_factor=8, mode='nearest') # upsample x8

    def make_C5(self,in_ch, out_ch):
        stage = nn.Sequential()
        stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                           out_channels=out_ch, kernel_size=1, stride=1, padding=0))# 1x1 channel resize
        stage.add_module('conv',nn.Conv2d(out_ch,out_ch,kernel=3, stride=2 , padding=1))  # 3x3 conv 2 stride
        return stage

    # 맞춰야할 기준 노드가 level3 노드라고 가정하고 짜는 코드
    def resize_node_feature(self, in_feat, target_feat,src_level):
        # 일단 단순하게 짜자, 성능이 확인되면 general한 구조 생각하기
        if src_level == 1:
            stage = nn.Sequential()
            stage.add_module('conv', nn.Conv2d(in_channels=in_feat,
                                               out_channels=target_feat,
                                               kernel_size=1,
                                               stride=1,
                                               padding=0))
            stage.add_module('max_pooling', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            stage.add_module('conv_2s',nn.Conv2d(in_channels=target_feat,
                                               out_channels=target_feat,
                                               kernel_size=3,
                                               stride=2,
                                               padding=1))
            return stage
        elif src_level == 2:
            stage = nn.Sequential()
            stage.add_module('conv', nn.Conv2d(in_channels=in_feat,
                                               out_channels=target_feat,
                                               kernel_size=1,
                                               stride=1,
                                               padding=0))
            stage.add_module('conv_2s', nn.Conv2d(in_channels=target_feat,
                                                  out_channels=target_feat,
                                                  kernel_size=3,
                                                  stride=2,
                                                  padding=1))
            return stage
        elif src_level == 3: # baseline node
            stage = nn.Conv2d(in_feat, target_feat,kernel_size=1,stride=1,padding=0)
            return stage

                # forward input D
    def forward(self, inputs):
        C1, C2, C3, C4 = inputs
                # C1 ~ C6 : original feature
        C5 = self.make_C5_(C4)
        C6 = self.make_C6_(C5)

        re_c1 = self.resize_C1(C1)
        re_c2 = self.resize_C2(C2)
        re_c3 = self.resize_C3(C3)
        re_c4 = self.resize_C4(C4)
        re_c5 = self.resize_C5(C5)
        re_c6 = self.resize_C6(C6)
        return [re_c1,re_c2,re_c3,re_c4,re_c5,re_c6] # 최종적으로 resize된 feature 반환


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class ResNet(nn.Module):
        # layers -> 각각 layer를 몇번 반복사는지 알려줌
        #  ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        self.num_graph_blocks = 2
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) #
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])  # C1 -> output_size 56x56
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) #C2 -> output_size 28x28
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) #C3 -> 14x14
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) #C4 -> 7x7

        # 변경 부분
        if block == BasicBlock:
            fpn_channel_sizes = [self.layer1[layers[0] - 1].conv2.out_channels , self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_channel_sizes = [self.layer1[layers[0] - 1].conv3.out_channels, self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = self.Node_feature_make(self.num_graph_blocks,fpn_channel_sizes)

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = losses.FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):

        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        Node_features = self.Nodefeats([x1, x2, x3, x4]) # FPN으로 부터 feature 추출 -> 나중에 이거 기반으로 컨트롤좀 해보기
        GCN_FPN_features = self.GCN_FPN(Node_features) # Node_features -> [C1 ~ C6]
        regression = torch.cat([self.regressionModel(feature) for feature in GCN_FPN_features], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in GCN_FPN_features], dim=1)

        anchors = self.anchors(img_batch)

        if self.training:
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            finalResult = [[], [], []]

            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])

            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

            for i in range(classification.shape[2]):
                scores = torch.squeeze(classification[:, :, i])
                scores_over_thresh = (scores > 0.05)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)
                anchorBoxes = anchorBoxes[scores_over_thresh]
                anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

                finalResult[0].extend(scores[anchors_nms_idx])
                finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                finalResult[2].extend(anchorBoxes[anchors_nms_idx])

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                if torch.cuda.is_available():
                    finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]



def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model
