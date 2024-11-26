"""
Description:
This script implements the proposed modules for efficient and accurate object detection:
1. Efficient Processing Modules (featuring PDWConv): Lightweight components designed to optimize feature extraction, incorporating Partial Depthwise Convolution (PDWConv) to balance efficiency and representation power. PDWConv processes a subset of input channels with depthwise convolutions, significantly reducing computational costs while preserving critical spatial and channel-wise features.
2. Fast_Head: A shared-feature detection head that optimizes resource usage by sharing features between object classification and bounding box regression tasks. This design reduces redundancy and accelerates inference without compromising performance.
3. Adaptive Sample Attention Loss (ASALoss): A novel loss function that dynamically assigns attention weights to samples based on their significance, focusing the model's learning on challenging examples while maintaining overall balance.
4. BboxLoss Combined with NWD_Loss: A hybrid loss function that integrates bounding box regression loss with Normalized Wasserstein Distance (NWD) Loss, enhancing localization precision and robustness.

These modules collectively aim to deliver high-performance object detection suitable for real-time and resource-constrained environments.
"""



######################################## EP begin ########################################

from timm.models.layers import DropPath
class Partial_DP_block(nn.Module):
    def __init__(self, ch, Rep_ratio=4, forward='split_cat'):
        super().__init__()
        self.Rep_channels = ch // Rep_ratio
        self.ch_untouched = ch - self.Rep_channels
        self.partial_DWconv = DWConv(self.Rep_channels, self.Rep_channels)
        self.Point_conv = Conv(ch, ch, 1)

        if forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.Rep_channels, self.ch_untouched], dim=1)
        x1 = self.partial_DWconv(x1)
        x = self.Point_conv(torch.cat((x1, x2), 1))
        return x

class Residual_PDP(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 Rep_ratio=4,
                 drop_path=0.1,
                 layer_scale_init_value=0.0,
                 type='split_cat'
                 ):
        super().__init__()
        self.out_ch = out_ch
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.Rep_ratio = Rep_ratio


        self.spatial_mixing = Partial_DP_block(
            out_ch,
            Rep_ratio,
            type
        )

        self.adjust_channel = None
        if in_ch != out_ch:
            self.adjust_channel = Conv(in_ch, out_ch, 1)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((out_ch)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(x)
        return x        

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1)*x)
        return x

class C3_PDP(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Residual_PDP(c_, c_) for _ in range(n)))

class EP(C2f):
    """Efficient Processing modules."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Residual_PDP(self.c, self.c) for _ in range(n))

######################################## EP end ########################################


class Fast_Head(nn.Module):
    """Detect Efficient head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.stem = nn.ModuleList(nn.Sequential(Partial_conv3(x, 4), Conv(x, x, 1)) for x in ch) # one PConv(CVsPR2023), one 1x1 Conv
        self.cv2 = nn.ModuleList(nn.Conv2d(x, 4 * self.reg_max, 1) for x in ch)
        self.cv3 = nn.ModuleList(nn.Conv2d(x, self.nc, 1) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = self.stem[i](x[i])
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a.bias.data[:] = 1.0  # box
            b.bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

        
class ASALoss:
    """ Adaptive Sample Attention Loss"""

    def __init__(self, loss_fcn, decay=0.999, tau=2000):
        super(ASALoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply SL to each element
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))# computing the value of current decay factor 
        self.is_train = True
        self.updates = 0
        self.iou_mean = 1.0
    
    def __call__(self, pred, true, auto_iou=0.5): 
        if self.is_train and auto_iou != -1:
            self.updates += 1 # plus 1 when calculating loss ecery time
            d = self.decay(self.updates) # a changing decay factor
            self.iou_mean = d * self.iou_mean + (1 - d) * float(auto_iou.detach()) # EMA(t) = α * data(t) + (1 - α) * EMA(t-1)
        auto_iou = self.iou_mean # latest EMA(t)
        loss = self.loss_fcn(pred, true)
        if auto_iou < 0.2:
            auto_iou = 0.2
        b1 = true <= auto_iou - 0.1
        a1 = 1.0
        b2 = (true > (auto_iou - 0.1)) & (true < auto_iou)
        a2 = math.exp(1.0 - auto_iou)
        b3 = true >= auto_iou
        a3 = torch.exp(-(true - 1.0))
        modulating_weight = a1 * b1 + a2 * b2 + a3 * b3
        loss *= modulating_weight
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
        

def wasserstein_loss(pred, target, eps=1e-7, constant=12.8):
    r"""`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.
    Code is modified from https://github.com/Zzh-tju/CIoU.
    Args:
        pred (Tensor): Predicted bboxes of format (x_min, y_min, x_max, y_max),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """

    b1_x1, b1_y1, b1_x2, b1_y2 = pred.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = target.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    
    b1_x_center, b1_y_center = b1_x1 + w1 / 2, b1_y1 + h1 / 2
    b2_x_center, b2_y_center = b2_x1 + w2 / 2, b2_y1 + h2 / 2
    center_distance = (b1_x_center - b2_x_center) ** 2 + (b1_y_center - b2_y_center) ** 2 + eps
    wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

    wasserstein_2 = center_distance + wh_distance
    return torch.exp(-torch.sqrt(wasserstein_2) / constant)


class BboxLoss(nn.Module):
    """ Combined with the nwd_loss"""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl
        self.nwd_loss =True
        self.iou_ratio = 0.5
        
        self.use_wiseiou = False
        
        if self.use_wiseiou:
            self.wiou_loss = WiseIouLoss(ltype='WIoU', monotonous=False, inner_iou=False, focaler_iou=False)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, mpdiou_hw=None):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        
        if self.use_wiseiou:
            wiou = self.wiou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask], ret_iou=False, ratio=0.7, d=0.0, u=0.95, **{'mpdiou_hw':mpdiou_hw[fg_mask]}).unsqueeze(-1) # Wise-MPDIoU,Wise-Inner-MPDIoU,Wise-Focaler-MPDIoU
            loss_iou = (wiou * weight).sum() / target_scores_sum
        else:
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
                
        if self.nwd_loss:
            nwd = wasserstein_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            nwd_loss = ((1.0 - nwd) * weight).sum() / target_scores_sum
            loss_iou = self.iou_ratio * loss_iou +  (1 - self.iou_ratio) * nwd_loss

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)
