
import torch
import torch.nn.functional as F
import torch.nn as nn

def image_to_pointcloud(depth, intrinsics):
    assert depth.dim() == 4
    assert depth.size(1) == 1

    X = depth * intrinsics.X_cam
    Y = depth * intrinsics.Y_cam
    return torch.cat((X, Y, depth), dim=1)


def pointcloud_to_image(pointcloud, intrinsics):
    assert pointcloud.dim() == 4

    batch_size = pointcloud.size(0)
    X = pointcloud[:, 0, :, :]  #.view(batch_size, -1)
    Y = pointcloud[:, 1, :, :]  #.view(batch_size, -1)
    Z = pointcloud[:, 2, :, :].clamp(min=1e-3)  #.view(batch_size, -1)

    # compute pixel coordinates
    U_proj = intrinsics.fu * X / Z + intrinsics.cu  # horizontal pixel coordinate
    V_proj = intrinsics.fv * Y / Z + intrinsics.cv  # vertical pixel coordinate

    # normalization to [-1, 1], required by torch.nn.functional.grid_sample
    U_proj_normalized = (2 * U_proj / (intrinsics.width - 1) - 1).view(batch_size, -1)
    V_proj_normalized = (2 * V_proj / (intrinsics.height - 1) - 1).view(batch_size, -1)

    # This was important since PyTorch didn't do as it claimed for points out of boundary
    # See https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
    # Might not be necessary any more
    U_proj_mask = ((U_proj_normalized > 1) + (U_proj_normalized < -1)).detach()
    U_proj_normalized[U_proj_mask] = 2
    V_proj_mask = ((V_proj_normalized > 1) + (V_proj_normalized < -1)).detach()
    V_proj_normalized[V_proj_mask] = 2

    pixel_coords = torch.stack([U_proj_normalized, V_proj_normalized], dim=2)  # [B, H*W, 2]
    return pixel_coords.view(batch_size, intrinsics.height, intrinsics.width, 2)


def batch_multiply(batch_scalar, batch_matrix):
    # input: batch_scalar of size b, batch_matrix of size b * 3 * 3
    # output: batch_matrix of size b * 3 * 3
    batch_size = batch_scalar.size(0)
    output = batch_matrix.clone()
    for i in range(batch_size):
        output[i] = batch_scalar[i] * batch_matrix[i]
    return output


def transform_curr_to_near(pointcloud_curr, r_mat, t_vec, intrinsics):

    # translation and rotmat represent the transformation from tgt pose to src pose
    batch_size = pointcloud_curr.size(0)
    XYZ_ = torch.bmm(r_mat, pointcloud_curr.view(batch_size, 3, -1))

    X = (XYZ_[:, 0, :] + t_vec[:, 0].unsqueeze(1)).view(-1, 1, intrinsics.height, intrinsics.width)
    Y = (XYZ_[:, 1, :] + t_vec[:, 1].unsqueeze(1)).view(-1, 1, intrinsics.height, intrinsics.width)
    Z = (XYZ_[:, 2, :] + t_vec[:, 2].unsqueeze(1)).view(-1, 1, intrinsics.height, intrinsics.width)

    pointcloud_near = torch.cat((X, Y, Z), dim=1)

    return pointcloud_near


def homography_from(rgb_near, depth_curr, r_mat, t_vec, intrinsics):
    # inverse warp the RGB image from the nearby frame to the current frame

    # to ensure dimension consistency
    r_mat = r_mat.view(-1, 3, 3)
    t_vec = t_vec.view(-1, 3)

    # compute source pixel coordinate
    pointcloud_curr = image_to_pointcloud(depth_curr, intrinsics)
    pointcloud_near = transform_curr_to_near(pointcloud_curr, r_mat, t_vec, intrinsics)
    pixel_coords_near = pointcloud_to_image(pointcloud_near, intrinsics)

    # the warping
    warped = F.grid_sample(rgb_near, pixel_coords_near)

    return warped



class Intrinsics:
    def __init__(self, width, height, fu, fv, cu=0, cv=0):
        self.height, self.width = height, width
        self.fu, self.fv = fu, fv  # fu, fv: focal length along the horizontal and vertical axes

        # cu, cv: optical center along the horizontal and vertical axes
        self.cu = cu if cu > 0 else (width - 1) / 2.0
        self.cv = cv if cv > 0 else (height - 1) / 2.0

        # U, V represent the homogeneous horizontal and vertical coordinates in the pixel space
        self.U = torch.arange(start=0, end=width).expand(height, width).float()
        self.V = torch.arange(start=0, end=height).expand(width, height).t().float()
        
        # X_cam, Y_cam represent the homogeneous x, y coordinates (assuming depth z=1) in the camera coordinate system
        self.X_cam = (self.U - self.cu) / self.fu
        self.Y_cam = (self.V - self.cv) / self.fv

        self.is_cuda = False

    def cuda(self):
        self.X_cam.data = self.X_cam.data.cuda()
        self.Y_cam.data = self.Y_cam.data.cuda()
        self.is_cuda = True
        return self

    def scale(self, height, width):
        # return a new set of corresponding intrinsic parameters for the scaled image
        ratio_u = float(width) / self.width
        ratio_v = float(height) / self.height
        fu = ratio_u * self.fu
        fv = ratio_v * self.fv
        cu = ratio_u * self.cu
        cv = ratio_v * self.cv
        new_intrinsics = Intrinsics(width, height, fu, fv, cu, cv)
        if self.is_cuda:
            new_intrinsics.cuda()
        return new_intrinsics

    def __print__(self):
        print('size=({},{})\nfocal length=({},{})\noptical center=({},{})'.
              format(self.height, self.width, self.fv, self.fu, self.cv,
                     self.cu))


def _SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1)(x)
    mu_y = nn.AvgPool2d(3, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    SSIM_img = torch.clamp((1 - SSIM) / 2, 0, 1)

    return F.pad(SSIM_img, pad=(1, 1, 1, 1), mode='constant', value=0)



class L1Loss(nn.Module):
    def __init__(self, args):
        super(L1Loss, self).__init__()

        self.args = args
        self.t_valid = 0.0001

    def forward(self, gt, pred):
        gt = torch.clamp(gt, min=0, max=self.args.maxdisp)
        pred = torch.clamp(pred, min=0, max=self.args.maxdisp)

        mask = (gt > self.t_valid).type_as(pred).detach()

        d = torch.abs(pred - gt) * mask
        
        d = torch.sum(d, dim=[1, 2])
        num_valid = torch.sum(mask, dim=[1, 2])

        loss = d / (num_valid + 1e-8)

        loss = loss.sum()

        return loss

class L2Loss(nn.Module):
    def __init__(self, args):
        super(L2Loss, self).__init__()

        self.args = args
        self.t_valid = 0.0001

    def forward(self, gt, pred):
        gt = torch.clamp(gt, min=0, max=self.args.maxdisp)
        pred = torch.clamp(pred, min=0, max=self.args.maxdisp)

        mask = (gt > self.t_valid).type_as(pred).detach()

        d = torch.pow(pred - gt, 2) * mask

        d = torch.sum(d, dim=[1, 2])
        num_valid = torch.sum(mask, dim=[1, 2])

        loss = d / (num_valid + 1e-8)

        loss = loss.sum()

        return loss


class PhotometricLoss(nn.Module):
    def __init__(self):
        super(PhotometricLoss, self).__init__()
        
        self.F = 320
        self.B = 0.25
        
        self.r_mat = torch.eye(3)
        self.t_vec = torch.tensor([self.B, 0, 0])
        self.intrinsics = Intrinsics(640,480, self.F, self.F)

    def forward(self, rgb_right, rgb_left, depth_right):

        # compute the corresponding intrinsic parameters
        batch, channel, height_, width_ = depth_right.shape
        #intrinsics_ = self.intrinsics.scale(height_, width_)

        # inverse warp from a nearby frame to the current frame
        warped_ = homography_from(rgb_left, depth_right, self.r_mat, self.t_vec, self.intrinsics_)
        
        """
        plt.imshow(rgb_left[0].permute(1,2,0) / 255.0)
        plt.show()
        plt.imshow(rgb_right[0].permute(1,2,0) / 255.0)
        plt.show()
        plt.imshow(warped_[0].permute(1,2,0) / 255.0)
        plt.show()
        """

        
        
        #diff = (warped_ - rgb_right).abs()
        #diff = torch.sum(diff, 1)  # sum along the color channel
        diff = _SSIM(warped_, rgb_right)

        """
        plt.imshow(diff[0].permute(1,2,0) / 255.0)
        plt.show()
        """
                
        # compare only pixels that are not black
        valid_mask = (torch.sum(warped_, 1) > 0).float() * (torch.sum(rgb_left, 1) > 0).float()
        valid_mask = valid_mask.byte().detach()
        if valid_mask.numel() > 0:

            diff = diff * valid_mask
            
            """
            plt.imshow(valid_mask[0])
            plt.show()
            plt.imshow(diff[0])
            plt.show()
            """
            
            if diff.nelement() > 0:
                self.loss = diff.mean()
            else:
                print(
                    "warning: diff.nelement()==0 in PhotometricLoss (this is expected during early stage of training, try larger batch size)."
                )
                self.loss = 0
        else:
            print("warning: 0 valid pixel in PhotometricLoss")
            self.loss = 0
        return self.loss

class InputOutputLoss(nn.Module):
    def __init__(self, args):
        super(InputOutputLoss, self).__init__()

        self.maxdisp = args.maxdisp
        
    def forward(self, input, output):

        mask = input < self.maxdisp
        mask.detach_()
        # loss = F.smooth_l1_loss(pred[mask], GT[mask], size_average=True)
        loss = (input[mask] - output[mask]).abs().mean()

        return loss

class SmoothnessLoss(nn.Module):
    def __init__(self):
        super(SmoothnessLoss, self).__init__()

    def forward(self, depth):
        def second_derivative(x):
            assert x.dim(
            ) == 4, "expected 4-dimensional data, but instead got {}".format(
                x.dim())
            horizontal = 2 * x[:, :, 1:-1, 1:-1] - x[:, :, 1:-1, :
                                                     -2] - x[:, :, 1:-1, 2:]
            vertical = 2 * x[:, :, 1:-1, 1:-1] - x[:, :, :-2, 1:
                                                   -1] - x[:, :, 2:, 1:-1]
            der_2nd = horizontal.abs() + vertical.abs()
            return der_2nd.mean()

        self.loss = second_derivative(depth)
        return self.loss


class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()

        loss = args.loss.split("+")
        self.fn = []
        self.ln = [] 
        self.weights = []
        for l in loss:
            w, ln = l.split("*")
            if ln == 'l1':
                loss_fn = L1Loss(args)
            elif ln == 'l2':
                loss_fn = L2Loss(args)
            elif ln == 'photo':
                loss_fn = PhotometricLoss(args)
            elif ln == 'smooth':
                loss_fn = SmoothnessLoss(args)
            elif ln == 'inputoutput':
                loss_fn = InputOutputLoss(args)

            self.fn.append(loss_fn)
            self.ln.append(ln)
            self.weights.append(float(w))

    def forward(self, GT, pred, imgL, imgR, K):
        
        loss = 0
        for w, fn, ln in zip(self.weights, self.fn, self.ln):
            
            if ln in ['l1', 'l2']:
                loss += w * fn(GT, pred)
            elif ln in ['photo']:
                depR = 0.25 * 320.0 / pred 
                loss += w * fn(imgR, imgL, depR)
            elif ln in ['inputoutput']:
                loss += w * fn(GT, pred)

        return loss