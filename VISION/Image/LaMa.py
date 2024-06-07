## From https://pub.towardsai.net/exploring-lama-resolution-robust-large-mask-inpainting-with-fourier-convolutions-a-brief-overview-593f29a3f8da

"""
1. Introduction to Image Inpainting
   -1. Objective
       Image inpainting reconstructs damaged or masked regions based on the surrounding context.
   -2. Challenge
       Larger masks make reconstruction more difficult due to the greater information to be restored and lesser context to rely on.

2. Problem with Large Masks
   -1. Increased Complexity
       As the size of the mask increases, the complexity of inpainting grows, making it harder to accurately restore the missing parts.
   -2. Reduced Context
       Larger masks provide less surrounding information for the model to infer the missing content, complicating the task.

3. LaMa’s Solution for Large Masks
   -1. Specialization
       LaMa is designed specifically to handle large masked areas effectively.
   -2. Innovative Structure
       Incorporates unique architectural elements and loss functions to enhance performance in challenging inpainting scenarios.

4. Network Architecture
   -1. GAN-Based
       Consists of a generator and a discriminator.
   -2. Input
       Concatenates an intact image with a binary mask to form the network input.
   -3. Structure
       Features a downscale stage, residual blocks (with Fast Fourier Convolution), and an upscale stage.
   -4. Output
       Produces a recovered image with the loss based on the discrepancy between the input and output images.

5. Fourier Transform and Spectral Transform
   -1. Fourier Transform
       Converts spatial/time domain data into the frequency domain, making each pixel represent specific frequencies.
   -2. Spectral Transform
       A key innovation, incorporating real-valued Fast Fourier Transform (Real FFT) to handle feature maps in the frequency domain.
   -3. Implementation
       Utilizes standard convolution blocks before and after Real FFT, followed by inverse Real FFT and a final 1x1 convolution.

6. Fast Fourier Convolution (FFC)
   -1. Dual Branches
       - Local Branch
         Standard convolution process (convolution -> batch normalization -> activation function).
       - Global Branch
         Applies spectral transform along with the standard convolution process.
   -2. Combination
       Outputs from both branches are concatenated, integrating regional and global features.

7. Loss Functions
   -1. High Receptive Field Perceptual Loss (HRFPL)
       - Purpose
         Addresses the challenge of restoring large masked areas by focusing on high-level context rather than detailed pixel-wise comparison.
       - Implementation
         Uses a network like Vgg19 to extract image features, optimizing the discrepancy between input and output feature maps.

   -2. Adversarial Loss
       - Purpose
         Enhances the realism of generated content by fostering competition between the generator and discriminator.
       - Implementation
         Combines generator and discriminator losses to refine the generated inpainted areas.

   -3. Additional Loss Functions:
       - Gradient Penalty
         Ensures smooth transitions and stability in the inpainted areas.
       - Feature Matching Loss
         Perceptual loss based on the features of the discriminator network.
       - Total Loss
         A weighted sum of HRFPL, adversarial loss, gradient penalty, and feature matching loss ensures comprehensive training.

8. Conclusion
   LaMa’s Impact: Offers robust solutions for large mask inpainting with innovative techniques like FFC and HRFPL.
   Advancements: Demonstrates significant improvements in handling challenging inpainting scenarios with high accuracy and efficiency.

LaMa stands out in the field of image inpainting, particularly for large masks, by integrating advanced convolution techniques and innovative loss functions,
making it a powerful tool for restoring heavily masked images.
"""

## Spectrum Transform
class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        
        self.downsample = nn.Identity()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )

        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        
        self.conv2 = torch.nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x).  # Paper:[Conv-BN-ReLU]
        output = self.fu(x) # Paper:[Real FFT2d - Conv-BN-ReLU - Inv Real FFT2d]

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs). # Paper:[Conv 1x1]

        return output

## Fast Fourier Convolution (FFC)
### resnet blocks
# n_blocks = 9

for i in range(n_blocks):
    cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
    if spatial_transform_layers is not None and i in spatial_transform_layers:
       cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
       model += [cur_resblock]


## High Receptive Field Perceptual Loss (HRFPL)
# Ref: https://github.com/advimman/lama/blob/3197c5e5e42503a66868e636ed48a7cefc5e8c28/saicinpainting/training/losses/perceptual.py#L67

loss = F.mse_loss(features_input, features_target, reduction='none')
loss = loss.mean(dim=tuple(range(1, len(loss.shape))))


## Adversarial Loss
# Ref: https://github.com/advimman/lama/blob/3197c5e5e42503a66868e636ed48a7cefc5e8c28/saicinpainting/training/trainers/default.py#L115
discr_real_pred, discr_real_features = self.discriminator(img)
discr_fake_pred, discr_fake_features = self.discriminator(predicted_img)
adv_gen_loss, adv_metrics = self.adversarial_loss.generator_loss(real_batch=img, fake_batch=predicted_img, discr_real_pred=discr_real_pred, discr_fake_pred=discr_fake_pred, mask=mask_for_discr)




