# Image Upscaling Code
import cv2
import os
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


def main():
    model_path = 'weights/RealESRGAN_x4plus.pth'
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False
    )

    images = os.listdir('inputs')

    for image in images:
        img = cv2.imread(f'inputs/{image}', cv2.IMREAD_UNCHANGED)
        output, _ = upsampler.enhance(img)
        cv2.imwrite(f'outputs/{image}', output)
        print(f'Upscaled {image}')


if __name__ == '__main__':
    main()
