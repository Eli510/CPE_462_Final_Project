import cv2
import os
import numpy as np
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


def show_img(img, title):
    disp = (img * 255).astype(np.uint8)
    cv2.imshow(title, disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def add_gaussian_noise(img, mean=0, std=0.05):
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy_img = img + noise
    return np.clip(noisy_img, 0, 1)


def add_salt_pepper_noise(img, amount=0.02):
    noisy_img = img.copy()
    num_salt = int(amount * img.size * 0.5)
    num_pepper = int(amount * img.size * 0.5)

    coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]
    noisy_img[coords[0], coords[1], :] = 1

    coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape[:2]]
    noisy_img[coords[0], coords[1], :] = 0

    return noisy_img


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

    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs/addedNoise', exist_ok=True)
    os.makedirs('inputs/addedNoise', exist_ok=True)


    for image in images:
        img_path = f'inputs/{image}'
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"[WARN] Couldn't read image: {img_path}")
            continue

        img = img.astype(np.float32) / 255.0
        
        output, _ = upsampler.enhance((img * 255).astype(np.uint8))
        cv2.imwrite(f'outputs/{image}', output)
        print(f'Upscaled {image} from inputs/')

    for image in images:
        img_path = f'inputs/{image}'
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"[WARN] Couldn't read image: {img_path}")
            continue 

        img = img.astype(np.float32) / 255.0

        noisy_img = add_gaussian_noise(img)
        
        noisy_image_path = f'inputs/addedNoise/{image}'
        cv2.imwrite(noisy_image_path, (noisy_img * 255).astype(np.uint8))
        print(f'Saved noisy image: {noisy_image_path}')
        
        noisy_img_input = cv2.imread(noisy_image_path, cv2.IMREAD_UNCHANGED)
        if noisy_img_input is None:
            print(f"[WARN] Couldn't read noisy image: {noisy_image_path}")
            continue
        
        output, _ = upsampler.enhance(noisy_img_input)
        cv2.imwrite(f'outputs/addedNoise/{image}', output)
        print(f'Upscaled {image} from inputs/addedNoise/')


if __name__ == '__main__':
    main()
