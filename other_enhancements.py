import cv2
import numpy as np
import os

def sharpen(img):
    yuv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    
    y = y.astype(np.float32) / 255.0
    k = np.array([[0, -0.0625, 0],
                  [-0.0625, 1.05, -0.0625],
                  [0, -0.0625, 0]], dtype=np.float32)
    y_sharp = cv2.filter2D(y, -1, k)
    y_sharp = np.clip(y_sharp, 0, 1) * 255.0

    yuv_sharp = cv2.merge([y_sharp.astype(np.uint8), u, v])
    res = cv2.cvtColor(yuv_sharp, cv2.COLOR_YUV2BGR).astype(np.float32) / 255.0
    return res

def adjust_bc(img, a, b):
    res = img * a + b
    res = np.clip(res, 0, 1)
    return res

def adjust_sat(img, sf):
    res = img.copy()
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            px = img[y, x]
            max_val = max(px)
            res[y, x] = px + (max_val - px) * sf
    res = np.clip(res, 0, 1)
    return res

def main():
    input_folder = 'outputs/'
    output_folder = 'enhancement/'

    os.makedirs(f'{output_folder}/brightness_contrast', exist_ok=True)
    os.makedirs(f'{output_folder}/saturation', exist_ok=True)
    os.makedirs(f'{output_folder}/sharpen', exist_ok=True)
    os.makedirs(f'{output_folder}/combined', exist_ok=True)

    images = os.listdir(input_folder)

    for image in images:
        img_path = os.path.join(input_folder, image)
        orig = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if orig is None:
            print(f"Failed to load image: {image}")
            continue

        orig = orig.astype(np.float32) / 255.0

        a = 1.05
        b = 0.1
        bc_adj = adjust_bc(orig, a, b)
        bc_output_path = os.path.join(output_folder, 'brightness_contrast', image)
        cv2.imwrite(bc_output_path, (bc_adj * 255).astype(np.uint8))
        print(f"Saved brightness & contrast adjusted image to {bc_output_path}")

        sf = -0.5
        sat_adj = adjust_sat(orig, sf)
        sat_output_path = os.path.join(output_folder, 'saturation', image)
        cv2.imwrite(sat_output_path, (sat_adj * 255).astype(np.uint8))
        print(f"Saved saturation adjusted image to {sat_output_path}")

        sharp = sharpen(orig)
        sharp_output_path = os.path.join(output_folder, 'sharpen', image)
        cv2.imwrite(sharp_output_path, (sharp * 255).astype(np.uint8))
        print(f"Saved sharpened image to {sharp_output_path}")

        temp1 = adjust_bc(orig, a, b)
        temp2 = adjust_sat(temp1, sf)
        temp3 = sharpen(temp2)
        final = temp3.copy()
        final_output_path = os.path.join(output_folder, 'combined', image)
        cv2.imwrite(final_output_path, (final * 255).astype(np.uint8))
        print(f"Saved final combined image to {final_output_path}")

if __name__ == "__main__":
    main()
