
try:
    print("Elementary imports: ")
    import os
    import json
    import glob
    import argparse
    print("numpy/scipy imports:")
    import numpy as np
    from scipy import signal as sg
    import scipy.ndimage as ndimage
    from scipy.ndimage.filters import maximum_filter
    from scipy import misc
    print("PIL imports:")
    from PIL import Image
    print("matplotlib imports:")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError:
    print("Need to fix the installation")
    raise
print("All imports okay. Yay!")


def find_tfl_lights(image1, red_image, green_image, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    x_red, y_red, x_green, y_green = convolution(image1,  red_image, green_image)
    return x_red, y_red, x_green, y_green
    # x = np.arange(-100, 100, 20) + c_image.shape[1] / 2
    # y_red = [c_image.shape[0] / 2 - 120] * len(x)
    # y_green = [c_image.shape[0] / 2 - 100] * len(x)
    #
    # return x, y_red, x, y_green


def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    image1 = image
    # image = image[:, :, 0]
    red_image = image[:, :, 0]
    green_image = image[:, :, 1]
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]
    show_image_and_gt(image, objects, fig_num)
    red_x, red_y, green_x, green_y = find_tfl_lights(image, red_image, green_image, some_threshold=42)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=4)


def convolution(image1, red_image, green_image):
    kernel = np.array([[-2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2/324, -2/324, -2/324],
                       [-2 / 324, -2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1/324, -2/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -2 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1/324, -1/324, -2/324],
                       [-2 / 324, -2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1/324, -2/324, -2/324],
                       [-2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2/324, -2/324, -2/324]])  # Gx + j*Gy
    # scharr = np.array([[-5, 9, 8], [-21, 8, -21], [-0.8, 0, 0]])
    red = sg.convolve2d(red_image, kernel, boundary='symm', mode='same')
    # fig, (ax_orig, ax_mag) = plt.subplots(2, 1, figsize=(6, 15))
    # ax_orig.imshow(red_image)
    # ax_orig.set_title('Original')
    # ax_orig.set_axis_off()
    # ax_mag.imshow(np.absolute(red))
    # ax_mag.set_title('Gradient magnitude')
    # ax_mag.set_axis_off()
    # # ax_ang.set_axis_off()
    # fig.show()
    # Image.fromarray(grad).show()
    max_light_red = ndimage.maximum_filter(red, size=5)
    # Image.fromarray(max_light).show()
    fig, (max_mag) = plt.subplots(1, 1, figsize=(6, 15))
    max_mag.imshow(np.absolute(max_light_red))
    x_red = []
    y_red = []
    d = 14
    image_height, image_width = max_light_red.shape[:2]
    for coordX in range(0, image_height-d, d):
        for coordY in range(0, image_width-d, d):
            currImage = max_light_red[coordX:coordX+d, coordY:coordY+d]
            localMax = np.amax(currImage)
            maxCurr = np.argmax(currImage)
            # coordinateX, coordinateY = np.unravel_index(np.argmax(currImage), currImage.shape)
            if localMax > 100:
                image1[coordX + maxCurr // d, coordY + maxCurr % d] = [255, 0, 0]
                x_red.append(coordX + maxCurr // d)
                y_red.append(coordY + maxCurr % d)
            currImage[:] = localMax
    print("image coordinates (red) :")
    print(x_red)
    print(len(x_red))
    print(y_red)
    # fig, (max_mag) = plt.subplots(1, 1, figsize=(6, 15))
    # max_mag.imshow(image1)
    green = sg.convolve2d(green_image, kernel, boundary='symm', mode='same')
    # fig, (ax_orig, ax_mag) = plt.subplots(2, 1, figsize=(6, 15))
    # ax_orig.imshow(red_image)
    # ax_orig.set_title('Original')
    # ax_orig.set_axis_off()
    # ax_mag.imshow(np.absolute(green))
    # ax_mag.set_title('Gradient magnitude')
    # ax_mag.set_axis_off()
    # ax_ang.set_axis_off()
    # fig.show()
    # Image.fromarray(grad).show()
    max_light_green = ndimage.maximum_filter(green, size=5)
    # Image.fromarray(max_light).show()
    fig, (max_mag) = plt.subplots(1, 1, figsize=(6, 15))
    max_mag.imshow(np.absolute(max_light_green))
    x_green = []
    y_green = []
    d = 24
    image_height, image_width = max_light_green.shape[:2]
    for coordX in range(0, image_height-d, d):
        for coordY in range(0, image_width-d, d):
            currImage = max_light_green[coordX:coordX+d, coordY:coordY+d]
            localMax = np.amax(currImage)
            maxCurr = np.argmax(currImage)
            # coordinateX, coordinateY = np.unravel_index(np.argmax(currImage), currImage.shape)
            if localMax > 100:
                image1[coordX + maxCurr // d, coordY + maxCurr % d] = [0, 255, 0]
                x_green.append(coordX + maxCurr // d)
                y_green.append(coordY + maxCurr % d)
            currImage[:] = localMax
    print("image coordinates (green) :")
    print(x_green)
    print(len(x_green))
    print(y_green)
    fig, (max_mag) = plt.subplots(1, 1, figsize=(6, 15))
    max_mag.imshow(image1)
    return x_red, y_red, x_green, y_green


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""
    # image = Image.open('erfurt_000015_000019_leftImg8bit.png').convert('L')
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = '../../data'
    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))
    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)
    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    main()