import os
from DicomRTTool.ReaderWriter import DicomReaderWriter, sitk
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image
# from fitz import *
from scipy.signal import convolve2d
import cv2
import numpy as np


def poly2mask(vertex_row_coords: np.array, vertex_col_coords: np.array,
              shape: tuple) -> np.array:
    """[converts polygon coordinates to filled boolean mask]

    Args:
        vertex_row_coords (np.array): [row image coordinates]
        vertex_col_coords (np.array): [column image coordinates]
        shape (tuple): [image dimensions]

    Returns:
        [np.array]: [filled boolean polygon mask with vertices at
                     (row, col) coordinates]
    """
    xy_coords = np.array([vertex_col_coords, vertex_row_coords])
    coords = np.expand_dims(xy_coords.T, 0)
    mask = np.zeros(shape)
    cv2.fillPoly(mask, coords, 1)
    return np.array(mask, dtype=bool)


def create_ground_truth_mask(path):
    reader = DicomReaderWriter()
    reader.set_contour_names_and_associations(contour_names=['t'])
    reader.down_folder(path)
    reader.get_images_and_mask()
    sitk.WriteImage(reader.annotation_handle, "Data/Mask.mhd")
    return


def create_png(path):
    pdffile = os.path.join(path, "ReportImage.pdf")
    doc = fitz.open(pdffile)
    page = doc.load_page(0)  # number of page
    pix = page.get_pixmap()
    output = "Data\Report.png"
    pix.save(output)
    doc.close()
    return


def find_rectangle_boundary(gray: np.array):
    white = np.min(gray, axis=-1)
    gray[white > 150] = 0
    blue = gray[..., 0]
    green = gray[..., 1]
    red = gray[..., 2]
    red_square = (red > 150) * (green > 100) * (blue > 100)
    boundaries = np.where(red_square)
    return boundaries


def find_green_cross(gray: np.array):
    white = np.min(gray, axis=-1)
    gray[white > 150] = 0
    green = gray[..., 1]
    cross = np.array([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]])
    convolved = convolve2d(green, (cross/9))
    cross_center = np.where(convolved == np.max(convolved))
    center_row = cross_center[0][0]
    center_col = cross_center[1][0]
    return center_row, center_col


def return_binary_mask(png_path):
    img = cv2.imread(png_path, cv2.COLOR_BGR2GRAY)
    boundaries = find_rectangle_boundary(np.array(img))
    numpy_image = np.array(img)
    inner_square = numpy_image[boundaries[0][0]+3:boundaries[0][-1]-3, boundaries[1][0]+3:boundaries[1][-1]-3]
    green_center = find_green_cross(np.array(img))
    invert_blue = 255 - inner_square[..., 0]
    invert_blue[invert_blue <= np.median(invert_blue)] = 0
    invert_green = 255 - inner_square[..., 1]
    invert_green[invert_green <= np.median(invert_green)] = 0
    red = inner_square[..., 2]
    red[red <= np.median(red)] = 0
    summed = (red/3 + invert_blue/3 + invert_green/3).astype('uint8')
    summed[summed > 90] = 255
    summed[summed < 255] = 0
    contours = np.where(summed == 255)
    inner_shape = inner_square.shape
    out_mask = poly2mask(contours[0], contours[1], (inner_shape[0], inner_shape[1]))
    return out_mask


def main():
    path = r'\\vscifs1\physicsqadata\BMA\CutoutWork'
    # create_mask(path)
    # create_png(path)
    mask = return_binary_mask("Data\Report.png")



    x = 1


if __name__ == '__main__':
    main()
