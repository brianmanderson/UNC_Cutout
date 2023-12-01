import glob
import os
import time

import pydicom
from Dicom_RT_and_Images_to_Mask.src.DicomRTTool.ReaderWriter import DicomReaderWriter, sitk
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image
from NiftiResampler.ResampleTools import ImageResampler, sitk
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


def return_binary_mask(numpy_binary_outline: np.array, out_shape):
    contours = np.where(numpy_binary_outline == 255)
    binary_mask = poly2mask(contours[0], contours[1], (out_shape[0], out_shape[1]))
    return binary_mask


def return_mask_handle_from_dot_report(png_path, size_mm=(250, 250)):
    img = cv2.imread(png_path, cv2.COLOR_BGR2GRAY)
    boundaries = find_rectangle_boundary(np.array(img))
    numpy_image = np.array(img)
    row_start = boundaries[0][0]
    row_stop = boundaries[0][-1]
    col_start = boundaries[1][0]
    col_stop = boundaries[1][-1]
    pad = 5
    inner_square = numpy_image[row_start+pad:row_stop-pad, col_start+pad:col_stop-pad]
    out_mask = np.zeros((row_stop-row_start, col_stop-col_start))
    # green_center = find_green_cross(np.array(img))
    invert_blue = 255 - inner_square[..., 0]
    invert_blue[invert_blue <= np.median(invert_blue)] = 0
    invert_green = 255 - inner_square[..., 1]
    invert_green[invert_green <= np.median(invert_green)] = 0
    red = inner_square[..., 2]
    red[red <= np.median(red)] = 0
    summed = (red/3 + invert_blue/3 + invert_green/3).astype('uint8')
    summed[summed > 90] = 255
    summed[summed < 255] = 0
    inner_shape = inner_square.shape
    binary_mask = return_binary_mask(summed, inner_shape)
    out_mask[pad: -pad, pad:-pad] += binary_mask.astype('int')
    out_mask = np.flipud(out_mask)
    mask_handle = sitk.GetImageFromArray(out_mask.astype('int'))
    mask_handle.SetSpacing((out_mask.shape[0]/size_mm[0], out_mask.shape[1]/size_mm[1]))
    return mask_handle


def return_mask_handle_from_scanner(jpeg_path):
    img = cv2.imread(jpeg_path, cv2.COLOR_BGR2GRAY)
    inner_square = np.array(img)
    # green_center = find_green_cross(np.array(img))
    invert_blue = 255 - inner_square[..., 0]
    invert_blue[invert_blue <= np.median(invert_blue)] = 0
    invert_green = 255 - inner_square[..., 1]
    invert_green[invert_green <= np.median(invert_green)] = 0
    invert_red = 255 - inner_square[..., 2]
    invert_red[invert_red <= np.median(invert_red)] = 0
    summed = (invert_red/3 + invert_blue/3 + invert_green/3).astype('uint8')
    summed[summed > 90] = 255
    summed[summed < 255] = 0
    inner_shape = inner_square.shape[:2]
    binary_mask = np.flipud(return_binary_mask(summed, inner_shape))
    mask_handle = sitk.GetImageFromArray(binary_mask.astype('int'))
    mask_handle.SetSpacing((0.35728, 0.35728))  # 72 dpi
    return mask_handle


<<<<<<< HEAD
def create_rt_mask(path, mask_handle: sitk.Image):
    reader = DicomReaderWriter()
    reader.down_folder(os.path.join(path, "CT"))
    reader.get_images()

=======
def create_rt_mask(reader: DicomReaderWriter, out_path, mask_handle: sitk.Image):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
>>>>>>> b1d89e313b276a9de23bb8f05a1f540548c99c72
    input_shape = reader.ArrayDicom.shape
    out_mask = np.zeros(input_shape + (2,)).astype('int')
    dicom_spacing = reader.dicom_handle.GetSpacing()

    resampler = ImageResampler()
    desired_dimensions = (dicom_spacing[1], dicom_spacing[2])
    resampled = resampler.resample_image(input_image_handle=mask_handle, output_spacing=desired_dimensions,
                                         interpolator='Nearest')
    resampled_array = sitk.GetArrayFromImage(resampled)
    mask_shape = resampled_array.shape

    row_start = int((input_shape[0]-mask_shape[0])/2)
    col_start = int((input_shape[1]-mask_shape[1])/2)
    out_mask[row_start:row_start + mask_shape[0], 1, col_start:col_start + mask_shape[1], 1] = resampled_array
    reader.prediction_array_to_RT(out_mask, output_dir=out_path, ROI_Names=["New"], ROI_Types=["PTV"])
    return


def temp_run():
    path = r'Data'

    reader = DicomReaderWriter()
    reader.down_folder(os.path.join(path, "CT"))
    reader.get_images()
    sitk.WriteImage(reader.dicom_handle, "Data/Dicom_Handle.mhd")

    ct_path = "Data/CT.mhd"
    resampler = ImageResampler()
    if not os.path.exists(ct_path):
        desired_dimensions = (0.5, 0.5, 5.0)
        resampled = resampler.resample_image(input_image_handle=reader.dicom_handle, output_spacing=desired_dimensions,
                                             interpolator='Linear')
        sitk.WriteImage(resampled, "Data/CT.mhd")
    ct_resampled = sitk.ReadImage(ct_path)
    ct_numpy = sitk.GetArrayFromImage(ct_resampled)
    out_mask = np.zeros(ct_numpy.shape, dtype='int')
    r, c = 100, 150
    out_mask[1:5, r:c, r] = 1
    out_mask[1:5, r:c, c] = 1
    out_mask[1:5, c, r:c+1] = 1
    out_mask[1:5, r, r:c] = 1
    out_mask[1:5, r:c, r:c] = 1
    out_mask[2:4, r+10:c-10, r+10:c-10] = 0
    out_mask[2:4, r+14:c-14, r+14:c-14] = 1
    points = np.where(out_mask > 0)
    indexes = np.array(points[::-1]).transpose()
    image: sitk.Image
    image = reader.dicom_handle
    physical = np.asarray([image.TransformContinuousIndexToPhysicalPoint(zz.astype('float')) for zz in indexes])
    out_handle = sitk.GetImageFromArray(out_mask)
    out_handle.SetSpacing(image.GetSpacing())
    out_handle.SetOrigin(image.GetOrigin())
    out_handle.SetDirection(image.GetDirection())
    boundary = sitk.BinaryContour(out_handle)
    sitk.WriteImage(out_handle, "Data/Handle.mhd")
    out_mask = np.zeros(reader.ArrayDicom.shape + (2,))
    out_mask[1, r:c, r] = 1
    out_mask[1, r:c, c] = 1
    out_mask[1, c, r:c+1] = 1
    out_mask[1, r, r:c] = 1

    reader.prediction_array_to_RT(out_mask, output_dir=path, ROI_Names=["New"])
    x = 1



def main():
    # create_png(path)
    # create_mask(path)
    # mask_handle = return_mask_handle_from_dot_report(os.path.join(path, "Report.png"))
    monitored_path = r'\\vscifs1\physicsQAdata\UNC_ElectronCutout'
    reader = DicomReaderWriter()
    reader.down_folder(r'\\vscifs1\physicsQAdata\BMA\CutoutWork\Exam')
    reader.get_images()
    while True:
        time.sleep(3)  # Sleep for 3 seconds between waiting
        for file in glob.glob(os.path.join(monitored_path, "*jpg")):
            time.sleep(3)  # Sleep for 3 seconds to make sure its uploaded
            mask_handle = return_mask_handle_from_scanner(file)
            create_rt_mask(reader, monitored_path, mask_handle)
            os.remove(file)


if __name__ == '__main__':
    temp_run()
