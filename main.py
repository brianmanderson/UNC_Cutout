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
import fitz


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


def create_png(file_path):
    doc = fitz.open(file_path)
    page = doc.load_page(0)  # number of page
    pix = page.get_pixmap()
    pix.save(file_path.replace(".pdf", ".png"))
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


def return_largest_label(stats: sitk.LabelIntensityStatisticsImageFilter, maximum_size: np.inf):
    largest_size = 0
    out_label = 1
    for l in stats.GetLabels():
        physical_size = stats.GetPhysicalSize(l)
        if physical_size > largest_size:
            if physical_size < maximum_size:
                out_label = l
                largest_size = physical_size
    return out_label, largest_size


def return_mask_handle_from_dot_report(png_path, size_mm=(250, 250)):
    img = cv2.imread(png_path, cv2.COLOR_BGR2GRAY)
    numpy_img = np.array(img)
    white = np.min(numpy_img, axis=-1)
    numpy_img[white > 150] = 0
    red = numpy_img[..., 2]
    red[red < 50] = 0
    red[red > 0] = 255
    labeled_truth, stats = return_stats_labels(red)
    """
    First lets find the outside boundary red box. Then we will focus on the cutout
    """
    largest_label, largest_size = return_largest_label(stats, np.inf)
    boundary_box = stats.GetBoundingBox(largest_label)
    out_mask = np.zeros((boundary_box[-1], boundary_box[2]))
    row_start = boundary_box[1]
    row_stop = row_start + boundary_box[-1]
    col_start = boundary_box[0]
    col_stop = col_start + boundary_box[2]
    """
    Now, find the binary mask of the cutout
    """
    pad = 5
    inner_square = red[row_start + pad:row_stop - pad, col_start + pad:col_stop - pad]
    labeled_truth, stats = return_stats_labels(inner_square)
    cutout_label, _ = return_largest_label(stats, largest_size)
    binary_mask = sitk.GetArrayFromImage(labeled_truth) == cutout_label
    out_mask[pad: -pad, pad:-pad] += binary_mask.astype('int')
    out_mask = np.flipud(out_mask)
    out_path = os.path.dirname(os.path.dirname(os.path.dirname(png_path)))  # Bump up two levels
    write_binary_mask_image(out_path, out_mask)

    mask_handle = sitk.GetImageFromArray(out_mask.astype('int'))
    mask_handle.SetSpacing((size_mm[0]/out_mask.shape[0], size_mm[1]/out_mask.shape[1]))
    return mask_handle


def return_stats_labels(numpy_array: np.array):
    summed_handle = sitk.GetImageFromArray(numpy_array.astype('int'))
    filled_in_handle = sitk.BinaryFillhole(summed_handle, fullyConnected=True, foregroundValue=255)
    cc = sitk.ConnectedComponent(filled_in_handle > 0)
    connected_component_filter = sitk.ConnectedComponentImageFilter()
    labeled_truth = connected_component_filter.Execute(filled_in_handle)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(cc, labeled_truth)
    return labeled_truth, stats


def return_largest_sitk_handle(numpy_array: np.array):
    labeled_truth, stats = return_stats_labels(numpy_array)
    l_label, l_size = return_largest_label(stats, np.inf)
    return labeled_truth == l_label


def write_binary_mask_image(out_path, binary_mask: np.array):
    img_out = np.zeros(binary_mask.shape + (3,), dtype="uint8")
    img_out[binary_mask > 0] = 255
    cv2.imwrite(os.path.join(out_path, "mask.jpg"), img_out)
    return None


def return_mask_handle_from_scanner(jpeg_path, folder: str):
    """
    :param jpeg_path:
    :param folder: name of the color, 'Red', 'Green', 'Blue', 'Black'
    :return:
    """
    img = cv2.imread(jpeg_path, cv2.COLOR_BGR2GRAY)
    inner_square = np.array(img)
    # green_center = find_green_cross(np.array(img))
    blue = inner_square[..., 0]
    green = inner_square[..., 1]
    red = inner_square[..., 2]
    if folder != "Blue":
        blue = 255 - blue
    blue[blue <= np.median(blue) + 50] = 0
    if folder != "Green":
        green = 255 - green
    green[green <= np.median(green) + 50] = 0
    if folder != "Red":
        red = 255 - red
    red[red <= np.median(red) + 50] = 0
    summed = (red/3 + blue/3 + green/3).astype('uint8')
    summed[summed > 50] = 255
    summed[summed < 255] = 0
    labeled_truth = return_largest_sitk_handle(summed)
    binary_mask = sitk.GetArrayFromImage(labeled_truth) == 1
    binary_mask[summed > 0] = 0 # Cutout the edge
    binary_mask = np.flipud(binary_mask)
    out_path = os.path.dirname(os.path.dirname(jpeg_path))  # Bump up two levels
    write_binary_mask_image(out_path, binary_mask)
    mask_handle = sitk.GetImageFromArray(binary_mask.astype('int'))
    mask_handle.SetSpacing((0.35728, 0.35728))  # 0.08467 for 300 dpi, 0.35728 for 72 dpi
    return mask_handle


def create_rt_mask(reader: DicomReaderWriter, out_path, mask_handle: sitk.Image):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
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
    reader.down_folder(os.path.join(path, "Exam"))
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


def main():
    # create_png(path)
    # create_mask(path)
    #
    monitored_path = r'\\vscifs1\physicsQAdata\UNC_ElectronCutout'
    dot_decimal_path = os.path.join(monitored_path, "FromDotDecimal")
    reader = DicomReaderWriter()
    if os.path.exists(r'.\Data\Exam'):
        reader.down_folder(r'.\Data\Exam')
    else:
        reader.down_folder(r'\\vscifs1\physicsQAdata\BMA\CutoutWork\Exam')
    reader.get_images()
    while True:
        time.sleep(3)  # Sleep for 3 seconds between waiting
        for folder in ["Red", "Green", "Blue", "Black"]:
            for file_name in os.listdir(os.path.join(monitored_path, folder)):
                if not file_name.lower().endswith("jpg"):
                    continue
                file = os.path.join(monitored_path, folder, file_name)
                print(f"Running off {file}")
                time.sleep(3)  # Sleep for 3 seconds to make sure its uploaded
                mask_handle = return_mask_handle_from_scanner(file, folder)
                create_rt_mask(reader, monitored_path, mask_handle)
                os.remove(file)
        # for folder in os.listdir(dot_decimal_path):
        #     applicator_size = int(folder.split('x')[0])
        #     for file_name in os.listdir(os.path.join(dot_decimal_path, folder)):
        #         file = os.path.join(dot_decimal_path, folder, file_name)
        #         if file_name.lower().endswith("pdf"):
        #             create_png(file)
        #             os.remove(file)
        #             continue
        #         if not file_name.lower().endswith("png"):
        #             continue
        #         mask_handle = return_mask_handle_from_dot_report(file,
        #                                                          size_mm=(applicator_size*10, applicator_size*10))
        #         create_rt_mask(reader, monitored_path, mask_handle)
        #         os.remove(file)


if __name__ == '__main__':
    main()
