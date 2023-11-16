import os
from DicomRTTool.ReaderWriter import DicomReaderWriter, sitk
from PlotScrollNumpyArrays.Plot_Scroll_Images import plot_scroll_Image
from fitz import *
import cv2
import numpy as np


def create_mask(path):
    reader = DicomReaderWriter()
    reader.set_contour_names_and_associations(contour_names=['t'])
    reader.down_folder(path)
    reader.get_images_and_mask()
    sitk.WriteImage(reader.annotation_handle, "Data/Mask.mhd")
    return


def main():
    path = r'\\vscifs1\physicsqadata\BMA\CutoutWork'
    # create_mask(path)
    pdffile = os.path.join(path, "ReportImage.pdf")
    doc = fitz.open(pdffile)
    page = doc.load_page(0)  # number of page
    pix = page.get_pixmap()
    output = "Data\Report.png"
    pix.save(output)
    doc.close()
    img = cv2.imread(output)
    img2 = np.array(img)  # Make a boundary of the red
    red = img2[..., 0]
    green = img2[..., 1]
    blue = img2[..., 2]
    img2[blue > 0] = 0
    img2[green > 0] = 0
    img2[red < 255] = 0
    x = 1


if __name__ == '__main__':
    main()
