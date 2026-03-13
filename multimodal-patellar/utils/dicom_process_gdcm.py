# load necessary packages
import matplotlib.pyplot as plt
import pydicom.uid
import sys
# from PyQt5 import QtGui
import os
import pydicom as dicom
import glob
from PIL import *
import matplotlib.pyplot as plt
from pylab import *
from tkinter.filedialog import *
import PIL.Image as Image
import gdcm
    
have_numpy = True

try:
    import numpy
except ImportError:
    have_numpy = False
    raise

sys_is_little_endian = (sys.byteorder == 'little')

NumpySupportedTransferSyntaxes = [
    pydicom.uid.ExplicitVRLittleEndian,
    pydicom.uid.ImplicitVRLittleEndian,
    pydicom.uid.DeflatedExplicitVRLittleEndian,
    pydicom.uid.ExplicitVRBigEndian,
]

# 支持"传输"语法
def supports_transfer_syntax(dicom_dataset):
    return (dicom_dataset.file_meta.TransferSyntaxUID in NumpySupportedTransferSyntaxes)

def needs_to_convert_to_RGB(dicom_dataset):
    return False


def should_change_PhotometricInterpretation_to_RGB(dicom_dataset):
    return False

def decompress_dicom(filename):
    """使用GDCM解压DICOM文件"""
    reader = gdcm.ImageReader()
    reader.SetFileName(filename)
    if not reader.Read():
        raise Exception("无法读取DICOM文件")
    
    change = gdcm.ImageChangeTransferSyntax()
    change.SetTransferSyntax(gdcm.TransferSyntax(gdcm.TransferSyntax.ImplicitVRLittleEndian))
    change.SetInput(reader.GetImage())
    if not change.Change():
        raise Exception("无法转换传输语法")
    
    writer = gdcm.ImageWriter()
    writer.SetImage(change.GetOutput())
    # 创建临时文件名
    temp_filename = filename + ".temp"
    writer.SetFileName(temp_filename)
    if not writer.Write():
        raise Exception("无法写入解压后的文件")
    
    return temp_filename

# 加载 Dicom图像
def get_pixeldata(dicom_dataset):
    """修改后的get_pixeldata函数，先尝试GDCM解压"""
    try:
        # 在函数开始就获取文件名
        original_filename = dicom_dataset.filename
        
        # 如果是压缩格式，先解压
        if (dicom_dataset.file_meta.TransferSyntaxUID not in NumpySupportedTransferSyntaxes):
            # 解压文件
            temp_filename = decompress_dicom(original_filename)
            # 重新读取解压后的文件
            dicom_dataset = dicom.read_file(temp_filename)
            # 删除临时文件
            os.remove(temp_filename)
        
        if not have_numpy:
            msg = ("The Numpy package is required to use pixel_array, and "
                   "numpy could not be imported.")
            raise ImportError(msg)
        if 'PixelData' not in dicom_dataset:
            raise TypeError("No pixel data found in this dataset.")
        
        pixel_array = dicom_dataset.pixel_array
        rows = dicom_dataset.Rows
        cols = dicom_dataset.Columns
        return pixel_array, rows, cols

    except Exception as e:
        raise Exception(f"处理DICOM文件 {original_filename} 时出错: {str(e)}")

def setDicomWinWidthWinCenter(img_data, winwidth, wincenter, rows, cols):
    img_temp = img_data
    img_temp.flags.writeable = True
    min = (2 * wincenter - winwidth) / 2.0 + 0.5
    max = (2 * wincenter + winwidth) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)

    for i in numpy.arange(rows):
        for j in numpy.arange(cols):
            img_temp[i, j] = int((img_temp[i, j]-min)*dFactor)

    min_index = img_temp < min
    img_temp[min_index] = 0
    max_index = img_temp > max
    img_temp[max_index] = 255

    return img_temp

def loadFileInformation(filename):
    information = {}
    ds = pydicom.read_file(filename)
    information['PatientID'] = ds.PatientID
    information['PatientName'] = ds.PatientName
    information['PatientBirthDate'] = ds.PatientBirthDate
    information['PatientSex'] = ds.PatientSex
    information['StudyID'] = ds.StudyID
    information['StudyDate'] = ds.StudyDate
    information['StudyTime'] = ds.StudyTime
    information['InstitutionName'] = ds.InstitutionName
    information['Manufacturer'] = ds.Manufacturer
    print(dir(ds))
    print(type(information))
    return information

def main_single(path, winwidth, wincenter):
    dcm = dicom.read_file(path) # load dicom_file
    print(f"原始DICOM图像尺寸: {dcm.Rows} x {dcm.Columns}")
    # 得到 CT 值，图像的 长， 宽
    pixel_array, dcm.Rows, dcm.Columns = get_pixeldata(dcm)

    img_data = pixel_array
    rows = dcm.Rows
    cols = dcm.Columns
    dcm_temp = setDicomWinWidthWinCenter(img_data, winwidth, wincenter, rows, cols)
    # 可视化
    dcm_img = Image.fromarray(dcm_temp)  # 将Numpy转换为PIL.Image
    dcm_img = dcm_img.convert('L')
    # plt.imshow(img, cmap=plt.cm.bone)
    # 保存为jpg文件，用作后面的生成label用
    dcm_img.save('./data/dicom_processed/temp1.jpg')
    print(f"最终保存的图像尺寸: {dcm_img.size}")
    # 显示图像
    # dcm_img.show()

def main_mulit(path, save_path, winwidth, wincenter):
    names = os.listdir(path)  # 读取文件夹中的所有文件名
    for i in range(len(names)):
        dicom_name = path+names[i]
        dcm = pydicom.read_file(dicom_name)  # 读取 dicom 文件
        pixel_array, dcm.Rows, dcm.Columns = get_pixeldata(dcm)  # 得到 dicom文件的 CT 值
        img_data = pixel_array
        rows = dcm.Rows
        cols = dcm.Columns
        dcm_temp = setDicomWinWidthWinCenter(img_data, winwidth, wincenter, rows, cols) # 调整窗位、窗宽
        #  可视化
        dcm_img = Image.fromarray(dcm_temp)  # 将 Numpy转换为 PIL.Image
        dcm_img = dcm_img.convert('L')
        # 批量保存
        dcm_img.save("%s%s.png" % (save_path, names[i][:-4]))

if __name__ == '__main__': 
    # bone W:1800 L:400
    # main_single('./data/阳性组1/2201438259高位髌骨0000.dcm',2000,800)
    main_mulit('../data/外部缺失数据/','../data/temp/',2000,500)