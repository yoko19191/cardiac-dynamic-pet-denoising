import os
import numpy as np
import pydicom

def load_4d_dicom(patient_path):
    """
    load a 4D DICOM files
    param:
    - patient_path: patient directory
    return:
    - images : a 4D np array (time, depth, height, width)
    """
    dicom_files = [pydicom.dcmread(os.path.join(patient_path, f)) for f in os.listdir(patient_path) if f.endswith('.dcm')]

    if not dicom_files:
        raise ValueError(f"No DICOM files found in the specified directory: {patient_path}")

    for file in dicom_files:
        if 'InstanceNumber' not in file:
            raise ValueError(f"InstanceNumber is missing from DICOM file: {file.filename}")
        if 'AcquisitionTime' not in file:
            raise ValueError(f"AcquisitionTime is missing from DICOM file: {file.filename}")
        if 'ImagePositionPatient' not in file:
            raise ValueError(f"ImagePositionPatient is missing from DICOM file: {file.filename}")

    dicom_files.sort(
        key=lambda x: (float(x.AcquisitionTime), int(x.InstanceNumber), [float(i) for i in x.ImagePositionPatient]))

    time_points = sorted(set(file.AcquisitionTime for file in dicom_files))

    images = []
    for time in time_points:
        time_slice_files = [file for file in dicom_files if file.AcquisitionTime == time]
        time_slices = []

        for file in time_slice_files:
            time_slices.append(file.pixel_array)

        images.append(time_slices)

    images = np.array(images)

    return images