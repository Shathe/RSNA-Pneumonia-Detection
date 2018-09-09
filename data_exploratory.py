import glob, pylab, pandas as pd
import pydicom, numpy as np
import cv2


def parse_data(df):
    """
    Method to read a CSV file (Pandas dataframe) and parse the 
    data into the following nested dictionary:

      parsed = {
        
        'patientId-00': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        },
        'patientId-01': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        }, ...

      }

    """
    # --- Define lambda to extract coords in list [y, x, height, width]
    extract_box = lambda row: [row['y'], row['x'], row['height'], row['width']]

    parsed = {}
    for n, row in df.iterrows():
        # --- Initialize patient entry into parsed 
        pid = row['patientId']
        if pid not in parsed:
            parsed[pid] = {
                'dicom': 'data/stage_1_train_images/%s.dcm' % pid,
                'label': row['Target'],
                'boxes': []}

        # --- Add box if opacity is present
        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))

    return parsed


def draw(data):
    """
    Method to draw single patient with bounding box(es) if present 

    """
    # --- Open DICOM file
    d = pydicom.read_file(data['dicom'])
    im = d.pixel_array

    # --- Convert from single-channel grayscale to 3-channel RGB
    im = np.stack([im] * 3, axis=2)

    # --- Add boxes with random color if present
    for box in data['boxes']:
        rgb = np.floor(np.random.rand(3) * 256).astype('int')
        im = overlay_box(im=im, box=box, rgb=rgb, stroke=6)

    pylab.imshow(im, cmap=pylab.cm.gist_gray)
    pylab.axis('off')
    pylab.show()
def overlay_box(im, box, rgb, stroke=1):
    """
    Method to overlay single box on image

    """
    # --- Convert coordinates to integers
    box = [int(b) for b in box]
    
    # --- Extract coordinates
    y1, x1, height, width = box
    y2 = y1 + height
    x2 = x1 + width

    im[y1:y1 + stroke, x1:x2] = rgb
    im[y2:y2 + stroke, x1:x2] = rgb
    im[y1:y2, x1:x1 + stroke] = rgb
    im[y1:y2, x2:x2 + stroke] = rgb

    return im


if __name__ == "__main__":
	print('Next file to explore: stage_1_train_labels.csv')
	raw_input("Please, press Enter to continue...")
	print ''
	print('It contains training set patientIds and labels (including bounding boxes)')
	print ('Each row contains a patientId, a target a target (either 0 or 1 for absence or presence of pneumonia, respectively) and the corresponding abnormality bounding box defined by the upper-left hand corner (x, y) coordinate and its corresponding width and height')
	print ''

	df = pd.read_csv('data/stage_1_train_labels.csv')
	print('Example of a patient with no pneumonia:')
	print(df.iloc[0])
	print ''

	print('Example of a patient with pneumonia:')
	print(df.iloc[4])
	print ''
	print('One important thing to keep in mind is that a given patientId may have multiple boxes')
	print ''
	print ''


	print('Next file to explore: stage_1_train_images/[patientid].dcm')
	raw_input("Please, press Enter to continue...")
	print ''
	print('Every patient has a file of this kind. This files is the medical image.')
	print('Each of this files contains a combination of header metadata of the image and the patient as well as underlying raw image arrays for pixel data')
	print ''

	patientId = df['patientId'][0]
	print('For patient with id: ' + str(patientId))

	dcm_file = 'data/stage_1_train_images/%s.dcm' % patientId
	dcm_data = pydicom.read_file(dcm_file)

	print(dcm_data)
	print ''
	raw_input("Please, press Enter to see the image...")
	img = dcm_data.pixel_array
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	parsed = parse_data(df)

	raw_input("Please, press Enter to see a image with bounding boxes...")
	draw(parsed['00436515-870c-4b36-a041-de91049b9ab4'])
	print ''
	print('Each bounding box without pneumonia is further categorized into normal or no lung opacity / not normal.')
	df_detailed = pd.read_csv('data/stage_1_detailed_class_info.csv')
	summary = {}
	for n, row in df_detailed.iterrows():
	    if row['class'] not in summary:
	        summary[row['class']] = 0
	    summary[row['class']] += 1
	    
	print(summary)
