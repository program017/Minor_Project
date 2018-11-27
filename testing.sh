echo Extracting Image from film
python image_extraction.py
echo Image Extraction done. Now removal of junk images
python face_identify.py
echo Prediction starts
python classify.py
read -sp "Program Ended"