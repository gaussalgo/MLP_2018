# # MLP_2018

## Task 00 - Initialization
* File `00_init.py`
* Install requirements (may need to restart session)
* Download data

## Task 01 - Image classification
* open `01_image_classification.py`
* go through the process description comments
* run the script
* check that it works 

## Task 02 - Image similarity search
* open `02_image_similarity.py`
* go through the process description comments
* run `02_image_similarity.py` on single thread (as is)
* check results

## Task 03 - Parallelization of image preprocessing
* open `03_01_is_master.py`
* go through the process description comments
* run `03_01_is_master.py` on 3 threads (workers)
* run `02_image_similarity.py`
* check results and compare speed of processing with task 02

## Task 04 - Flask server and similarity of uploaded image upload and similarity
* create teams
* add a simple flask web app to the project that can upload new images
* export the server to CDSW_PUBLIC_PORT (port 8080 - https://www.cloudera.com/documentation/data-science-workbench/latest/topics/cdsw_web_ui.html)
* run the web server and open the link from the top right hand corner dropdown
* copy URL and send to your smart phone
* upload a picture of your shoe :)
* update path in `02_image_similarity.py`
* check which shoes are similar