# # Initial script

# ## Install requirements
!pip install --upgrade pip
!pip install -r requirements.txt
# !pip install -r requirements.txt --upgrade --ignore-installed


# ## Download imges from s3 and unpack them to data folder
# Expects aws credentials in file ``~/.aws/credentials``:
# [default]
# aws_access_key_id = YOUR_ACCESS_KEY
# aws_secret_access_key = YOUR_SECRET_KEY
#
import boto3

s3 = boto3.resource('s3') 

# Download images from s3
s3.Bucket('mlp2018').download_file('unique_1k_images.tgz', 'data/unique_1k_images.tgz')

# Unpack and store
!tar -zxf data/unique_1k_images.tgz -C data
