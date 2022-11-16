#Importing Libraries
from tensorflow.keras.layers import Dense,Flatten,Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
from tensorflow.keras.models import load_model
import cv2
from skimage.transform import resize 
#Importing Dataset
import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='ibm_api_key_id',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.private.us.cloud-object-storage.appdomain.cloud')

bucket = 'Bucket'
object_key = 'Dataset.zip'

streaming_body_1 = cos_client.get_object(Bucket=bucket, Key=object_key)['Body']

# Your data file was loaded into a botocore.response.StreamingBody object.
# Please read the documentation of ibm_boto3 and pandas to learn more about the possibilities to load the data.
# ibm_boto3 documentation: https://ibm.github.io/ibm-cos-sdk-python/
# pandas documentation: http://pandas.pydata.org/
#Unzipping Dataset
from io import BytesIO
import zipfile
unzip = zipfile.ZipFile(BytesIO(streaming_body_1.read()), 'r')
file_paths = unzip.namelist()
for path in file_paths:
    unzip.extract(path)
    
import os
os.listdir('.')
['.virtual_documents', 'body.h5', 'body.tgz', 'Dataset', 'body_cloud.tar.gz']
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.1,zoom_range=0.1,horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
#MODEL FOR BODY TYPE DETECTION
trainPath = '/home/wsuser/work/Dataset/body/training'
testPath = '/home/wsuser/work/Dataset/body/validation'
training_set = train_datagen.flow_from_directory(trainPath,target_size=(244,244),batch_size=10,class_mode='categorical')
test_set = train_datagen.flow_from_directory(testPath,target_size=(244,244),batch_size=10,class_mode='categorical')
#Found 979 images belonging to 3 classes.
#Found 171 images belonging to 3 classes.
training_set.class_indices
{'00-front': 0, '01-rear': 1, '02-side': 2}
Declaring Model Variable
vgg=VGG16(input_shape=(244,244,3),weights='imagenet',include_top=False)



for layer in vgg.layers:
  layer.trainable=False


x=Flatten()(vgg.output)


prediction=Dense(3,activation='softmax')(x)


model=Model(inputs=vgg.input,outputs=prediction)
model.summary()
Model: "model_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_2 (InputLayer)        [(None, 244, 244, 3)]     0         
                                                                 
 block1_conv1 (Conv2D)       (None, 244, 244, 64)      1792      
                                                                 
 block1_conv2 (Conv2D)       (None, 244, 244, 64)      36928     
                                                                 
 block1_pool (MaxPooling2D)  (None, 122, 122, 64)      0         
                                                                 
 block2_conv1 (Conv2D)       (None, 122, 122, 128)     73856     
                                                                 
 block2_conv2 (Conv2D)       (None, 122, 122, 128)     147584    
                                                                 
 block2_pool (MaxPooling2D)  (None, 61, 61, 128)       0         
                                                                 
 block3_conv1 (Conv2D)       (None, 61, 61, 256)       295168    
                                                                 
 block3_conv2 (Conv2D)       (None, 61, 61, 256)       590080    
                                                                 
 block3_conv3 (Conv2D)       (None, 61, 61, 256)       590080    
                                                                 
 block3_pool (MaxPooling2D)  (None, 30, 30, 256)       0         
                                                                 
 block4_conv1 (Conv2D)       (None, 30, 30, 512)       1180160   
                                                                 
 block4_conv2 (Conv2D)       (None, 30, 30, 512)       2359808   
                                                                 
 block4_conv3 (Conv2D)       (None, 30, 30, 512)       2359808   
                                                                 
 block4_pool (MaxPooling2D)  (None, 15, 15, 512)       0         
                                                                 
 block5_conv1 (Conv2D)       (None, 15, 15, 512)       2359808   
                                                                 
 block5_conv2 (Conv2D)       (None, 15, 15, 512)       2359808   
                                                                 
 block5_conv3 (Conv2D)       (None, 15, 15, 512)       2359808   
                                                                 
 block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0         
                                                                 
 flatten_1 (Flatten)         (None, 25088)             0         
                                                                 
 dense_1 (Dense)             (None, 3)                 75267     
                                                                 
=================================================================
Total params: 14,789,955
Trainable params: 75,267
Non-trainable params: 14,714,688
_________________________________________________________________
Compiling the Model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
Training model
r = model.fit_generator(
    training_set,
    validation_data = test_set,
    epochs = 25,
    steps_per_epoch=979//10,
    validation_steps = 171//10
)
/tmp/wsuser/ipykernel_164/289406290.py:1: UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.
  r = model.fit_generator(
Epoch 1/25
97/97 [==============================] - 339s 3s/step - loss: 1.1511 - acc: 0.5459 - val_loss: 0.9324 - val_acc: 0.6294
Epoch 2/25
97/97 [==============================] - 328s 3s/step - loss: 0.6237 - acc: 0.7534 - val_loss: 0.7954 - val_acc: 0.6941
Epoch 3/25
97/97 [==============================] - 331s 3s/step - loss: 0.4937 - acc: 0.8070 - val_loss: 1.1732 - val_acc: 0.6176
Epoch 4/25
97/97 [==============================] - 326s 3s/step - loss: 0.4349 - acc: 0.8411 - val_loss: 0.9766 - val_acc: 0.6824
Epoch 5/25
97/97 [==============================] - 326s 3s/step - loss: 0.3661 - acc: 0.8617 - val_loss: 1.1987 - val_acc: 0.6529
Epoch 6/25
97/97 [==============================] - 325s 3s/step - loss: 0.2681 - acc: 0.8875 - val_loss: 0.9087 - val_acc: 0.6941
Epoch 7/25
97/97 [==============================] - 325s 3s/step - loss: 0.2292 - acc: 0.9195 - val_loss: 1.0251 - val_acc: 0.6647
Epoch 8/25
97/97 [==============================] - 326s 3s/step - loss: 0.1248 - acc: 0.9659 - val_loss: 1.0597 - val_acc: 0.6706
Epoch 9/25
97/97 [==============================] - 323s 3s/step - loss: 0.1315 - acc: 0.9639 - val_loss: 1.0529 - val_acc: 0.6647
Epoch 10/25
97/97 [==============================] - 322s 3s/step - loss: 0.0922 - acc: 0.9752 - val_loss: 0.9898 - val_acc: 0.6588
Epoch 11/25
97/97 [==============================] - 323s 3s/step - loss: 0.0913 - acc: 0.9825 - val_loss: 1.5796 - val_acc: 0.6529
Epoch 12/25
97/97 [==============================] - 322s 3s/step - loss: 0.1447 - acc: 0.9536 - val_loss: 1.1999 - val_acc: 0.6706
Epoch 13/25
97/97 [==============================] - 325s 3s/step - loss: 0.0746 - acc: 0.9763 - val_loss: 1.1819 - val_acc: 0.6647
Epoch 14/25
97/97 [==============================] - 325s 3s/step - loss: 0.1078 - acc: 0.9711 - val_loss: 1.0919 - val_acc: 0.7176
Epoch 15/25
97/97 [==============================] - 327s 3s/step - loss: 0.0659 - acc: 0.9866 - val_loss: 1.0925 - val_acc: 0.6824
Epoch 16/25
97/97 [==============================] - 326s 3s/step - loss: 0.0996 - acc: 0.9721 - val_loss: 1.2487 - val_acc: 0.6706
Epoch 17/25
97/97 [==============================] - 327s 3s/step - loss: 0.0683 - acc: 0.9845 - val_loss: 1.1608 - val_acc: 0.6824
Epoch 18/25
97/97 [==============================] - 328s 3s/step - loss: 0.0477 - acc: 0.9856 - val_loss: 1.5155 - val_acc: 0.6706
Epoch 19/25
97/97 [==============================] - 327s 3s/step - loss: 0.0748 - acc: 0.9825 - val_loss: 1.1204 - val_acc: 0.7235
Epoch 20/25
97/97 [==============================] - 324s 3s/step - loss: 0.0498 - acc: 0.9866 - val_loss: 1.2369 - val_acc: 0.6706
Epoch 21/25
97/97 [==============================] - 323s 3s/step - loss: 0.0736 - acc: 0.9876 - val_loss: 1.1987 - val_acc: 0.6706
Epoch 22/25
97/97 [==============================] - 325s 3s/step - loss: 0.0691 - acc: 0.9886 - val_loss: 1.1737 - val_acc: 0.7059
Epoch 23/25
97/97 [==============================] - 325s 3s/step - loss: 0.1011 - acc: 0.9711 - val_loss: 1.2466 - val_acc: 0.6882
Epoch 24/25
97/97 [==============================] - 327s 3s/step - loss: 0.0756 - acc: 0.9814 - val_loss: 1.5177 - val_acc: 0.6588
Epoch 25/25
97/97 [==============================] - 327s 3s/step - loss: 0.0480 - acc: 0.9866 - val_loss: 1.3861 - val_acc: 0.7353
model.save('body.h5')
!tar -zcvf body.tgz body.h5
body.h5
ls -1 
body.h5
body.tgz
Dataset/
!pip install watson-machine-learning-client --upgrade
Requirement already satisfied: watson-machine-learning-client in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (1.0.391)
Requirement already satisfied: boto3 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from watson-machine-learning-client) (1.18.21)
Requirement already satisfied: tqdm in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from watson-machine-learning-client) (4.62.3)
Requirement already satisfied: lomond in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from watson-machine-learning-client) (0.3.3)
Requirement already satisfied: pandas in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from watson-machine-learning-client) (1.3.4)
Requirement already satisfied: requests in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from watson-machine-learning-client) (2.26.0)
Requirement already satisfied: urllib3 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from watson-machine-learning-client) (1.26.7)
Requirement already satisfied: tabulate in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from watson-machine-learning-client) (0.8.9)
Requirement already satisfied: certifi in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from watson-machine-learning-client) (2022.9.24)
Requirement already satisfied: ibm-cos-sdk in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from watson-machine-learning-client) (2.11.0)
Requirement already satisfied: botocore<1.22.0,>=1.21.21 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from boto3->watson-machine-learning-client) (1.21.41)
Requirement already satisfied: jmespath<1.0.0,>=0.7.1 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from boto3->watson-machine-learning-client) (0.10.0)
Requirement already satisfied: s3transfer<0.6.0,>=0.5.0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from boto3->watson-machine-learning-client) (0.5.0)
Requirement already satisfied: python-dateutil<3.0.0,>=2.1 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from botocore<1.22.0,>=1.21.21->boto3->watson-machine-learning-client) (2.8.2)
Requirement already satisfied: six>=1.5 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from python-dateutil<3.0.0,>=2.1->botocore<1.22.0,>=1.21.21->boto3->watson-machine-learning-client) (1.15.0)
Requirement already satisfied: ibm-cos-sdk-s3transfer==2.11.0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from ibm-cos-sdk->watson-machine-learning-client) (2.11.0)
Requirement already satisfied: ibm-cos-sdk-core==2.11.0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from ibm-cos-sdk->watson-machine-learning-client) (2.11.0)
Requirement already satisfied: idna<4,>=2.5 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from requests->watson-machine-learning-client) (3.3)
Requirement already satisfied: charset-normalizer~=2.0.0 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from requests->watson-machine-learning-client) (2.0.4)
Requirement already satisfied: pytz>=2017.3 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from pandas->watson-machine-learning-client) (2021.3)
Requirement already satisfied: numpy>=1.17.3 in /opt/conda/envs/Python-3.9/lib/python3.9/site-packages (from pandas->watson-machine-learning-client) (1.20.3)
Connecting with IBM CLOUD
from ibm_watson_machine_learning import APIClient
wml_credentials = {"url":"https://us-south.ml.cloud.ibm.com", "apikey":"apikey"}
client = APIClient(wml_credentials)
Python 3.7 and 3.8 frameworks are deprecated and will be removed in a future release. Use Python 3.9 framework instead.
def guid_from_space_name(client,space_name):
    space = client.spaces.get_details()
    return(next(item for item in space['resources'] if item['entity']["name"]==space_name)['metadata']['id'])
space_uid = guid_from_space_name(client, 'spacename')
#space_uid
client.set.default_space(space_uid)
'SUCCESS'
software_spec_uid = client.software_specifications.get_uid_by_name("tensorflow_rt22.1-py3.9")
#software_spec_uid
client.software_specifications.list()
-----------------------------  ------------------------------------  ----
NAME                           ASSET_ID                              TYPE
default_py3.6                  0062b8c9-8b7d-44a0-a9b9-46c416adcbd9  base
kernel-spark3.2-scala2.12      020d69ce-7ac1-5e68-ac1a-31189867356a  base
pytorch-onnx_1.3-py3.7-edt     069ea134-3346-5748-b513-49120e15d288  base
scikit-learn_0.20-py3.6        09c5a1d0-9c1e-4473-a344-eb7b665ff687  base
spark-mllib_3.0-scala_2.12     09f4cff0-90a7-5899-b9ed-1ef348aebdee  base
pytorch-onnx_rt22.1-py3.9      0b848dd4-e681-5599-be41-b5f6fccc6471  base
ai-function_0.1-py3.6          0cdb0f1e-5376-4f4d-92dd-da3b69aa9bda  base
shiny-r3.6                     0e6e79df-875e-4f24-8ae9-62dcc2148306  base
tensorflow_2.4-py3.7-horovod   1092590a-307d-563d-9b62-4eb7d64b3f22  base
pytorch_1.1-py3.6              10ac12d6-6b30-4ccd-8392-3e922c096a92  base
tensorflow_1.15-py3.6-ddl      111e41b3-de2d-5422-a4d6-bf776828c4b7  base
autoai-kb_rt22.2-py3.10        125b6d9a-5b1f-5e8d-972a-b251688ccf40  base
runtime-22.1-py3.9             12b83a17-24d8-5082-900f-0ab31fbfd3cb  base
scikit-learn_0.22-py3.6        154010fa-5b3b-4ac1-82af-4d5ee5abbc85  base
default_r3.6                   1b70aec3-ab34-4b87-8aa0-a4a3c8296a36  base
pytorch-onnx_1.3-py3.6         1bc6029a-cc97-56da-b8e0-39c3880dbbe7  base
kernel-spark3.3-r3.6           1c9e5454-f216-59dd-a20e-474a5cdf5988  base
pytorch-onnx_rt22.1-py3.9-edt  1d362186-7ad5-5b59-8b6c-9d0880bde37f  base
tensorflow_2.1-py3.6           1eb25b84-d6ed-5dde-b6a5-3fbdf1665666  base
spark-mllib_3.2                20047f72-0a98-58c7-9ff5-a77b012eb8f5  base
tensorflow_2.4-py3.8-horovod   217c16f6-178f-56bf-824a-b19f20564c49  base
runtime-22.1-py3.9-cuda        26215f05-08c3-5a41-a1b0-da66306ce658  base
do_py3.8                       295addb5-9ef9-547e-9bf4-92ae3563e720  base
autoai-ts_3.8-py3.8            2aa0c932-798f-5ae9-abd6-15e0c2402fb5  base
tensorflow_1.15-py3.6          2b73a275-7cbf-420b-a912-eae7f436e0bc  base
kernel-spark3.3-py3.9          2b7961e2-e3b1-5a8c-a491-482c8368839a  base
pytorch_1.2-py3.6              2c8ef57d-2687-4b7d-acce-01f94976dac1  base
spark-mllib_2.3                2e51f700-bca0-4b0d-88dc-5c6791338875  base
pytorch-onnx_1.1-py3.6-edt     32983cea-3f32-4400-8965-dde874a8d67e  base
spark-mllib_3.0-py37           36507ebe-8770-55ba-ab2a-eafe787600e9  base
spark-mllib_2.4                390d21f8-e58b-4fac-9c55-d7ceda621326  base
autoai-ts_rt22.2-py3.10        396b2e83-0953-5b86-9a55-7ce1628a406f  base
xgboost_0.82-py3.6             39e31acd-5f30-41dc-ae44-60233c80306e  base
pytorch-onnx_1.2-py3.6-edt     40589d0e-7019-4e28-8daa-fb03b6f4fe12  base
pytorch-onnx_rt22.2-py3.10     40e73f55-783a-5535-b3fa-0c8b94291431  base
default_r36py38                41c247d3-45f8-5a71-b065-8580229facf0  base
autoai-ts_rt22.1-py3.9         4269d26e-07ba-5d40-8f66-2d495b0c71f7  base
autoai-obm_3.0                 42b92e18-d9ab-567f-988a-4240ba1ed5f7  base
pmml-3.0_4.3                   493bcb95-16f1-5bc5-bee8-81b8af80e9c7  base
spark-mllib_2.4-r_3.6          49403dff-92e9-4c87-a3d7-a42d0021c095  base
xgboost_0.90-py3.6             4ff8d6c2-1343-4c18-85e1-689c965304d3  base
pytorch-onnx_1.1-py3.6         50f95b2a-bc16-43bb-bc94-b0bed208c60b  base
autoai-ts_3.9-py3.8            52c57136-80fa-572e-8728-a5e7cbb42cde  base
spark-mllib_2.4-scala_2.11     55a70f99-7320-4be5-9fb9-9edb5a443af5  base
spark-mllib_3.0                5c1b0ca2-4977-5c2e-9439-ffd44ea8ffe9  base
autoai-obm_2.0                 5c2e37fa-80b8-5e77-840f-d912469614ee  base
spss-modeler_18.1              5c3cad7e-507f-4b2a-a9a3-ab53a21dee8b  base
cuda-py3.8                     5d3232bf-c86b-5df4-a2cd-7bb870a1cd4e  base
autoai-kb_3.1-py3.7            632d4b22-10aa-5180-88f0-f52dfb6444d7  base
pytorch-onnx_1.7-py3.8         634d3cdc-b562-5bf9-a2d4-ea90a478456b  base
-----------------------------  ------------------------------------  ----
Note: Only first 50 records were displayed. To display more use 'limit' parameter.
model_details = client.repository.store_model(model = 'body.tgz' , meta_props = {
    client.repository.ModelMetaNames.NAME : "body", 
    client.repository.ModelMetaNames.TYPE : "tensorflow_rt22.1",
    client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: software_spec_uid
})
model_id = client.repository.get_model_id(model_details)
client.repository.download(model_id, 'body_cloud.tar.gz')
Successfully saved model content to file: 'body_cloud.tar.gz'
'/home/wsuser/work/body_cloud.tar.gz'
model_body = load_model('body.h5')
MODEL FOR LEVEL TYPE DETECTION
trainPath = '/home/wsuser/work/Dataset/level/training'
testPath = '/home/wsuser/work/Dataset/level/validation'
training_set = train_datagen.flow_from_directory(trainPath,target_size=(244,244),batch_size=10,class_mode='categorical')
test_set = train_datagen.flow_from_directory(testPath,target_size=(244,244),batch_size=10,class_mode='categorical')
Found 979 images belonging to 3 classes.
Found 171 images belonging to 3 classes.
training_set.class_indices
{'01-minor': 0, '02-moderate': 1, '03-severe': 2}
Declaring Model Variable
vgg=VGG16(input_shape=(244,244,3),weights='imagenet',include_top=False)



for layer in vgg.layers:
  layer.trainable=False


x=Flatten()(vgg.output)


prediction=Dense(3,activation='softmax')(x)


model1=Model(inputs=vgg.input,outputs=prediction)
model1.summary()
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 244, 244, 3)]     0         
                                                                 
 block1_conv1 (Conv2D)       (None, 244, 244, 64)      1792      
                                                                 
 block1_conv2 (Conv2D)       (None, 244, 244, 64)      36928     
                                                                 
 block1_pool (MaxPooling2D)  (None, 122, 122, 64)      0         
                                                                 
 block2_conv1 (Conv2D)       (None, 122, 122, 128)     73856     
                                                                 
 block2_conv2 (Conv2D)       (None, 122, 122, 128)     147584    
                                                                 
 block2_pool (MaxPooling2D)  (None, 61, 61, 128)       0         
                                                                 
 block3_conv1 (Conv2D)       (None, 61, 61, 256)       295168    
                                                                 
 block3_conv2 (Conv2D)       (None, 61, 61, 256)       590080    
                                                                 
 block3_conv3 (Conv2D)       (None, 61, 61, 256)       590080    
                                                                 
 block3_pool (MaxPooling2D)  (None, 30, 30, 256)       0         
                                                                 
 block4_conv1 (Conv2D)       (None, 30, 30, 512)       1180160   
                                                                 
 block4_conv2 (Conv2D)       (None, 30, 30, 512)       2359808   
                                                                 
 block4_conv3 (Conv2D)       (None, 30, 30, 512)       2359808   
                                                                 
 block4_pool (MaxPooling2D)  (None, 15, 15, 512)       0         
                                                                 
 block5_conv1 (Conv2D)       (None, 15, 15, 512)       2359808   
                                                                 
 block5_conv2 (Conv2D)       (None, 15, 15, 512)       2359808   
                                                                 
 block5_conv3 (Conv2D)       (None, 15, 15, 512)       2359808   
                                                                 
 block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0         
                                                                 
 flatten (Flatten)           (None, 25088)             0         
                                                                 
 dense (Dense)               (None, 3)                 75267     
                                                                 
=================================================================
Total params: 14,789,955
Trainable params: 75,267
Non-trainable params: 14,714,688
_________________________________________________________________
model1.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
Training model
y = model1.fit_generator(
    training_set,
    validation_data = test_set,
    epochs = 25,
    steps_per_epoch=979//10,
    validation_steps = 171//10
)
/tmp/wsuser/ipykernel_7116/2253460449.py:1: UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.
  y = model1.fit_generator(
Epoch 1/25
97/97 [==============================] - 328s 3s/step - loss: 1.1968 - acc: 0.5614 - val_loss: 1.0383 - val_acc: 0.6000
Epoch 2/25
97/97 [==============================] - 324s 3s/step - loss: 0.7241 - acc: 0.7265 - val_loss: 0.8500 - val_acc: 0.6471
Epoch 3/25
97/97 [==============================] - 323s 3s/step - loss: 0.5437 - acc: 0.7967 - val_loss: 0.9910 - val_acc: 0.5941
Epoch 4/25
97/97 [==============================] - 324s 3s/step - loss: 0.3847 - acc: 0.8596 - val_loss: 0.9415 - val_acc: 0.6471
Epoch 5/25
97/97 [==============================] - 323s 3s/step - loss: 0.3124 - acc: 0.8906 - val_loss: 1.1811 - val_acc: 0.5882
Epoch 6/25
97/97 [==============================] - 323s 3s/step - loss: 0.2261 - acc: 0.9112 - val_loss: 1.2515 - val_acc: 0.6294
Epoch 7/25
97/97 [==============================] - 323s 3s/step - loss: 0.1783 - acc: 0.9412 - val_loss: 1.2657 - val_acc: 0.5824
Epoch 8/25
97/97 [==============================] - 326s 3s/step - loss: 0.1439 - acc: 0.9494 - val_loss: 1.2686 - val_acc: 0.5647
Epoch 9/25
97/97 [==============================] - 325s 3s/step - loss: 0.1028 - acc: 0.9742 - val_loss: 1.1297 - val_acc: 0.5941
Epoch 10/25
97/97 [==============================] - 324s 3s/step - loss: 0.0942 - acc: 0.9721 - val_loss: 1.1987 - val_acc: 0.5706
Epoch 11/25
97/97 [==============================] - 322s 3s/step - loss: 0.1088 - acc: 0.9639 - val_loss: 1.7247 - val_acc: 0.6118
Epoch 12/25
97/97 [==============================] - 323s 3s/step - loss: 0.1039 - acc: 0.9690 - val_loss: 1.1907 - val_acc: 0.5882
Epoch 13/25
97/97 [==============================] - 322s 3s/step - loss: 0.0671 - acc: 0.9814 - val_loss: 1.2263 - val_acc: 0.6471
Epoch 14/25
97/97 [==============================] - 322s 3s/step - loss: 0.0817 - acc: 0.9732 - val_loss: 1.3644 - val_acc: 0.5765
Epoch 15/25
97/97 [==============================] - 323s 3s/step - loss: 0.0664 - acc: 0.9763 - val_loss: 1.2678 - val_acc: 0.6235
Epoch 16/25
97/97 [==============================] - 323s 3s/step - loss: 0.0514 - acc: 0.9907 - val_loss: 1.2850 - val_acc: 0.6294
Epoch 17/25
97/97 [==============================] - 326s 3s/step - loss: 0.0254 - acc: 0.9990 - val_loss: 1.4064 - val_acc: 0.5824
Epoch 18/25
97/97 [==============================] - 335s 3s/step - loss: 0.0462 - acc: 0.9886 - val_loss: 1.3600 - val_acc: 0.6176
Epoch 19/25
97/97 [==============================] - 348s 4s/step - loss: 0.0294 - acc: 0.9959 - val_loss: 1.4163 - val_acc: 0.6059
Epoch 20/25
97/97 [==============================] - 332s 3s/step - loss: 0.0376 - acc: 0.9886 - val_loss: 1.4418 - val_acc: 0.6118
Epoch 21/25
97/97 [==============================] - 324s 3s/step - loss: 0.0315 - acc: 0.9979 - val_loss: 1.3438 - val_acc: 0.5941
Epoch 22/25
97/97 [==============================] - 325s 3s/step - loss: 0.0261 - acc: 0.9959 - val_loss: 1.3276 - val_acc: 0.6471
Epoch 23/25
97/97 [==============================] - 326s 3s/step - loss: 0.0178 - acc: 0.9990 - val_loss: 1.3663 - val_acc: 0.5941
Epoch 24/25
97/97 [==============================] - 327s 3s/step - loss: 0.0157 - acc: 0.9979 - val_loss: 1.3891 - val_acc: 0.6000
Epoch 25/25
97/97 [==============================] - 331s 3s/step - loss: 0.0334 - acc: 0.9948 - val_loss: 1.7455 - val_acc: 0.5824
model1.save('level.h5')
!tar -zcvf level.tgz level.h5
level.h5
ls -1 
body_cloud.tar.gz
body.h5
body.tgz
Dataset/
level.h5
level.tgz
model_details = client.repository.store_model(model = 'level.tgz' , meta_props = {
    client.repository.ModelMetaNames.NAME : "level", 
    client.repository.ModelMetaNames.TYPE : "tensorflow_rt22.1",
    client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: software_spec_uid
})
model_id = client.repository.get_model_id(model_details)
client.repository.download(model_id, 'level_cloud.tar.gz')
model_body = load_model('level.h5')
os.listdir('.')
['level.h5',
 '.virtual_documents',
 'level_cloud.tar.gz',
 'level.tgz',
 'body.h5',
 'body.tgz',
 'Dataset',
 'body_cloud.tar.gz']
client.repository.download('model_id','body_cloud.tar.gz')
client.repository.download('model1_id','level_cloud.tar.gz')
Footer
