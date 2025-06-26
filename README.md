# Bear-Image-Classifier
deep learning-based image classifier. uses CNN. classify different types of bears (tensorflow, keras, streamlit)

---

# Images :

![](https://github.com/paramvarsha12/Bear-Image-Classifier/blob/c32fd41d8a2c3ccab1142100cc7ea68a1f03e9fc/Screenshot%202025-06-26%20123248.png
)
![](https://github.com/paramvarsha12/Bear-Image-Classifier/blob/c32fd41d8a2c3ccab1142100cc7ea68a1f03e9fc/Screenshot%202025-06-26%20123355.png
)
![](https://github.com/paramvarsha12/Bear-Image-Classifier/blob/c32fd41d8a2c3ccab1142100cc7ea68a1f03e9fc/Screenshot%202025-06-26%20153003.png)








---

# How it works : 
- *classify.py* is the main python file that trains the model to recognise different types of bears by learning from images
- We use a neural network (CNN) here to look at 50-100 images of each type of bear [grizzly, polar, pandas, black], and figures out what makes each bear different (suppose the eyes, nose and the fur)

1. first we download the dataset from kaggle [https://www.kaggle.com/datasets/hoturam/bear-dataset], download the zip file, extract it, and put the respective images in their own folder (ignore teddy)
2. you put all of the images first in the train folder
3. you put 10 images of each bear under the test folder (because we are using 20% test and 80% train)
4. we need to resize the image to (128,128) pixels and then we;ll feed images to the model in groups of 16 (batch_size=16)
5. then we change the pixel values from 0-255 to 0-1, which is more stable for the model to understand
6. we tell the model to treat it as a multi-class-classification problem since there are more than 2 types of bears
7. it will give an output something similar to [0, 1, 0, 0] → it's class 2, [1, 0, 0, 0] → it's class 1
8. we apply 32 filters of size 3x3 to extract low-level-features like edges and textures or fur from the input image
9. we use something called *maxPooling* to reduce the dimensions for the model to focus on the important features rather than the whole image
10. we then use something called "*flatten()*" to reduce the 2d outputs or matrix to a single 1d vector for the model to understand
11. we then use something called "*Dense(128, activation='relu')*" a connected layer of 128 neurons to interpret the extracted features
12. "*Dense(num_classes, activation='softmax')*" basically gives the probabilty of what type of bear it got as a result, so the model gives something called the *confidence score* and we get it in a probability of something between 0 and 1 using the "*softmax*" function
13. then we train the model using '*python classify.py*'
14. then the frontend is done in app.py
15. run it using '*streamlit run app.py*'

---

# Installations :
- install python version 3.10 (this is the only one that works with tensorflow, the later versions dont usually work to my understanding)
- set up a virtual environment in python
- run these commands to install : '*pip install tensorflow keras matplotlib numpy pillow scikit-learn*'

---

AUTHOR : PARAM VARSHA (26/06/2025)
