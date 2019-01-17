# Luan Castheloge's portifolio
Hi, I am a computer engineering graduated by Universidade Federal do Espirito Santo (UFES) at Brazil.
Data is everything nowadays and transform then in information make me exciting!
I started to study data science in June 2017 and this repository is to help anyone else that wants to understand this huge and amazing role.  
So let’s explore!

## How this repository is organized?
I organized this portfolio by python tests, data analysis, and machine learning examples. After that, I put all my small projects separated by the used technics. In these small projects, I used some dataset or image taken from the web.
My approach is to show my skills through direct and objective practices about data science.


### Applications with some libraries ( :open_file_folder: libraries_applications)
These libraries (**Numpy, Panda and Seaborn**) are that I most like to use. They are so powerful and attend all my need for this portfolio.
In this folder, I show how to use some functionalities that helped me to create my projects.


### Data analysis ( :open_file_folder: data_analysis)
When I think about Data analysis, I immediately think about processes/subprocesses like: a process of inspecting, cleansing, transforming, and modeling data with the goal of discovering useful information, inferring conclusions and supporting decision-making.

#### Titanic Analysis
The Titanic analysis is a classic application to transform all these data in some information. In my algorithm, I check the relation about the type of cabin and age of passengers.
First of all, I fitted the date replacing my null values for the average of ages. Second I checked my data and results with a seaborn chart. Third I checked the relation between the sector embarked and if the passenger survived. Besides that, I did some analysis for the passengers by age, if they were responsible for someone and the fare.
Finally, I did a linear regression to the age and a possibility that the passenger survived.


### Machine Learning ( :open_file_folder: machine_learning_examples)
On this section, I applicated some machine learning techniques trying to make some predictions or decision based on data sets.

#### K-Nearest-Neighbors
In pattern recognition, the k-nearest neighbors algorithm (k-NN) is a non-parametric method used for classification and regression. This algorithm is among the simplest of all machine learning algorithms. It’s a useful technique can be used to assign weight to the contributions of the neighbors, so that the nearer neighbors contribute more to the average than the more distant ones. For example, a common weighting scheme consists in giving each neighbor a weight of 1/d, where d is the distance to the neighbor.
##### Iris problem
I created a solution for the classic Iris problem. Based on the Iris characteristics I applicated the KNN algorithm to classification what species of Iris I was trying to predict. So if I have data about the Iris I can tell what species it is.

#### K-Nearest-Clustering
Sometimes you maybe confuse k-nearest neighbors with k-nearest clustering. But the K-nearest clustering is popular for cluster analysis in data mining. k-means clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster.
##### Image filter by color
I really liked to create this solution because when I had a visual result become more exciting. This solution is about a filter image colors, I separated by three colors: yellow, green and blue. Before the algorithm application, I play with some functions from cv2 library to edit the main image exploring the processing image.
First of all, I created a function that separated what are the main colors present in an image. I worked to hexadecimal and RGB values, so I had to create a function to convert these values. After that, I mapped what files were an image on a specific folder. Finally, I created a function that matches these images by colors (previously defined YGB). If the image matches with any color I put there.


#### Linear regression
#### Natural language processing (NLP)
#### Principal Component Analysis (PCA)
#### Support Vector Machines
#### Neural Network
