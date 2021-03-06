# Luan Castheloge's portifolio
Hi, I am a computer engineering graduated by Universidade Federal do Espirito Santo (UFES) at Brazil.
Data is everything nowadays and transform then in information is exciting!
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
It is a approach to modeling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables).
I developed a simple Python code with random data, to create a linear regression with Scikit Learn 

#### Natural language processing (NLP)
This role is, for me, the closest distance between computer languages and humans languages. The big challenges are generated and comprehension humans communication.
##### Spam classification
I took an old problem, spam messages. In this project, I had to work with the Natural Language Tool Kit (NLTK). I had a data set with many massages classified as SPAM or HAM.
First of all, I did data analysis and I realized that the messages format were a problem. The messages had punctuation and maybe cause some noise to the results. So I remove that punctuation. After that, I use stopword from NLTK, and remove then to the meaning easier to get. I separated all words and count, and I analyzed how many times this word show up on a SPAM/HAM massage. I used Naive Bayes to classify and do predictions.
##### Review Classification
In this analysis, I get a dataset with reviews (0 to 5). I started removed all reviews with 0. Second I separated all reviews by description length and star review. I got the text and star data to work with. Soon after, I used naive baive to make some predictions.
I used TfidfTransformer to correlate the most written words and reviews. The results were placed in a confusion matrix.

#### Principal Component Analysis (PCA)
Is the most used tool in exploratory data analysis and for making predictive models. It is often used to visualize genetic distance and relatedness between populations.
On this mini project, I analyzed the dataset about breast cancer, and I did some correlations.

#### Support Vector Machines
In this training algorithm, it’s possible builds a model making it a non-probabilistic binary linear classifier. I compared a confuse matrix before and after the SVC, and I could see the improvement.

#### Neural Network
A multilayer perceptron (MLP) is a class of artificial neural network that consists of, at least, three layers of nodes: an input layer, a hidden layer, and an output layer. This class utilizes a supervised learning technique called backpropagation for training. 
I wrote a python program that applied the MLP to the breast cancer problem. In the end, I am able to predict through the data if it is or is not cancer.

