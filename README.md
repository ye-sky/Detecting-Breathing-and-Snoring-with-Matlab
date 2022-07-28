 Software Design:
MATLAB:
To accurately get and detect breathing and snoring with machine learning we used MATLAB We used Logistic Regression on MATLAB which is basically a map classification algorithm.the problem of, the variable (or output) y can only take discrete values for a given function (or input) set X and take that forward. The algorithm builds a regression model and for the specified data the items belong to the category numbered "1". Logistic regression uses a sigmoid function to model the data. 
𝑔 (𝑧) = 1 1 + 𝑒-𝑧 (5)

 This function maps a real value to a value between 0 and 1. Machine learning uses sigmoid to map to predicted probabilities. The cost function represents the purpose of the optimization. By writing and minimizing cost functions, you can minimize errors and develop accurate models. 

𝐽 (θ) = - 1 𝑚 Σ [𝑦 (𝑖) 𝑙𝑜𝑔 (ℎθ (𝑥 (𝑖) )) + (1 -𝑦 (𝑥 (𝑖) )

sigmoid functions. We are using gradient descent to reduce the cost function. Minimize the cost function by running the dip function on each parameter. 

θ𝑗 = θ𝑗-αΣ𝑚 𝑖 = 1 (ℎθ (𝑥 (𝑖) ) -𝑦 (𝑖) (𝑖) (𝑖) ) 𝑥

Our model provided an accuracy of 73% of our pulse shape. If the samples you are trying to classify are highly correlated, or is highly nonlinear, then the logistic regression coefficients do not predict the gain/loss well in each feature. 
SVM (Support Vector Machine) "Support Vector Machine" (SVM) is a machine learning algorithm for teachers. This algorithm plots each data item as a point in an n-dimensional space (n is how many features you have) and each feature value is a specific coordinate value. Then it performs classification by discovering a hyperplane that distinguishes the two classes well. The reasons for choosing to use SVM are many. Because the SVM uses a fraction of the training points (also called the support vectors) in the decision function, the is also more memory efficient. It is also versatile and can be specified as a decision function and built using various kernel features. When designing a SVM, we built the SVM using several common kernels with optimized hyperparameters. Kernel C Value Gamma Accuracy RBF 10 0.01 88% Linear 1 0.001 83% Sigmoid 0.1 1 43% Poly 100 0.001 73% .1 SVM44 Accuracy Table Results 

Forest (RF), as the name implies, consists of a number of separate decision trees that operate in an ensemble. It outputs individual tree class predictions for random forests, and the class with the most votes is the model prediction. Prediction tree sample. The basic concept behind Random Forest is simple yet powerful. The data science peak is why the random forest model works so well is that many relatively uncorrelated models (trees) acting as committees outperform any of the individually constructed models. It is important to have low correlations between models. The uncorrelated model can produce ensemble predictions that are more accurate than any of the individual predictions. 

The reason for this surprising effect is to protect the trees from each of the 22 individual errors (unless they are always all causing errors in the same orientation). Some trees may be wrong, but many others are correct, so with group the tree can move in the right direction. The Random Forest is a meta-estimator that combines multiple decision tree classifiers to for various subsamples of a data set, using averages to improve prediction accuracy and control over totals. The subsample size is always the same as the original input sample size, but the sample is drawn instead when bootstrapped. To design an individual estimator, you need parameters, such as: Estimators: Number of trees in the forest (n = 100)  max_depth: Maximum depth of the tree (the node was expanded when all leaves contained less than min_samples_split samples.)  min_samples_split: Minimum number of samples required to split internal nodes (minimum = 2) bootstrap: Whether to use several samples multiple times in a single tree (to use several samples) To evaluate the performance of a machine learning model multiple times in a single tree, it needs to be tested with some invisible data.As can be seen in figure 4.1 and 4.2 General. Cross-validation (CV) is one of the techniques used to validate machine learning models, though there are resampling steps used to evaluate models when data is limited. In order to run CV, you have to set aside data samples/portions that are not used to train the model. later use this sample for testing/validation. A common method used for crossover test is the Fold crossover test.
The Fold method can yield a model with less bias compared to compared to other methods. This is to ensure that all observations from the original data set can be represented in the training and test sets. Start by randomly splitting the entire data into Folds. Then use the K-1 (K minus 1) fold to fit the model.
 
![image](https://user-images.githubusercontent.com/59439727/181628771-82d661b1-97df-46cf-b3b7-f4e128cb7ae4.png)

Figure 4.1. MATLAB Breathing Simulation


![image](https://user-images.githubusercontent.com/59439727/181628801-25f72269-2ad3-4ba1-9942-be7b7536d9f5.png)

 Figure 4.2. MATLAB Snoring Simulation


The result is derived from a sample of the data stored in the file. In short, this method works for offline recorded breathing samples. Real-time processing of data is not possible. Noise and high amplitude glitches are still a challenge for the technology we have adopted. Background noise is avoided in certain situations as shown in figure 5.10. Therefore, it is difficult to completely remove background noise. We believe that a more stringent signal processing method is needed to solve this problem. Peak values indicate Dorsum and divergent instances. However, in any case, the peak detection algorithm is not powerful enough. This is also one of the problems caused by background noise and sudden defects. For good quality recordings, this technology works well. This was demonstrated in experiments performed as part of validation. This method has not been satisfactorily performed in long recordings, in this case an hour. This will let you know that the recording quality has deteriorated. Our observations have shown that it is difficult to detect such respirations when the amplitude of the respirations is very low as shown in figure 5.13. Therefore, some respirations have not been detected. Classification depends on the maximum value that occurs in the recording, so high-amplitude glitches can lead to erroneous results.
 ![image](https://user-images.githubusercontent.com/59439727/181629121-139e41ae-edc9-42a4-a3b1-8509939f31a9.png)

Figure 5.7.  Breathing raw signal

 ![image](https://user-images.githubusercontent.com/59439727/181629159-71e65b99-a75f-41bc-863f-b5947c36a3d3.png)

Figure 5.8.  Breathing envelops



![image](https://user-images.githubusercontent.com/59439727/181629218-22e4b05e-626e-4004-860e-6ec41fb45ef4.png)

 
Figure 5.9.  Breathing analysis


![image](https://user-images.githubusercontent.com/59439727/181629253-aa380ae4-3777-4925-a308-4f2b63323ca0.png)


 
Figure 5.10.  Breathing per minute


![image](https://user-images.githubusercontent.com/59439727/181629297-632a5a12-e8ed-40cd-a56a-74fc9159fc79.png)

 
Figure 5.11.  Snoring envelops

 ![image](https://user-images.githubusercontent.com/59439727/181629334-194f3d88-9480-4bd9-96da-dd06d7b20079.png)

Figure 5.12.  Snores per minute

![image](https://user-images.githubusercontent.com/59439727/181629362-ff8386da-f41a-4a70-8ae1-a374c210e54e.png)

 
Figure 5.13.  Snores absolute value
