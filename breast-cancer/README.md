This folder contains analysis of the breast cancer dataset that can be taken from sklearn.

Step 1. 

The dataset contains data on 569 instances of patients who were diagnosed with malignant or benign cancer. True class distribution is 212 - malignant and 357 - benign. The number of attributes/features is 30. 

Hence, the problem this dataset invites us to investigate is the following:

We want to be able to use the data from screenings to learn about the type of cancer early enough to be able to deal with it at the early stages. The motivation is actually pretty cool for datasets like these, because, in fact, sitting in front of your computer and using the tools available, one can actually help save a person's life potentially. Awesome.

How would I solve the problem in real life?

I would obtain a degree of domain knowledge and consult with specialists in the area. We would list the potential features that seem to be important to professionals in the field using their broad domain knowledge. Then, we would design the size and the properties of the database to be collected. In that case, we could enlist the variable types that we want to see for each variable. Then we would conduct preliminary data analysis and see what these results entail. At this point, it seems to be wise to interpret results together with the specialists to see how the experiment is turning out and etc. Eventually, the experiment will be finalised and we will report back with some findings and potentially try and use some newly gained knowledge in the field and test it out.

Step 2.

Since this is a toy dataset, it is well prepared for analysis already unlike the real datasets. No missing values are present. At this point, potential rescaling of data is possible, also one would check some summary statistics on the data and plot some graphs so visualize the data. The latter can help spot some patterns and may give us some intuition. One should also check the need for variable transformations, i.e. from string type variables to categorical.

Step 3.

We already know that this problem falls into the domain of supervised learning, specifically, this is a classification problem where the target variable is coded 0(Benign) and 1 (Malignant). Therefore, our search for algorithms is based on different classification methods. At this point, the ML specialist needs to think about how different algorithms may or may not be suitable for the problem at hand, taking into account it's specificity. Typical candidates include KNN, SVM, Logistic regression, Naive Bayes, Decition trees and so on. For these algorithms parameter tuning needs to take place which is often performed by brute force when we simply try out models with different values for hyperparameters.

Step 4.

Reflect upon the results. Try different metrics for the quality of models. Tweak the choice of algorithms or parameters. Re-run the model.

Step 5. 

Analyse the results and present findings.
