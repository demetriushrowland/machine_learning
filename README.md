# machine_learning

In the folder "Exam Taking with Mixture of Experts", you will find a mixture of experts model designed to answer exam questions consisting of quadratic polynomial minimization, identifying the correct unscrambling of a word, and reading comprehension. The experts used are an RNN and two language models respectively. Each problem is mapped to the corresponding expert via a router, which consists of a sentence embedding, followed by a linear model. We compare results against a benchmark, namely GPT-3.5, on a holdout set of 50 exam problems.

In the folder BETEL, you will find a replication of the results of the paper "Bayesian Estimation and Comparison of Moment Condition Models" as well as an implementation on a new data set. I have summarized the paper in the pdf "Bayesian Inference for MCMs" and compiled a slideshow in "Bayesian Analysis of MCMs."

In the folder Gaussian Processes, you will find the results of an implementation of 3-class GP classification on a data set consisting of the chest CT scans of patients infected with COVID-19, patients with viral pneumonia, and normal patients. The jupyter notebook entitled "main" contains the principal results and the pdf entitled "report" contains an analysis of these results.

In the folder Compilation, you will find a gathering of several programs that implement various machine learning algorithms on simulated data sets. These include algorithms for dimension reduction, regression, classification and density estimation.

In the folder Linear Algebra, you will find a header file written in C++ that implements various matrix calculations like matrix multiplication and the PALU decomposition.

