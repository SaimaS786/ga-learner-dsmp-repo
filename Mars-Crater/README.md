### Project Overview

 This project includes the technique described by L. Bandeira (Bandeira, Ding, Stepinski. 2010.Automatic Detection of Sub-km Craters Using Shape and Texture Information). We will identify crater candidates in the image using the pipeline. Each crater candidate image block is normalized to a standard scale of 48 pixels. Each of the nine kinds of image masks probes the normalized image block in four different scales of 12 pixels, 24 pixels, 36 pixels, and 48 pixels, with a step of a third of the mask size (meaning 2/3 overlap). We totally extract 1,090 Haar-like attributes using nine types of masks as the attribute vectors to represent each crater candidate. The dataset was converted to the Weka ARFF format by Joseph Paul Cohen in 2012. We will determine if the instance is a crater or not a crater. (1=Crater, 0=Not Crater)


### Learnings from the project

 After completing this project I learnt how Decision tree and Random Forest are used to determine a particular instance.


### Approach taken to solve the problem

 Various concepts of machine learning were taken into consideration by me while solving the problem.


