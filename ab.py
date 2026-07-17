-1) manual trial and error method see if a feature increasing metric keep or move forward

0)Makin per user based feature selection till we get tar 1 and far 0 and taking intersection of all users featuresby making a ranking for each user based features 


1) margin metric based :: the first genuine session - the first imposter session score as metric and try to get this from -ve to +ve 
 then we can make user based threshold 


1b) Distribution separation like quantify
 how far apart the genuine and impostor score distributions are 
 and optimize the entire distributions 


2)is there something like we make rules like reward for feture that is maximizing 
or another rule like for finding pair of features that will increase margin for features that individually are not doing  then again reward like 
and like this can we make rules and make model learning to get more rewards like reward based learning 


3) Making subsets of features 
	3a)diversity like based on sensor make multiple subsets for, acc based gyro based etc and try including one or two from each subset 
	3b) make clusters and take one from each:: on what basis is the clustering done ?
	3c) what other type of subsets can we make ?


4) build model for feature selection like the svm does is for data based one is collectly classified and which ones mis qualified right 
now from the beginning without removing any feature we will do for features 
but the issue is for dat the classification or the label is definite and defined as if it is positive or negative 
but for a particular feature what this can be ?


5) take all features at the beginning 
next find each feature importance according to the user 
importance rules: feature that is giving score than a threshold for genuine / imposter session 
and write more rules which is similar to consensus ranking only but the ranking methos or the rules is very highly custom ones 

based on this we will get the importance and 1300+ features are ranked accordingly which is custom ranking and finally take intersection from all users  ranking


6) Goodest Validation:
louo leave one user out like once we make the set of features we will run louo and get which features are surviving for all runs and we will take them finally 
like we are validating the features 




7) combinations like 0 + 1 ie, user based feature selection then Ranking based on multiple things like high margin giving, score, general rankings fisher Boruta and more 



8) studying feature rankings and tweak where we need customizations 


9) use DL model to get features 


10) features that are really helping individually are really fine they will for sure go into final model 
feature that individually wont work work may work in combinations so

s1)find feature working individually ideally pure
s2) now study the situation how to make features useful and find features work in combinations 
q1)if a set of features work will there how they might disturb other sometimes ?
 like this divide the entire data set try doing different thing for each set 



Todo:
1.not all modules using only the module 4 try for eah module one run 
so we can change things inbetween like if module 2 does the feature selection in a way and the this is used for next module if we get other feature selection we can keep them here manually 

2) Hyper parameter tuning can we do this and feature selection separately ?
in this can we have custom rules like the margin adding with threshold ?

3)
