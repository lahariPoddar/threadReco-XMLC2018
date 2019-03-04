# threadReco-XMLC2018

Sources for our paper titled 

"Cold Start Thread Recommendation as Extreme Multi-label Classification",  Extreme Multilabel Classification for Social Media, WWW, 2018.

Kindly cite our paper if you use the sources.

```
@inproceedings{halder2018cold,
  title={Cold Start Thread Recommendation as Extreme Multi-label Classification},
  author={Halder, Kishaloy and Poddar, Lahari and Kan, Min-Yen},
  booktitle={Companion of the The Web Conference 2018 on The Web Conference 2018},
  pages={1911--1918},
  year={2018},
  organization={International World Wide Web Conferences Steering Committee}
}
```

## Files

* threadReco.py : Contains the python code for main model.
* threadReco_data.py : Contains data processing functions. 

## Instructions to run

* Required Data Files: 
  1) Embedding file: The code expects a GloVe pre-trained embedding file in the main folder. The embedding file can be downloaded from https://nlp.stanford.edu/projects/glove/
  2) Input Data File: A '.tsv' file containing the following fields with tab as the delimeter.
      * postID : An ID of the thread (not used by the model. For cross-reference purposes)
      * postText : textual content of the thread
      * users: List of comma-separated user ids of users who have interacted with the post

* To train and test the model, run the following

      python3 threadReco.py
    
 ## Requirements
 
 The following packages are required to run the code
 
 * Python 3
 * Tensorflow
 * Keras
 
 
