User Action Learning
===========================
classification of user using user behavior data

Tree
----
```python
    ├── data                                        ## sample_demo
    │   ├── test_data_demo                          ### consists of 1 sample
    │   └── train_data_demo                         ### consists of 1 sample
    ├── learning                                    ## Training
    │   ├── keras_user_behavior_demo.py             
    │   └── layers                                  
    │       ├── attention2.py                       ### customized attention layer
    │       └── __init__.py
    └── README.md

```
Sample Explanation
----
* Sample in data/train_data_demo is like：hc_c45d06de-2193-11e8-96de-00163e1020bd 220 2018-03-07  0   0   [[2467603.48, 0, 0, "16", "116"], [2467603.258, 0.2220001220703125, 0, "16", "119"], ... ]  
* They represent: sample_id \t length_of_list \t sample_date \t label \t sample_type \t user_behavior_sequence 
* user_behavior_sequence is like [[2467603.48, 0, 0, "16", "116"], [2467603.258, 0.2220001220703125, 0, "16", "119"], ... ], is variable length sequence (namely the length of user_behavior for each sample are not the same). Each behavior consists of 5 elements, which are xx, duration, os_type, page_id, point_id. The first element 'xx' can be ignored.

Training
----
* keras_user_behavior_demo.py
* the keras apis can be refered from documents: https://keras-cn.readthedocs.io/en/latest/  or https://keras.io

Requirements
---
keras==2.2.4  
tensorflow==1.11.0  
