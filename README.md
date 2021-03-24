# M6 Video Analysis - MCV

## Video Surveillance for Road Traffic Monitoring

### Team 7
| Members        |  Mail                           | Github |
| :---           | ---:                            | ---: |
| Alex Tempelaar | alexander.tempelaar@e-campus.uab.cat | Tempe25 |
| Víctor ubieto  | victor.ubieto@e-campus.uab.cat   | victorubieto |
| Mar Ferrer     | mar.ferrerf@e-campus.uab.cat  | namarsu |
| Antoni Rodríquez| antoni.rodriguez@e-campus.uab.cat  | antoniRodriguez |


## Project Schedule
## Week 1 ([Slides](https://docs.google.com/presentation/d/1kNpgATzLse7ZOE_rHp5N3c7yqyxgLaUbcgDiwzJ4isI/edit?usp=sharing))
- Task 1: Detection metrics.
- Task 2: Detection metrics. Temporal analysis.
- Task 3: Optical flow evaluation metrics.
- Task 4: Visual representation optical flow.

Process to run the code

1. Add paths to data files:
   - Task 1.1
     - Groundtruth xml path: --Lines 17
   - Task 1.2
     - Groundtruth xml path: --Lines 78
     - Prediction paths: --Lines 80-82
   - Task 2
     - Groundtruth xml path: --Line 105
     - Video path: --Line 106
     - Prediction paths: --Lines 108-110
   - Task 3
     - Flow prediction: --Line 148
     - Groundtruth flow: --Line 149
   - Task 4
     - Flow prediction: --Line 173
     - Groundtruth flow: --Line 174
     - Image path: --Line 175
2. Select the task to run leaving it uncommented: --Lines 184-188
3. Run: >> python lab1.py

## Week 2 
- Task 1: Gaussian modeling of the background for foreground extraction and evaluation. 
- Task 2: Adaptive modeling and evaluation. 
- Task 3: State-of-the-art methods for foreground extraction. 
- Task 4: Use color spaces to perform foreground extraction. 

Process to run the code

1. Add paths to data files:
   - Task 1.1
     - Groundtruth xml path: --Lines 11
     - Video path: --Lines 15
   - Task 1.2
     - The user variables are the same than in task 1.1
   - Task 2
     - Groundtruth xml path: --Line 51
     - Video path: --Line 55
     - Prediction paths: --Lines 108-110
   - Task 3
     - Groundtruth xml path: --Lines: 103
     - Video path: --Lines: 106
   - Task 4
     - The experiments were executed using the previous code. 
2. Select the task to run leaving it uncommented: --Lines 126-128
3. Run: >> python lab2.py

## Week 3 
- Task 1: Object detection
- Task 1.1: Off-the-shelf
- Task 1.2: Fine-tune to your data
- Task 1.3: K-Fold Cross Validation
- Task 2: Object tracking
- Task 2.1: Tracking by Overlap
- Task 2.2: Tracking with a Kalman Filter
- Task 2.3: IDF1 score

Process to run the code

1. Add paths to data files:
   - Task 1.1
     - Model name: --Line 21
     - Groundtruth xml path: --Lines 22
     - Video path: --Lines 23
   - Task 1.2_B (Cross_val technique B)
     - Hyperparameters: --Lines 72-74
     - Gt path: -- Line 78
     - Video path: --Line 79
   - Task 1.2_C  (Cross_val technique C)     
     - Hyperparameters: --Lines 195-197
     - Gt path: -- Line 201
     - Video path: --Line 202
   - Task 2.1
     - Pkl path (with bboxes and socores): --Lines: 316
     - Video path: --Lines: 317
     - Gt path: --Line 318
     - Parameters: --Lines 319-323
   - Task 2.2
     - Pkl path (with bboxes and socores): --Lines: 498
     - Video path: --Lines: 499
     - Gt path: --Line 500
     - Parameters: --Lines 501-502
2. Select the task to run leaving it uncommented: --Lines 626-630
3. Run: >> python lab3.py

