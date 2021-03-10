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

