# Detection of Grice’s Maxims Violations: A Deep Learning Approach Using the CMV Dataset
Our research aims to create a computational model that can identify breaches of Grice's maxims in conversational discourse, focusing specifically on relevance and manner.

## [Fine-tuning FLAN-T5 Model](finetune_s2s_model.py)
This code is training the Flan-T5-Base model on Manner Violation Detection without augmentations.

In order to finetune on Relevance Violation Detection please change line 71 to get_relevance_class.

To use augmentations please uncomment lines 100-123.

## Results
| Detection | Model Name                      | Class 0 - Precision | Class 0 - Recall | Class 0 - F1-Score | Class 1 - Precision | Class 1 - Recall | Class 1 - F1-Score | Accuracy |
|-----------|--------------------------------|----------------------|------------------|---------------------|----------------------|------------------|---------------------|----------|
| Relevance | GPT API                        | 0.923                | 0.431            | 0.593               | 0.071                | 0.523            | 0.12                | 0.441    |
| Relevance | Over Sampling - Linear SVM     | 0.949                | 0.428            | 0.59                | 0.078                | 0.656            | 0.133               | 0.441    |
| Relevance | **Over Sampling - Flan-T5**   | 0.96                 | 0.95             | 0.95                | 0.28                 | 0.32             | **0.29**            | **0.91**     |
| Manner    | GPT API                        | 0.8                  | 0.5              | 0.64                | 0.16                 | 0.59             | 0.25                | 0.51     |
| Manner    | **Under Sampling - Flan-T5**  | 0.92                 | 0.66             | 0.77                | 0.21                 | 0.62             | **0.31**            | **0.66**     |


## [Read More](Grice_s_Maxims_Violations__Deep_Learning_Approach_for_Detection_and_User_Guidance_for_CMV_Dataset.pdf)
