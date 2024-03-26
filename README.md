# Grice-s-Maxims-Violations-Deep-Learning-Approach-for-Detection-for-CMV-Dataset
Our research aims to create a computational model that can identify breaches of Grice's maxims in conversational discourse, focusing specifically on relevance and manner.

## [About Our Work](Grice_s_Maxims_Violations__Deep_Learning_Approach_for_Detection_and_User_Guidance_for_CMV_Dataset.pdf)
## [Fine-tuning FLAN-T5 Model](finetune_s2s_model.py)
This code is training the Flan-T5-Base model on Manner Violation Detection without augmentations.

In order to finetune on Relevance Violation Detection please change line 71 to get_relevance_class.

To use augmentations please uncomment lines 100-123.
