Basic code for finetuning models with a hidden chain-of-thought.
The format is [Question] [COT] [Answer] where [Question] and [Answer] are taken from the dataset, and [COT] is generated by the model being finetuned.
The CE loss is computed only on the [Answer].
The KL regularization loss is computed on both [COT] and [Answer].