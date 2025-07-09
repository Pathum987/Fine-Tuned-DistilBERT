#step 1: Install necessary libraries
#next do steps in the dataset.py file

pip install transformers datasets torch tensorflow numpy


#step 6: Load pre-trained model and set up for sequence classification

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

#step 7: Prepare the dataset for training

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="no",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    save_strategy="epoch", 
    logging_dir="./logs",
    logging_steps=100,
    seed=42,
    
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

#step 8: Train the model

trainer.train()

#step 9: Save the model and tokenizer

trainer.save_model("my_imdb_model")
tokenizer.save_pretrained("my_imdb_model")

#step 10: Load the model and tokenizer for inference

from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained("my_imdb_model")
tokenizer = AutoTokenizer.from_pretrained("my_imdb_model")

#step 11: Create a text classification pipeline

from transformers import pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

#step 12: Use the pipeline for inference

reviews = [
    "Absolutely loved this movie!",
    "This film was boring and disappointing."
]
outputs = classifier(reviews)
print(outputs)

