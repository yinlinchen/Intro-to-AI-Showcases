Authors: Sai Puppala, Josh Klein, Sid Pothineni, Kenneth Hsu, Dani Tolessa
Date: 11/27/23

We created a sentiment analysis tool that analyzes a user's input for sentiment and emotion. It also
returns a category of website it thinks that input would fit best, and it stores each analysis of that
session in a brief history box.

Git Link: https://git.cs.vt.edu/joshmk31/intro-to-ai-sentiment-analysis

We used models from hugging face for this assignment.
Sentiment analysis model: https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student?text=the+sky+is+cloudy
Emotional analysis model: https://huggingface.co/SamLowe/roberta-base-go_emotions?
Website classification model: https://huggingface.co/alimazhar-110/website_classification?text=I+like+you.+I+love+you


To use this program, you must do the following:

pip install tk (this is tkinter)
pip install transformers
pip install tensorflow
pip install torch

* You may need to use py -m pip install ... if pip install does not work.

After installing these, you should be able to use the following command to run the program:

py ui.py



Troubleshooting:
    If you are having difficulty running the program, the following could be helpful:

    - If using VS Code, ensure the interpreter (botom right) is pointed at the location of the pip installs
    - Try "pip install --upgrade pip" if you are receiving errors, and try to pip install --upgrade the packages
    - Try "pip install --upgrade transformers" if you are receiving errors about the transformers package (or any other package)