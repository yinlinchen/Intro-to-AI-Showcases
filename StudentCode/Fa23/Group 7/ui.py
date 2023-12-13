# used https://realpython.com/python-gui-tkinter/, https://docs.python.org/3/library/tkinter.ttk.html
# Emotion analysis: https://huggingface.co/SamLowe/roberta-base-go_emotions?
# Website claassification: https://huggingface.co/alimazhar-110/website_classification?text=I+like+you.+I+love+you
# Sentiment analysis: https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student?text=the+sky+is+cloudy

# Authors: Sai Puppala, Josh Klein, Sid Pothineni, Kenneth Hsu, Dani Tolessa
# Date: 11/27/23

import tkinter as tk
from tkinter import ttk
from transformers import pipeline
from tkinter import messagebox
sentiment_analysis_model = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_analysis_model)

emotion_analysis_model = "SamLowe/roberta-base-go_emotions"
emotion_analyzer = pipeline("sentiment-analysis", model=emotion_analysis_model)

website_class_model = "alimazhar-110/website_classification"
website_class_analyzer = pipeline("text-classification", model=website_class_model)

history = []

def submit_for_analysis():
    input_text = text_entry.get("1.0", tk.END).strip()
    
    # get teh results of the used models
    sentiment_result = sentiment_analyzer(input_text)
    emotion_result = emotion_analyzer(input_text)
    website_result = website_class_analyzer(input_text)


    # Display the results of sentiment analysis
    sentiment_label = sentiment_result[0]["label"].capitalize()
    sentiment_score = round(sentiment_result[0]["score"], 2)

    emotion_label = emotion_result[0]["label"].capitalize()
    emotion_score = round(emotion_result[0]["score"], 2)

    sentiment_result_text.set(f"Sentiment: {sentiment_label} (Intensity: {sentiment_score})\nEmotion: {emotion_label} (Confidence: {emotion_score})")

    # Change background color to red if negative, green if positive
    if sentiment_label.lower() == "negative":
        sentiment_color = "#f75f52" # a light red
    elif sentiment_label.lower() == "positive":
        sentiment_color = "#4cf579" # a light green
    else:
        sentiment_color = "#dae86f" # a light yellow

    sentiment_result_label.config(background=sentiment_color)


    # Display results of website analysis
    website_label = website_result[0]["label"]
    website_score = round(website_result[0]["score"], 2)
    website_results_text.set(f"Best fit Website category: {website_label} (Confidence: {website_score})")

    history.append((input_text, sentiment_label, sentiment_score, emotion_label, emotion_score))
    update_history_display()

    #Determine potential warning output for emotion
    if (emotion_label == "Anger" or emotion_label == "Annoyance" or
        emotion_label == "Disgust" or emotion_label == "Dissapointment") and emotion_score > 0.5:
        messagebox.showwarning("Warning", f"Our Language Model has detected a high level of {emotion_label.lower()}. Please ensure your language is respectful.")



def update_history_display():
    history_text.config(state=tk.NORMAL)  # enable the widget before update
    history_text.delete("1.0", tk.END)  # clear the current to be replaced

    for i, entry in enumerate(reversed(history)):
        input_text, sentiment_label, sentiment_score, emotion_label, emotion_score = entry
        history_text.insert(tk.END, f"{i+1}. \"{input_text}\" - Sentiment: {sentiment_label} ({sentiment_score}), Emotion: {emotion_label} ({emotion_score})\n\n")

    history_text.config(state=tk.DISABLED)  # disable widget after update

# Create the main window
window = tk.Tk()
window.geometry("700x700")
window.title("Sentiment Analyzer")

# setting the style
style = ttk.Style()
style.theme_use("clam")

# creates the frame for the header
header_frame = ttk.Frame(window, padding="10")
header_frame.pack(fill="x")

# header label
header_label = ttk.Label(header_frame, text="Sentiment Analyzer", font=("Helvetica", 16))
header_label.pack()

# frame for user input
text_frame = ttk.Frame(window, padding="10")
text_frame.pack(fill="x")

# user input text box
text_entry = tk.Text(text_frame, width=80, height=10)
text_entry.pack(padx=5, pady=5)

# buttons Frame
buttons_frame = ttk.Frame(window, padding="10")
buttons_frame.pack(fill="x")

# creates analyze button
analyze_button = ttk.Button(buttons_frame, text="Analyze", command=submit_for_analysis)
analyze_button.pack(side="left", padx=5)

# create exit button
exit_button = ttk.Button(buttons_frame, text="Exit", command=window.quit)
exit_button.pack(side="right", padx=5)

# frame for analysis results
result_frame = ttk.Frame(window, padding="10")
result_frame.pack(fill="both", expand=True)

# text output of sentiment analysis
sentiment_result_text = tk.StringVar()
sentiment_result_label = ttk.Label(result_frame, textvariable=sentiment_result_text, font=("Helvetica", 12), background="#DDDDDD", wraplength=500)
sentiment_result_label.pack(padx=5, pady=5, fill="both")

# text output of website classification
website_results_text = tk.StringVar()
website_results_label = ttk.Label(result_frame, textvariable=website_results_text, font=("Helvetica", 12), background="#DDDDDD", wraplength=500)
website_results_label.pack(padx=5, pady=5, fill="both")


# creates history frame
history_frame = ttk.Frame(window, padding="10")
history_frame.pack(fill="both", expand=True)

history_label = ttk.Label(history_frame, text="Analysis History", font=("Helvetica", 12))
history_label.pack()

scrollbar = ttk.Scrollbar(history_frame)
scrollbar.pack(side="right", fill="y")

history_text = tk.Text(history_frame, width=80, height=5, yscrollcommand=scrollbar.set)
history_text.pack(side="left", padx=5, pady=5, fill="both", expand=True)
scrollbar.config(command=history_text.yview)

window.mainloop()
