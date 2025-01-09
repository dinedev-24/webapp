%%writefile app.py
# Paste the content of your Sentiment_analysis.py file here
# or if the file is in Google Drive, read it into the Colab notebook

with open('/content/Sentiment_analysis.py', 'r') as file:
    content = file.read()

with open('app.py', 'w') as f:
    f.write(content)
