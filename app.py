from flask import Flask, render_template, request
from llmUtil import OllamaLLM

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        llm = OllamaLLM(model="llama3.1")  # Use a valid model name
        user_input = request.form['user_input']
        response = llm.invoke(user_input)
        return render_template('index.html', response=response)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
