
import os
from flask import Flask, request, render_template,jsonify
import executable

app = Flask(__name__)

app = Flask(__name__,static_folder='static')

@app.route('/')
def index():
  return render_template('index.html')




@app.route('/i',methods = ["POST"])
def my_link():
    if request.method == "POST":
      data = request.get_json()
      result = data["result"]
      print(result)
      try:
        executable.some_magic(result)
      except Exception:
        print(Exception)
      return jsonify({"status":"success"})


if __name__ == '__main__':
  app.run(debug=True)