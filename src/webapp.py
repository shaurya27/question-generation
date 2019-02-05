from predict import *

from flask import Flask,request, jsonify
from predict import *

app = Flask(__name__)
@app.route('/question_generation', methods=['POST'])
def question_generate():
    data = request.get_json()
    #print data
    p,s = user_input(data)
    print p.size()
    result = str(evaluate(p,s,enc1,enc2,dec))
    print result
    return jsonify({"generated_question": eval(result)})

if __name__ == '__main__':
      app.run(host='0.0.0.0', port= 4283)
