from flask import Flask
from flask import jsonify
app = Flask(__name__)

@app.route('/home/', methods=['GET', 'POST'])
def welcome():
    return "Hello, welcome to the home page."

@app.route('/home/<string:name>/')
def hello(name):
    return "Hello " + name
           
@app.route('/<int:number>/')
def incrementer(number):
    return "Incremented number is " + str(number+1)

@app.route('/json_test/')
def print_list():
    return jsonify(list(range(10)))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
