from webserver.server import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=30000, debug=False, use_reloader=False)