# Launch with
#
# gunicorn -D --threads 4 -b 0.0.0.0:5000 --access-logfile server.log --timeout 60 server:app glove.6B.300d.txt bbc

from flask import Flask, render_template
from doc2vec import *
import sys

app = Flask(__name__)



@app.route("/")
def articles():
    """Show a list of article titles"""
    # return render_template('articles.html')
    return render_template('articles.html', articles=articles,
                           topics=topics,filenames=filenames)




@app.route("/article/<topic>/<filename>")
def article(topic,filename):
    """
    Show an article with relative path filename. Assumes the BBC structure of
    topic/filename.txt so our URLs follow that.
    """
    for a in articles:
        if topic+'/'+filename == a[0]:
            text = a[2]
            title = a[1]
            break
    rec_articles = recommended(a,articles,n_best)
    rec_topics = [a[0].split('/')[0] for a in rec_articles]
    rec_filenames = [a[0].split('/')[1] for a in rec_articles]
    return render_template('article.html',topic2=topic,fname2=filename,text2=text,title2=title,
                           rec_articles2=rec_articles,rec_topics2=rec_topics,rec_filenames2=rec_filenames)




# initialization
glove_filename = './glove/glove.6B.300d.txt'
articles_dirname = './bbc'

# i = sys.argv.index('server:app')
# glove_filename = sys.argv[i+1]
# articles_dirname = sys.argv[i+2]

n_best = 5
gloves = load_glove(glove_filename)
articles = load_articles(articles_dirname, gloves)
topics = [a[0].split('/')[0] for a in articles]
filenames = [a[0].split('/')[1] for a in articles]

# app.run('0.0.0.0')
# app.run()