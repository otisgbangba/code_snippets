

# Code snippets

A collection of useful code snippets, including code for Python, R, PHP, SQL and Bash.

Version: 7.16 (2020-07-04)

© 1999—2020, Kalinin Alexandr



## Machine Learning

### R (Classification)

```
data <- read.csv("example.csv")
```


```
ncol(data)
```


```
nrow(data)
```


```
head(data, 5)
```


```
tail(data, 5)
```


```
summary(data)
```


```
cor(data)
```


```
table(data$target)
```


```
aggregate(data, list(data$target), mean)
```


```
data$target <- as.factor(data$target)
colors <- c("#5580E2", "#D2381D")
plot(data, col = colors[data$target])
```


```
hist(data$alpha, col = "#5D82B9", border = "black")
```


```
library(ggplot2)

ggplot(data, aes(alpha, beta, colour = target)) + 
    geom_point()
```


```
library(randomForest)

model <- randomForest(
    target ~ ., 
    data = data, 
    importance = TRUE,
    proximity = TRUE,
    ntree = 500
)

model$type
model$confusion
importance(model, type = 2)
```


### Python (Classification)

```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
```


```
dataset = pd.read_csv('example.csv')
```


```
dataset.shape
```


```
dataset.info()
```


```
dataset.head(4)
```


```
dataset.tail(4)
```


```
dataset.sample(4)
```


```
dataset.describe()
```


```
dataset.corr()
```


```
dataset.groupby('target').median()
```


```
dataset.groupby('target').agg(['mean', 'std'])
```


```
dataset.hist(bins=20, figsize=(10, 10))
plt.show()
```


```
sns.heatmap(dataset.corr(), square=True, annot=True)
plt.show()
```


```
sns.pairplot(dataset, hue='target', vars=dataset.columns[1:])
plt.show()
```


```
sns.relplot(x='alpha', y='beta', hue='target', data=dataset)
plt.show()
```


```
sns.jointplot(x='alpha', y='beta', kind='kde', data=dataset)
plt.show()
```


```
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
```


```
data = np.loadtxt('example.csv', delimiter=',', skiprows=1)
model = RandomForestClassifier(n_estimators=1000, max_depth=100)
print(cross_val_score(model, data[:, 1:], data[:, 0], cv=5, scoring='f1'))
```


### Python (Image recognition)

```
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import decode_predictions
```


```
model = ResNet50(weights='imagenet')
```


```
def predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = np.expand_dims(image.img_to_array(img), axis=0)
    preds = model.predict(preprocess_input(x))
    title = decode_predictions(preds, top=1)[0][0][1]
    plt.title(title)
    plt.imshow(img)
    plt.show()
```


### Python (Clustering, Dimensionality reduction)

```
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
```


```
iris = load_iris()
target = iris.target
X = preprocessing.scale(iris.data)
```


```
kmeans = KMeans(n_clusters=3).fit(X)
X_tr = PCA(n_components=2).fit_transform(X)
```


```
plt.scatter(X_tr[:, 0], X_tr[:, 1], c=target)
plt.show()
```


```
plt.scatter(X_tr[:, 0], X_tr[:, 1], c=kmeans.labels_)
plt.show()
```


### Python (Pandas)

```
import re
import pandas as pd
from sqlalchemy import create_engine
```


```
engine = create_engine('sqlite:///demo.db', echo=False)
```


```
raw = [
    'id:1 R:2 =red',
    'id:2 R:3 =red',
    'id:3 R:3 =red',
    'id:4 R:4 =green',
    'id:5 R:7 =red',
    'id:6 R:6 =red',
    'id:7 R:6 =red',
    'id:8 R:5 =green',
    'id:9 R:4 =green',
    'id:10 R:3 =green',
    'id:11 R:2 =red',
    'id:12 R:1 =green',
]
```


```
def parse(line):
    match = re.search('id:(\d+) R:(\d+) =(\w+)', line)
    id = int(match.group(1))
    score = int(match.group(2))
    type = match.group(3)
    return {'id': id, 'score': score, 'type': type}
```


```
data = pd.DataFrame(map(parse, raw)).set_index('id')
```


```
data['type'] = data['type'].map({'red': 1, 'green': 0})
```


```
data['score'] = data['score'].apply(lambda x: x - 3)
```


```
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def classifier(x):
    y = 0.041 + (0.061 * x)
    return int(sigmoid(y) > 0.5)


data['target'] = data['score'].apply(classifier)
```


```
data.to_csv('_temp.csv', index=False)
```


```
data.to_excel('_temp.xlsx') 
```


```
data.to_json('_temp.json', orient='records')
```


```
data.to_html('_temp.html')
```


```
data.to_sql('example', con=engine)
```


```
sql = """

SELECT * FROM example;

"""


pd.read_sql(sql, con=engine)
```


```
sql = """

SELECT 
    type,
    MIN(score) AS score_min,
    AVG(score) AS score_mean,
    MAX(score) AS score_max
FROM example
GROUP BY type
ORDER BY type DESC;

"""


pd.read_sql(sql, con=engine)
```


```
sql = """

SELECT 
    type,
    score,
    SUM(score) OVER (PARTITION BY type) total_scores,
    DENSE_RANK() OVER (ORDER BY score) dense_rank
FROM example
ORDER BY dense_rank DESC;

"""


pd.read_sql(sql, con=engine)
```


### Python (Helpers)

```
import re
import json
import requests
from bs4 import BeautifulSoup
```


```
def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as jsonfile:
        return json.load(jsonfile)
```


```
def save_json(filename, data):
    with open(filename, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True, ensure_ascii=False)
```


```
def fetch(url, headers):
    response = requests.get(url=url, headers=headers)
    return response.content
```


```
def parse_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.title.string
    text = soup.get_text()
    links = []
    for link in soup.find_all('a'):
        links.append(link.get('href'))
    return {'title': title, 'text': text, 'links': links}
```




## Natural language processing (corpus)

### Python (Django)

```
#!/bin/bash


cat > requirements.txt <<EOL
django==3.0.7
psycopg2==2.8.5
djangorestframework==3.11.0
markdown==3.2.2
django-filter==2.3.0
EOL


cat > Dockerfile <<EOL
FROM python:3
ENV PYTHONUNBUFFERED 1
RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/
RUN pip install -r requirements.txt
COPY . /code/
EOL


cat > docker-compose.yml <<EOL
version: '3'

services:
  db:
    image: postgres
    environment:
      - POSTGRES_DB=postgres
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
    ports:
      - "5432:5432"
  redis:
    image: "redis:alpine"
    ports:
      - "6379:6379"
  web:
    build: .
    command: python manage.py runserver 0.0.0.0:8000
    volumes:
      - .:/code
    ports:
      - "8000:8000"
    depends_on:
      - db
      - redis
EOL


docker-compose build
docker-compose run web django-admin startproject application .
docker-compose up -d
docker-compose run web python manage.py startapp corpus
docker-compose run web python manage.py makemigrations
docker-compose run web python manage.py migrate
docker-compose logs web
chown -R $USER:$USER .
docker-compose run web python manage.py createsuperuser
```


```
from django.contrib import admin
from django.urls import include, path
from rest_framework import routers
from corpus import views


router = routers.DefaultRouter()
router.register(r'documents', views.DocumentViewSet)


urlpatterns = [
    path('', include(router.urls)),
    path('admin/', admin.site.urls)
]
```


```
from django.contrib import admin
from .models import Document


admin.site.register(Document)
```


```
from .models import Document
from rest_framework import viewsets
from .serializers import DocumentSerializer


class DocumentViewSet(viewsets.ModelViewSet):
    queryset = Document.objects.all().order_by('-id')
    serializer_class = DocumentSerializer
```


```
from .models import Document
from rest_framework import serializers


class DocumentSerializer(serializers.ModelSerializer):
    content_size = serializers.SerializerMethodField()
    
    def get_content_size(self, obj):
        return len(obj.content)
    
    class Meta:
        model = Document
        fields = ['id', 'target', 'content', 'content_size']
```


```
from django.db import models


class Document(models.Model):
    TARGETS = (
        (0, 'Bad'),
        (1, 'Good'),
    )
    target = models.IntegerField(default=TARGETS[0][0], choices=TARGETS)
    content = models.TextField()
    
    def __str__(self):
        return 'ID: {} (target: {})'.format(self.id, self.target)
```


### PHP (Laravel 7)

```
<?php

use Illuminate\Support\Facades\Route;

Route::get('/documents/{id}', 'DocumentController@show');
```


```
<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Repositories\DocumentRepository;

class DocumentController extends Controller
{
    /**
     * The document repository implementation.
     *
     * @var DocumentRepository
     */
    protected $documents;

    /**
     * Create a new controller instance.
     *
     * @param DocumentRepository $documents
     * @return void
     */
    public function __construct(DocumentRepository $documents)
    {
        $this->documents = $documents;
    }

    /**
     * @param int $id
     * @return string
     */
    public function show($id)
    {
        $document = $this->documents->find($id);
        
        return response()->json($document);
    }
}
```


```
<?php

namespace App\Repositories;

use App\Document;

class DocumentRepository 
{
    /**
     * @param int $id
     * @return Document
     */
    public function find($id)
    {
        return Document::find($id);
    }
}
```


```
<?php

namespace App\Console\Commands;

use Faker;
use App\Document;
use Illuminate\Console\Command;

class ImportDocuments extends Command
{
    /**
     * The name and signature of the console command.
     *
     * @var string
     */
    protected $signature = 'import:documents';

    /**
     * The console command description.
     *
     * @var string
     */
    protected $description = 'Import documents';

    /**
     * Create a new command instance.
     *
     * @return void
     */
    public function __construct()
    {
        parent::__construct();
    }

    /**
     * Execute the console command.
     *
     * @return mixed
     */
    public function handle()
    {
        $faker = Faker\Factory::create();
        
        for($i = 0; $i < 1000; $i++) {
            $document = new Document();
            $document->content = $faker->text;
            $document->target = rand(0, 1);
            $document->save();
        }
    }
}
```


```
<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

class CreateDocumentsTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('documents', function (Blueprint $table) {
            $table->id();
            $table->text('content');
            $table->unsignedTinyInteger('target');
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('documents');
    }
}
```




## Fetch internet resources

### Bash

```
curl -o 1.html -vvv -H "User-Agent: Bot" https://example.com
```


```
wget --debug -a main.log -O 2.html --user-agent="Bot" https://example.com
```


### PHP

```
class Bot
{
    /**
     * @param string $url
     * @param string $userAgent
     * @return string
     */
    public function fetch($url, $userAgent)
    {
        $handle = curl_init();
        curl_setopt($handle, CURLOPT_URL, $url);
        curl_setopt($handle, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($handle, CURLOPT_USERAGENT, $userAgent);
        curl_setopt($handle, CURLOPT_FOLLOWLOCATION, true);
        curl_setopt($handle, CURLOPT_CONNECTTIMEOUT, 5);
        curl_setopt($handle, CURLOPT_TIMEOUT, 5);
        $result = curl_exec($handle);
        curl_close($handle);
        
        return $result;
    }
}
```


### PHP

```
// composer require guzzlehttp/guzzle:~6.0
require 'vendor/autoload.php';

class Bot
{
    /**
     * @param string $url
     * @param string $userAgent
     * @return string
     */
    public function fetch($url, $userAgent)
    {
        $client = new GuzzleHttp\Client();
        $headers = ['User-Agent' => $userAgent];
        $res = $client->request('GET', $url, ['headers' => $headers]);
        
        return (string) $res->getBody();
    }
}
```


### PHP

```
class Bot
{
    /**
     * @param string $url
     * @return string
     */
    public function fetch($url)
    {
        return file_get_contents($url);
    }
}
```


### Java

```
import java.util.Scanner;
import java.net.URL;
import java.net.URLConnection;

public class Client {
    public static String fetch(String url, String userAgent) {
        String content = null;
        URLConnection connection = null;
        try {
            connection =  new URL(url).openConnection();
            connection.setRequestProperty("User-Agent", userAgent);
            Scanner scanner = new Scanner(connection.getInputStream());
            scanner.useDelimiter("\Z");
            content = scanner.next();
            scanner.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        return content;
    }
}
```


### Go

```
package main

import (
    "log"
    "time"
    "io/ioutil"
    "net/http"
)

func fetch(url, userAgent string, timeout time.Duration) []byte {
    req, err := http.NewRequest("GET", url, nil)
    if err != nil {
        log.Fatalln(err)
    }

    req.Header.Set("User-Agent", userAgent)
    client := &http.Client{Timeout: time.Second * timeout}
    resp, err := client.Do(req)
    if err != nil {
        log.Fatalln(err)
    }

    defer resp.Body.Close()
    body, err := ioutil.ReadAll(resp.Body)
    if err != nil {
        log.Fatalln(err)
    }

    return body
}
```




## SQL

### MariaDB (Full-Text search)

```
CREATE TABLE documents (
    id INT UNSIGNED AUTO_INCREMENT NOT NULL PRIMARY KEY,
    content TEXT NOT NULL
) ENGINE=InnoDB;


INSERT INTO documents (id, content) VALUES 
(1, 'Ad primos ictus non corruit ardua quercus'),
(2, 'Adprime in vita esse utile, ut ne quid nimis'),
(3, 'Benefacta male locata malefacta arbitror'),
(4, 'Citius, altius, fortius!'),
(5, 'Damnant, quod non intellegunt'),
(6, 'Hic mortui vivunt, hic muti loquuntur'),
(7, 'Nemo omnia potest scire');


ALTER TABLE documents ADD FULLTEXT (content);


SELECT *
FROM documents
WHERE MATCH (content) AGAINST ('non' IN NATURAL LANGUAGE MODE)
LIMIT 10;
```


### MariaDB (Statistics)

```
CREATE TABLE scores (
    id INT UNSIGNED AUTO_INCREMENT NOT NULL PRIMARY KEY,
    document_id INT UNSIGNED NOT NULL,
    score INT UNSIGNED NOT NULL
) ENGINE=InnoDB;


INSERT INTO scores (document_id, score) VALUES 
(1, 11),
(2, 18),
(2, 22),
(4, 36),
(5, 4),
(2, 6),
(4, 7),
(4, 8),
(5, 11),
(6, 22),
(7, 33),
(3, 46);


SELECT 
    document_id, 
    MIN(score),
    AVG(score),
    MAX(score),
    STD(score)
FROM scores
GROUP BY document_id
ORDER BY document_id DESC;
```


### PostgreSQL (Statistics)

```
CREATE TABLE stars (
    id SERIAL PRIMARY KEY,
    rank INTEGER NOT NULL,
    score INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL
);


INSERT INTO stars VALUES 
(1, 1, 2, '2020-01-01 00:00:00'),
(2, 1, 2, '2020-01-01 00:01:00'),
(3, 1, 3, '2020-01-01 00:02:00'),
(4, 1, 4, '2020-01-01 00:03:00'),
(5, 2, 10, '2020-01-01 01:00:00'),
(6, 2, 20, '2020-01-01 01:01:00'),
(7, 2, 30, '2020-01-01 01:02:00'),
(8, 2, 20, '2020-01-01 01:03:00'),
(9, 2, 10, '2020-01-01 01:04:00');


SELECT 
    rank, 
    MIN(score),
    AVG(score),
    MAX(score)
FROM stars
GROUP BY rank
ORDER BY rank DESC;


SELECT 
    rank, 
    score,
    SUM(score) OVER (PARTITION BY rank) total_scores,
    DENSE_RANK() OVER (ORDER BY score) dense_rank
FROM stars;


SELECT 
    date_trunc('hour', created_at) AS hour, 
    array_agg(rank) AS ranks
FROM stars
GROUP BY hour;
```


### PostgreSQL (REGEXP)

```
CREATE TABLE messages (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL
);


INSERT INTO messages VALUES 
(1, 'L:1 R:5'),
(2, 'L:2 R:4'),
(3, 'L:3 R:3'),
(4, 'L:4 R:2'),
(5, 'L:5 R:1'),
(6, 'L:10 R:22'),
(7, 'L:100 R:23'),
(8, 'L:1000 R:24'),
(9, 'L:10000 R:25');


SELECT
    REGEXP_MATCHES(content, 'L:(\d+) R:\d+') AS a,
    REGEXP_MATCHES(content, 'L:\d+ R:(\d+)') AS b
FROM messages;
```


### PostgreSQL (Nested Sets)

```
CREATE TABLE tree (
    id SERIAL PRIMARY KEY,
    left_key INTEGER NOT NULL,
    right_key INTEGER NOT NULL
);


INSERT INTO tree VALUES
(1, 1, 10),
(2, 2, 9),
(3, 3, 8),
(4, 4, 7),
(5, 5, 6);


SELECT * 
FROM tree 
WHERE left_key >= 3 AND right_key <= 8
ORDER BY left_key;
```


### PostgreSQL (EXPLAIN)

```
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    mode INTEGER NOT NULL,
    content TEXT NOT NULL
);


INSERT INTO documents (mode, content)
SELECT
    round(random()),
    repeat(md5(random()::text), 1024)
FROM generate_series(1, 10000) data(i);


UPDATE documents
SET mode = 5
WHERE id IN (1, 8, 845, 3636, 9899);


EXPLAIN ANALYZE
SELECT id, mode
FROM documents
WHERE mode = 5;


CREATE INDEX idx_documents_mode ON documents(mode);


EXPLAIN ANALYZE
SELECT id, mode
FROM documents
WHERE mode = 5;
```


### PostgreSQL (Many-to-Many relationship)

```
CREATE TABLE tag (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL
);


CREATE TABLE doc (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL
);


CREATE TABLE doc_tag (
    doc_id INT,
    tag_id INT,
    CONSTRAINT doc_tag_pkey PRIMARY KEY (doc_id, tag_id)
);


INSERT INTO tag VALUES
(1, 'A'),
(2, 'B');


INSERT INTO doc VALUES
(1, 'Q'),
(2, 'W'),
(3, 'E');


INSERT INTO doc_tag VALUES
(1, 1),
(1, 2),
(3, 1);


SELECT
    d.id,
    d.content,
    array_agg(t.title)
FROM doc_tag dt
JOIN tag t ON t.id = dt.tag_id
JOIN doc d ON d.id = dt.doc_id
GROUP BY d.id;
```




## Linux commands

### Bash

```
grep -rnw './' -e 'title'
```


```
tar -zcvf archive.tar.gz ./dir
tar -xzvf archive.tar.gz
```


```
gpg --generate-key
gpg --recipient demo --armor --encrypt 1.html
gpg --decrypt 1.html.asc
```


```
gpg --symmetric 2.html
gpg -o 2.html -d 2.html.gpg
```


```
git fetch --all
git checkout -b ID-1
git reset --hard origin/dev
git diff
git status
git add --all
git commit -a -m "ID-1"
git rebase -i origin/dev
git push origin ID-1
```


```
git hash-object info.txt
md5sum info.txt
sha1sum info.txt
```


