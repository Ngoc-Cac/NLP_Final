# Topic Modeling: Clustering documents into topics and application in recommending articles
This is a group final project for NLP. We tackled Topic Modeling problem by using clustering algorithms to cluster similar documents into one topic. Afterwards, we try to apply the clustered results as well as the model to give recommendations to a user's prompt.

## Project's Corpus
The project's corpus is provided in `NLP_Final/Corpus`. We collected news article from [VNExpress](https://vnexpress.net/) within these five topics: .

## Our Pipeline
1. Documents Embedding with TF-IDF: after we have created our corpus, we then create emebddings for the collected articles using TF-IDF. In order to ensure correct tokenization, we have constructed our own word tokenization function. Furthermore, to reduce the dimension of the embeddings, we also choose words that have document frequency higher than 5% and lower than 70% of total documents.
2. Clustering and evaluation: after embedded documents are created, we clustered these documents using KMeans, DBSCAN, HAC and BERTopic. To ensure appropriate hyperparameters have been chosen, we ran multiple clustering with different hyperparameter settings and chose the one with highest silhouette score. After tuning the hyperparameters, we then evaluate the results across models using silhouette score and rand index to compare the result according to their true topic label.
3. Apply results to user recommendation: After having trained models, we then created a recommendation system to recommend the most appropriate articles to a specific user prompts. You can check this out in `NLP_Final/Code/predict.ipynb` or `NLP_Final/Code/gui.py` for a more user friendly interface.

## User Interface
Before running the user interface, make sure that PyQt6 is installed. When running, make sure that you are running from `./Code` or else you will run into an error.

---

### Here are our Contributors:
- Dương Gia Bảo
- Trịnh Ngọc Các
- Nguyễn Thị Ngọc Diệp
- Nguyễn Huy Hoàng