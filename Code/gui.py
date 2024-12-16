import os.path as osp
import joblib
import pandas as pd

# model stuff
from sklearn.metrics import pairwise_distances
from bertopic import BERTopic

# gui stuff
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import PyQt6.QtWidgets as QtWidgets


from typing import Iterable, Optional, Literal


output_dir = osp.join('..', 'Output')

vectoriser = joblib.load(osp.join(output_dir, 'tfidf_vec.pkl'))
kmean_model = joblib.load(osp.join(output_dir, 'KMeans', 'kmean_model.pkl'))
bertopic_model = BERTopic.load(osp.join(output_dir, 'BERTopic', 'model'))

clustered_docs = {'KMeans': pd.read_csv(osp.join(output_dir, 'KMeans', 'clustered.csv'),
                                        encoding='utf-8'),
                  'BERTopic': pd.read_csv(osp.join(output_dir, 'BERTopic', 'clustered.csv'),
                                        encoding='utf-8')}
features = clustered_docs['KMeans'].drop(['cluster', 'link'], axis=1)

def recommend(prompt: str, top_n: int = 5, *,
              model: Literal['BERTopic', 'KMeans'] = 'KMeans') -> Iterable[str]:
    if not isinstance(prompt, str):
        raise ValueError("prompt needs to be str")
    if not isinstance(top_n, int):
        raise ValueError("top_n needs to be an integer")

    temp = vectoriser.transform([prompt])
    promt_df = pd.DataFrame(temp.toarray(), columns=vectoriser.get_feature_names_out())
    
    if model == 'KMeans':
        mask = (clustered_docs[model]['cluster'] == kmean_model.predict(promt_df)[0])
    else:
        mask = (clustered_docs[model]['cluster'] == bertopic_model.transform(prompt)[0][0])
    
    dist_mat = pairwise_distances(promt_df, features[mask], metric='cosine')
    sorted_indices = dist_mat[0].argsort()

    if top_n > len(sorted_indices) or top_n < 1:
        top_n = len(sorted_indices)
    return clustered_docs[model]['link'].iloc[sorted_indices[-1:-(top_n + 1):-1]].values

def format_link(link: str) -> str:
    return f"<a href=\"{link}\">{link}</a>"

TOP_N: int = 100
MODEL_TO_USE: Literal['BERTopic', 'KMeans'] = 'BERTopic'
class Home(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.hbox = QtWidgets.QHBoxLayout()

        self.init_input()

        self.scroller = QtWidgets.QScrollArea()
        self.scroller.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

        self.hbox.addWidget(self.scroller)

        self.setLayout(self.hbox)

    def init_input(self):
        self.input_box = QtWidgets.QTextEdit()
        self.input_box.setFixedHeight(550)

        self.recommend_butt = QtWidgets.QPushButton(text='Recommend Me!')
        self.model_butt = QtWidgets.QPushButton(text='Model: BERTopic')

        self.recommend_butt.clicked.connect(self.handle_prompt)
        self.recommend_butt.setFixedWidth(150)
        self.recommend_butt.setFixedHeight(30)

        self.model_butt.clicked.connect(self.change_model)
        self.model_butt.setFixedWidth(150)
        self.model_butt.setFixedHeight(30)

        vbox = QtWidgets.QVBoxLayout()
        hbox = QtWidgets.QHBoxLayout()

        hbox.addWidget(self.recommend_butt)
        hbox.addWidget(self.model_butt)

        vbox.addLayout(hbox)
        vbox.addWidget(self.input_box)
        vbox.setAlignment(hbox, Qt.AlignmentFlag.AlignCenter)

        self.hbox.addLayout(vbox)

    def change_model(self):
        global MODEL_TO_USE
        if MODEL_TO_USE == 'BERTopic':
            self.model_butt.setText('Model: KMeans')
            MODEL_TO_USE = 'KMeans'
        else:
            self.model_butt.setText('Model: BERTopic')
            MODEL_TO_USE = 'BERTopic'

    def handle_prompt(self):
        self.recommend_butt.setDisabled(True)
        prompt = self.input_box.toPlainText()

        vbox = QtWidgets.QVBoxLayout()
        for link in  recommend(prompt, TOP_N, model=MODEL_TO_USE):
            label = QtWidgets.QLabel(format_link(link))
            label.setOpenExternalLinks(True)
            vbox.addWidget(label)
        
        temp_wid = QtWidgets.QWidget()
        temp_wid.setLayout(vbox)
        self.scroller.setWidget(temp_wid)

        self.input_box.clear()
        self.recommend_butt.setDisabled(False)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Topic Modeling - VNExpress Article Recommender")
        self.setGeometry(350, 100, 800, 600)
        self.setCentralWidget(Home())



def main():
    ult_tictactoe = QtWidgets.QApplication([])
    root = MainWindow()
    root.show()
    ult_tictactoe.exec()

if __name__=="__main__":
    main()