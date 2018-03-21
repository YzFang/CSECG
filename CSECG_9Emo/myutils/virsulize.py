#coding:utf-8
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib

matplotlib.use('qt4agg')  
#指定默认字体  
matplotlib.rcParams['font.sans-serif'] = ['SimHei']   
matplotlib.rcParams['font.family']='sans-serif'  
#解决负号'-'显示为方块的问题  
matplotlib.rcParams['axes.unicode_minus'] = False  

#可视化词向量
def plot_with_labels(low_dim_embs,labels,filename='tsne.png'):
    assert low_dim_embs.shape[0]>=len(labels), "More labels than embeddings"
    plt.figure(figsize=(18,18))
    for i,label in enumerate(labels):
        x,y=low_dim_embs[i,:]
        plt.scatter(x,y)
        plt.annotate(label,
                     xy=(x,y),
                     xytext=(5,2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)
def visualize_vectors(embeddings,words):
    tsne=TSNE(perplexity=30,n_components=2,init='pca',n_iter=5000)
    plot_only=500
    #降维
    low_dim_embs=tsne.fit_transform(embeddings[:plot_only,:])
    labels=words[:plot_only]
    plot_with_labels(low_dim_embs,labels)